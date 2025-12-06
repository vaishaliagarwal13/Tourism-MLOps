import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score,
)
import xgboost as xgb

import mlflow
import mlflow.sklearn

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# ---------------------------------------------------
# HF + paths configuration
# ---------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

DATASET_REPO_ID = "vaishaliagarwal/tourism-prediction-mlops"   
BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main"

# DATASET_REPO_ID = "vaishaliagarwal/tourism-prediction-mlops"       # dataset repo
MODEL_REPO_ID = "vaishaliagarwal/tourism-prediction-mlops-model"  # model repo to create/use
MODEL_FILENAME = "best_tourism_model_xgb_v1.joblib"


# ---------------------------------------------------
# Data loading helper
# ---------------------------------------------------
def load_splits_from_hf():
    """
    Load Xtrain, Xtest, ytrain, ytest from the HF dataset using HTTP URLs.
    """
    Xtrain = pd.read_csv(f"{BASE_URL}/Xtrain.csv")
    Xtest = pd.read_csv(f"{BASE_URL}/Xtest.csv")
    ytrain = pd.read_csv(f"{BASE_URL}/ytrain.csv")
    ytest = pd.read_csv(f"{BASE_URL}/ytest.csv")

    # Ensure ytrain, ytest are 1D Series
    if isinstance(ytrain, pd.DataFrame):
        ytrain = ytrain.iloc[:, 0]
    if isinstance(ytest, pd.DataFrame):
        ytest = ytest.iloc[:, 0]

    return Xtrain, Xtest, ytrain, ytest


# ---------------------------------------------------
# Pipeline definition
# ---------------------------------------------------
def build_preprocessor():
    numeric_features = [
        "Age",
        "NumberOfPersonVisiting",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "NumberOfChildrenVisiting",
        "DurationOfPitch",
        "NumberOfFollowups",
        "MonthlyIncome",
        "PitchSatisfactionScore",
    ]

    categorical_features = [
        "TypeofContact",
        "CityTier",
        "Occupation",
        "Gender",
        "MaritalStatus",
        "Designation",
        "ProductPitched",
        "Passport",
        "OwnCar",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_and_log():
    # You can point MLflow to a server if you have one; otherwise it logs locally.
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("MLOps_experiment")

    Xtrain, Xtest, ytrain, ytest = load_splits_from_hf()

    # Compute scale_pos_weight for imbalance (0 = no, 1 = yes)
    class_counts = ytrain.value_counts()
    # guard: avoid division by zero
    if 0 in class_counts and 1 in class_counts and class_counts[1] > 0:
        scale_pos_weight = class_counts[0] / class_counts[1]
    else:
        scale_pos_weight = 1.0

    preprocessor = build_preprocessor()

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
    )

    # pipeline: preprocessor + XGB
    model_pipeline = make_pipeline(preprocessor, xgb_model)

    # hyperparameter grid (note: step name is 'xgbclassifier' in make_pipeline)
    param_grid = {
        "xgbclassifier__n_estimators": [50, 75, 100],
        "xgbclassifier__max_depth": [2, 3, 4],
        "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
        "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
        "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
        "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
    }

    # main MLflow run
    with mlflow.start_run(run_name="xgb_tourism_gridsearch") as parent_run:
        grid_search = GridSearchCV(
            model_pipeline,
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring="f1",  # or "recall", "roc_auc", etc.
        )

        grid_search.fit(Xtrain, ytrain)

        # log each parameter combination as nested run
        results = grid_search.cv_results_
        for i in range(len(results["params"])):
            param_set = results["params"][i]
            mean_score = results["mean_test_score"][i]
            std_score = results["std_test_score"][i]

            with mlflow.start_run(run_name=f"child_run_{i}", nested=True):
                mlflow.log_params(param_set)
                mlflow.log_metric("mean_test_score", mean_score)
                mlflow.log_metric("std_test_score", std_score)

        # log best parameters in parent run
        mlflow.log_params(grid_search.best_params_)

        best_model = grid_search.best_estimator_

        # classification threshold (like sample)
        classification_threshold = 0.45

        # train metrics
        y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
        y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

        # test metrics
        y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
        y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

        train_report = classification_report(ytrain, y_pred_train, output_dict=True)
        test_report = classification_report(ytest, y_pred_test, output_dict=True)

        mlflow.log_metrics(
            {
                "train_accuracy": train_report["accuracy"],
                "train_precision": train_report["1"]["precision"],
                "train_recall": train_report["1"]["recall"],
                "train_f1-score": train_report["1"]["f1-score"],
                "test_accuracy": test_report["accuracy"],
                "test_precision": test_report["1"]["precision"],
                "test_recall": test_report["1"]["recall"],
                "test_f1-score": test_report["1"]["f1-score"],
            }
        )

        print("Train classification report:\n", classification_report(ytrain, y_pred_train))
        print("Test classification report:\n", classification_report(ytest, y_pred_test))

        # Save best model locally
        os.makedirs("tourism_project/model_building", exist_ok=True)
        local_model_path = os.path.join(
            "tourism_project/model_building", MODEL_FILENAME
        )
        joblib.dump(best_model, local_model_path)
        print("Best model saved to:", local_model_path)

        # Ensure HF model repo exists
        try:
            api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
            print(f"Model repo '{MODEL_REPO_ID}' already exists.")
        except RepositoryNotFoundError:
            print(f"Model repo '{MODEL_REPO_ID}' not found. Creating it...")
            create_repo(
                repo_id=MODEL_REPO_ID,
                repo_type="model",
                private=False,
                token=HF_TOKEN,
            )
            print(f"Model repo '{MODEL_REPO_ID}' created.")

        # Upload model file to HF model repo
        api.upload_file(
            path_or_fileobj=local_model_path,
            path_in_repo=MODEL_FILENAME,
            repo_id=MODEL_REPO_ID,
            repo_type="model",
        )
        print(f"Model uploaded to HF model repo: {MODEL_REPO_ID}")


if __name__ == "__main__":
    train_and_log()
