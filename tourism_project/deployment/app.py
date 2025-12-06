import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# HF model repo where the trained model is uploaded
MODEL_REPO_ID = "vaishaliagarwal/tourism-prediction-mlops-model"
MODEL_FILENAME = "best_tourism_model_xgb_v1.joblib"

# Download and load the model from HF Hub
model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Prediction")

st.write(
    """
This application predicts the probability that a customer will purchase
the **Wellness Tourism Package** based on their profile and interactions.
Fill the details below and click **Predict**.
"""
)

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=80, value=30)
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Free Lancer", "Small Business", "Large Business", "Other"],
    )
    Designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    )

with col2:
    NumberOfPersonVisiting = st.number_input(
        "Number Of Person Visiting", min_value=1, max_value=10, value=2
    )
    PreferredPropertyStar = st.slider(
        "Preferred Property Star", min_value=1, max_value=5, value=3
    )
    NumberOfTrips = st.number_input(
        "Number Of Trips per Year", min_value=0, max_value=20, value=2
    )
    NumberOfChildrenVisiting = st.number_input(
        "Number Of Children Visiting (below 5)", min_value=0, max_value=5, value=0
    )
    Passport = st.selectbox("Passport (0 = No, 1 = Yes)", [0, 1])
    OwnCar = st.selectbox("Own Car (0 = No, 1 = Yes)", [0, 1])

st.subheader("Interaction & Pitch Details")

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
)
PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score", min_value=1, max_value=5, value=3
)
NumberOfFollowups = st.number_input(
    "Number Of Followups", min_value=0, max_value=20, value=2
)
DurationOfPitch = st.number_input(
    "Duration Of Pitch (minutes)", min_value=0, max_value=120, value=20
)
MonthlyIncome = st.number_input(
    "Monthly Income", min_value=0, max_value=1_000_000, value=50_000, step=5_000
)

if st.button("Predict"):
    input_dict = {
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "ProductPitched": ProductPitched,
        "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch,
    }

    df = pd.DataFrame([input_dict])
    proba = model.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)

    st.markdown(f"### Predicted Probability of Purchase: **{proba:.2f}**")
    if pred == 1:
        st.success("Model prediction: Customer is **likely to purchase** the package.")
    else:
        st.warning("Model prediction: Customer is **unlikely to purchase** the package.")
