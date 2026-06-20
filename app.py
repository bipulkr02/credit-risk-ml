"""
app.py
------
Streamlit web app for the Credit Risk Analysis project.

Lets a user enter an applicant's details through a simple form and get:
- Probability of default
- A credit score (300-900 scale)
- A risk rating (Poor / Average / Good / Excellent)

Run locally with:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Credit Risk Prediction", page_icon="🏦", layout="centered")


# ---------------------------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")
    return model, scaler, encoders


def engineer_features_single(row: dict) -> pd.DataFrame:
    """Mirror the feature engineering done in src/data_preprocessing.py for a single record."""
    df = pd.DataFrame([row])

    df["credit_to_income_ratio"] = df["credit_amount"] / (df["monthly_income"] + 1)
    df["est_monthly_installment"] = df["credit_amount"] / df["duration_months"]
    df["installment_to_income_ratio"] = df["est_monthly_installment"] / (df["monthly_income"] + 1)
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"]
    )
    df["employment_stability"] = df["employment_years"] / (df["age"] - 17)

    return df


def predict_risk(row: dict, model, scaler, encoders):
    df = engineer_features_single(row)

    # apply the same label encoders fit during training
    for col, le in encoders.items():
        df[col] = le.transform(df[col].astype(str))

    # column order must match training
    feature_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else df.columns.tolist()
    df = df[feature_cols]

    X_scaled = scaler.transform(df)
    default_prob = model.predict_proba(X_scaled)[0][1]

    # Map probability of default -> a 300-900 credit score (higher prob => lower score)
    credit_score = int(900 - default_prob * 600)
    credit_score = max(300, min(900, credit_score))

    if credit_score >= 750:
        rating = "Excellent"
    elif credit_score >= 650:
        rating = "Good"
    elif credit_score >= 500:
        rating = "Average"
    else:
        rating = "Poor"

    return default_prob, credit_score, rating


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏦 Credit Risk Prediction System")
st.write(
    "Enter an applicant's details below to estimate their probability of "
    "loan default, a derived credit score, and a risk rating."
)

try:
    model, scaler, encoders = load_artifacts()
except FileNotFoundError:
    st.error(
        "No trained model found in `models/`. Run `python main.py` first "
        "to generate the dataset and train the model."
    )
    st.stop()

with st.form("applicant_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        job_level = st.selectbox(
            "Job Level", options=[0, 1, 2, 3],
            format_func=lambda x: ["Unskilled", "Skilled", "Highly Skilled", "Management"][x]
        )
        housing = st.selectbox("Housing", options=["own", "rent", "free"])
        saving_accounts = st.selectbox("Savings Account", options=["little", "moderate", "rich", "none"])
        checking_account = st.selectbox("Checking Account", options=["little", "moderate", "rich", "none"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=5000, max_value=500000, value=45000, step=1000)

    with col2:
        credit_amount = st.number_input("Loan / Credit Amount (₹)", min_value=1000, max_value=2000000, value=150000, step=1000)
        duration_months = st.selectbox("Loan Duration (months)", options=[6, 12, 18, 24, 36, 48, 60], index=3)
        purpose = st.selectbox("Loan Purpose", options=["car", "furniture", "education", "business", "appliances", "repairs"])
        employment_years = st.number_input("Years of Employment", min_value=0, max_value=50, value=5)
        existing_credits = st.number_input("Existing Credit Lines", min_value=1, max_value=10, value=1)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("Predict Credit Risk")

if submitted:
    applicant = {
        "age": age,
        "job_level": job_level,
        "housing": housing,
        "saving_accounts": saving_accounts,
        "checking_account": checking_account,
        "credit_amount": credit_amount,
        "duration_months": duration_months,
        "purpose": purpose,
        "employment_years": employment_years,
        "existing_credits": existing_credits,
        "dependents": dependents,
        "monthly_income": monthly_income,
    }

    default_prob, credit_score, rating = predict_risk(applicant, model, scaler, encoders)

    st.subheader("📊 Prediction Result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Default Probability", f"{default_prob*100:.1f}%")
    m2.metric("Credit Score", credit_score)
    m3.metric("Risk Rating", rating)

    rating_colors = {"Excellent": "🟢", "Good": "🟡", "Average": "🟠", "Poor": "🔴"}
    st.write(f"{rating_colors.get(rating, '')} Overall risk rating: **{rating}**")

    st.progress(min(max(default_prob, 0.0), 1.0))

st.markdown("---")
st.caption(
    "Built with scikit-learn / XGBoost + Streamlit. "
    "This is a demo model trained on synthetic data — not financial advice."
)
