import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/loan_approval_model.pkl")

# Page settings
st.set_page_config(page_title="AI Banking Decision System", layout="centered")

# Title
st.title("AI Banking Decision System")
st.write("Enter customer financial data to predict loan approval.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income", min_value=0.0, value=3000.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
employment_years = st.number_input("Employment Years", min_value=0.0, value=3.0)
existing_debt = st.number_input("Existing Debt", min_value=0.0, value=2000.0)
loan_term = st.number_input("Loan Term (months)", min_value=1, value=24)
risk_level = st.slider("Risk Level", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

# Prediction button
if st.button("Predict Decision"):
    input_data = pd.DataFrame([{
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "existing_debt": existing_debt,
        "loan_term": loan_term,
        "risk_level": risk_level
    }])

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("Loan Decision: Approved")
    else:
        st.error("Loan Decision: Rejected")

    st.write(f"Approval Confidence: {prediction_proba[1]:.2%}")
    st.write(f"Rejection Confidence: {prediction_proba[0]:.2%}")
# %%
