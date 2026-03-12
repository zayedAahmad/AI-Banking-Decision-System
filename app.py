import streamlit as st
import pandas as pd

from utils.model_loader import load_artifacts
from utils.preprocessing import prepare_input_data
from utils.interpretation import generate_insights, get_risk_label

# Load model files
model, model_columns, label_encoders = load_artifacts()

st.set_page_config(
    page_title="AI Loan Prediction & Risk Analysis System",
    
    layout="wide"
)

st.title("AI Loan Prediction & Risk Analysis System")
st.markdown(
    "An AI-powered system for predicting **loan approval status** using customer financial and demographic data."
)

st.divider()

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Applicant Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

with right_col:
    st.subheader("Loan Information")

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0,
        step=100.0
    )
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=120.0,
        step=1.0
    )
    loan_amount_term = st.number_input(
        "Loan Amount Term",
        min_value=1.0,
        value=360.0,
        step=1.0
    )
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.divider()

if st.button("Predict Loan Status", use_container_width=True):
    raw_input = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    input_data = prepare_input_data(raw_input, label_encoders, model_columns)

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    approval_confidence = prediction_proba[1]
    rejection_confidence = prediction_proba[0]
    risk_score = 1 - approval_confidence
    risk_label = get_risk_label(risk_score)

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

    with result_col2:
        if risk_label == "Low Risk":
            st.success(f"Risk Score: {risk_score:.2%} | {risk_label}")
        elif risk_label == "Medium Risk":
            st.warning(f"Risk Score: {risk_score:.2%} | {risk_label}")
        else:
            st.error(f"Risk Score: {risk_score:.2%} | {risk_label}")

    st.divider()

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("Approval Confidence", f"{approval_confidence:.2%}")

    with metric_col2:
        st.metric("Rejection Confidence", f"{rejection_confidence:.2%}")

    st.divider()

    st.subheader("Input Summary")
    st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)

    st.divider()

    st.subheader("Quick Interpretation")
    insights = generate_insights(
        credit_history=credit_history,
        applicant_income=applicant_income,
        coapplicant_income=coapplicant_income,
        loan_amount=loan_amount,
        education=education
    )
    st.markdown("\n".join(insights))