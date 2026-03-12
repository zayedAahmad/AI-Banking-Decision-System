import streamlit as st
import pandas as pd
import joblib

# Load model and helpers
model = joblib.load("models/loan_approval_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

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
    st.subheader(" Applicant Information")

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
    input_data = pd.DataFrame([{
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
    }])

    # Encode categorical columns
    for col in input_data.columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Reorder columns to match training
    input_data = input_data[model_columns]

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

    with result_col2:
        risk_score = 1 - prediction_proba[1]
        if risk_score < 0.30:
            st.success(f"Risk Score: {risk_score:.2%}  |  Low Risk")
        elif risk_score < 0.60:
            st.warning(f"Risk Score: {risk_score:.2%}  |  Medium Risk")
        else:
            st.error(f"Risk Score: {risk_score:.2%}  |  High Risk")

    st.divider()

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("Approval Confidence", f"{prediction_proba[1]:.2%}")

    with metric_col2:
        st.metric("Rejection Confidence", f"{prediction_proba[0]:.2%}")

    st.divider()

    st.subheader(" Input Summary")

    summary_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self Employed": self_employed,
        "Applicant Income": applicant_income,
        "Coapplicant Income": coapplicant_income,
        "Loan Amount": loan_amount,
        "Loan Amount Term": loan_amount_term,
        "Credit History": credit_history,
        "Property Area": property_area
    }])

    st.dataframe(summary_data, use_container_width=True)

    st.divider()

    st.subheader("Quick Interpretation")

    insights = []

    if credit_history == 1.0:
        insights.append("- Strong credit history improves approval chances.")
    else:
        insights.append("- Weak or missing credit history reduces approval chances.")

    if applicant_income >= 5000:
        insights.append("- Higher applicant income may support repayment ability.")
    else:
        insights.append("- Lower applicant income may reduce repayment capacity.")

    if coapplicant_income > 0:
        insights.append("- Coapplicant income may strengthen the application.")

    if loan_amount > 200:
        insights.append("- Higher loan amount may increase lending risk.")

    if education == "Graduate":
        insights.append("- Graduate education may correlate with stronger applicant profiles.")

    st.markdown("\n".join(insights))