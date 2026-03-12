import streamlit as st
import pandas as pd
import joblib

# Load model and helpers
model = joblib.load("models/loan_approval_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

st.set_page_config(page_title="AI Loan Prediction System", layout="wide")

st.title("AI Loan Prediction System")
st.write("Predict loan approval using machine learning.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0.0, value=5000.0)

with col2:
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    LoanAmount = st.number_input("Loan Amount", min_value=0.0, value=120.0)
    Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=1.0, value=360.0)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.divider()

if st.button("Predict Loan Status"):
    input_data = pd.DataFrame([{
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }])

    # Encode categorical columns
    for col in input_data.columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

    input_data = input_data[model_columns]

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("AI Decision")

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

    st.metric("Approval Confidence", f"{prediction_proba[1]:.2%}")
    st.metric("Rejection Confidence", f"{prediction_proba[0]:.2%}")

    st.subheader("Input Summary")
    st.dataframe(input_data)