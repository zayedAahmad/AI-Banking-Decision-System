import joblib


def load_artifacts():
    model = joblib.load("models/loan_approval_model.pkl")
    model_columns = joblib.load("models/model_columns.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    return model, model_columns, label_encoders