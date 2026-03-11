#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os 
import joblib

# Load the dataset
df = pd.read_csv('bank_loan_dataset.csv')

# feature and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/loan_approval_model.pkl')

print("\nModel trained and saved successfully!")
# %%
