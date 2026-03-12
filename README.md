# AI Loan Prediction & Risk Analysis System

This project is a machine learning web app that predicts whether a loan application is likely to be approved or rejected based on customer and loan information.

I built this project to practice applying machine learning to a real financial use case and to turn the model into an interactive web application that anyone can test online.

## Live App
[Try the app here](https://ai-banking-decision-system-zayed.streamlit.app/)

## What this project does
The app takes user input such as:
- gender
- marital status
- dependents
- education
- self employment status
- applicant income
- coapplicant income
- loan amount
- loan term
- credit history
- property area

Then it uses a trained machine learning model to predict:
- whether the loan is approved or rejected
- the confidence of approval
- the confidence of rejection

## Why I built it
I wanted to build a project that is practical, interactive, and related to real-world financial decision making. Instead of stopping at training a model in Python, I wanted to deploy it as a live app and make it usable.

## Tools and technologies
- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## How it works
1. The dataset is loaded and cleaned
2. Categorical values are encoded
3. A machine learning model is trained
4. The trained model is saved
5. The app loads the model and predicts the result from user input

## Screen
![App Screenshot](images/app.png)

## Run the project locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py