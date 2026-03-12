def get_risk_label(risk_score):
    if risk_score < 0.30:
        return "Low Risk"
    elif risk_score < 0.60:
        return "Medium Risk"
    return "High Risk"


def generate_insights(credit_history, applicant_income, coapplicant_income, loan_amount, education):
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

    return insights