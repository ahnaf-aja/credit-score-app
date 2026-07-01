import streamlit as st
import pandas as pd
from inference import CreditScorePredictor, TEST_CASES
from pipeline import DataPreprocessor

st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="centered")


@st.cache_resource
def load_predictor():
    return CreditScorePredictor()


predictor = load_predictor()
LABEL_COLOR = {"Good": "🟢", "Standard": "🟡", "Poor": "🔴"}

st.title("💳 Credit Score Classification")
st.caption("DTSC6012001 - Model Deployment | NIM 2802394443 | "
           f"Model terbaik: **{predictor.metadata['best_model']}**")

# quick-fill test case
st.subheader("Input Data Nasabah")
preset = st.selectbox("Isi cepat dengan test case (opsional):",
                      ["— manual —"] + list(TEST_CASES.keys()))
defaults = TEST_CASES[preset] if preset in TEST_CASES else {}


def d(key, fallback):
    return defaults.get(key, fallback)


col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 14, 100, int(d("Age", 30)))
    occupation = st.selectbox("Occupation", [
        "Engineer", "Teacher", "Mechanic", "Developer", "Scientist", "Lawyer",
        "Architect", "Accountant", "Entrepreneur", "Doctor", "Journalist",
        "Manager", "Media_Manager", "Musician", "Writer"],
        index=0)
    annual_income = st.number_input("Annual Income", 0.0, 1e7, float(d("Annual_Income", 45000)))
    monthly_salary = st.number_input("Monthly Inhand Salary", 0.0, 1e6, float(d("Monthly_Inhand_Salary", 3600)))
    num_bank = st.number_input("Num Bank Accounts", 0, 20, int(d("Num_Bank_Accounts", 4)))
    num_card = st.number_input("Num Credit Card", 0, 15, int(d("Num_Credit_Card", 4)))
    interest = st.number_input("Interest Rate", 0, 50, int(d("Interest_Rate", 12)))
    num_loan = st.number_input("Num of Loan", 0, 15, int(d("Num_of_Loan", 2)))
    type_loan = st.text_input("Type of Loan", str(d("Type_of_Loan", "Personal Loan")))
    delay = st.number_input("Delay from due date", -5, 100, int(d("Delay_from_due_date", 10)))
    num_delayed = st.number_input("Num of Delayed Payment", 0, 60, int(d("Num_of_Delayed_Payment", 8)))
with col2:
    changed_limit = st.number_input("Changed Credit Limit", -50.0, 50.0, float(d("Changed_Credit_Limit", 10.0)))
    inquiries = st.number_input("Num Credit Inquiries", 0, 50, int(d("Num_Credit_Inquiries", 4)))
    credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"],
                              index=["Good", "Standard", "Bad"].index(d("Credit_Mix", "Standard")))
    debt = st.number_input("Outstanding Debt", 0.0, 1e5, float(d("Outstanding_Debt", 900)))
    util = st.number_input("Credit Utilization Ratio", 0.0, 100.0, float(d("Credit_Utilization_Ratio", 33)))
    history = st.text_input("Credit History Age", str(d("Credit_History_Age", "18 Years and 5 Months")))
    pay_min = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"],
                           index=["Yes", "No", "NM"].index(d("Payment_of_Min_Amount", "No")))
    emi = st.number_input("Total EMI per month", 0.0, 1e5, float(d("Total_EMI_per_month", 150)))
    invested = st.number_input("Amount invested monthly", 0.0, 1e5, float(d("Amount_invested_monthly", 200)))
    behaviour = st.selectbox("Payment Behaviour", [
        "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
        "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
        "High_spent_Medium_value_payments", "High_spent_Large_value_payments"],
        index=0)
    balance = st.number_input("Monthly Balance", 0.0, 1e6, float(d("Monthly_Balance", 350)))
    month = st.selectbox("Month", ["January", "February", "March", "April", "May",
                                   "June", "July", "August"], index=2)

record = {
    "Age": age, "Occupation": occupation, "Annual_Income": annual_income,
    "Monthly_Inhand_Salary": monthly_salary, "Num_Bank_Accounts": num_bank,
    "Num_Credit_Card": num_card, "Interest_Rate": interest, "Num_of_Loan": num_loan,
    "Type_of_Loan": type_loan, "Delay_from_due_date": delay,
    "Num_of_Delayed_Payment": num_delayed, "Changed_Credit_Limit": changed_limit,
    "Num_Credit_Inquiries": inquiries, "Credit_Mix": credit_mix,
    "Outstanding_Debt": debt, "Credit_Utilization_Ratio": util,
    "Credit_History_Age": history, "Payment_of_Min_Amount": pay_min,
    "Total_EMI_per_month": emi, "Amount_invested_monthly": invested,
    "Payment_Behaviour": behaviour, "Monthly_Balance": balance, "Month": month,
}

if st.button("🔮 Prediksi Credit Score", type="primary", use_container_width=True):
    res = predictor.predict_one(record)
    label = res["credit_score"]
    st.markdown(f"## {LABEL_COLOR.get(label, '')} Prediksi: **{label}**")
    st.metric("Confidence", f"{res['confidence']*100:.1f}%")
    st.write("Probabilitas tiap kelas:")
    st.bar_chart(pd.Series(res["probabilities"]))
    st.json(res)

with st.expander("ℹ️ Performa model (test set)"):
    st.json(predictor.metadata["metrics"])
