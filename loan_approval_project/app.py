import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.write(
    "Enter applicant information to view the "
    "machine learning model's prediction."
)

model_path = Path(__file__).with_name("loan_model.pkl")

if not model_path.exists():
    st.error(
        "Model not found. Run train_model.py first."
    )
    st.stop()

with open(model_path, "rb") as file:
    model = pickle.load(file)

with st.form("loan_form"):
    st.subheader("Applicant Details")

    income = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        value=600000,
        step=10000
    )

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=0,
        value=200000,
        step=10000
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[1, 0],
        format_func=lambda x: (
            "Positive history" if x == 1
            else "Negative history"
        )
    )

    education = st.selectbox(
        "Education",
        options=[1, 0],
        format_func=lambda x: (
            "Graduate" if x == 1
            else "Not Graduate"
        )
    )

    submitted = st.form_submit_button(
        "Predict Loan Status"
    )

if submitted:
    applicant = pd.DataFrame([{
        "Income": income,
        "Loan_Amount": loan_amount,
        "Credit_History": credit_history,
        "Education": education
    }])

    prediction = int(model.predict(applicant)[0])
    probabilities = model.predict_proba(applicant)[0]
    confidence = max(probabilities) * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Prediction: Approved")
    else:
        st.warning("Prediction: Rejected")

    st.metric(
        "Model confidence in its predicted class",
        f"{confidence:.2f}%"
    )

    st.caption(
        "Educational demonstration only. "
        "This prediction is not a real loan decision "
        "or a guarantee of approval."
    )