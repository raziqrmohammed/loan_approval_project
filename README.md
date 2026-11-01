# Loan Approval Prediction System

A machine learning project that predicts whether a loan application may be approved based on an applicant’s income, loan amount, credit history, and education.

## Features

- Predicts an approval or rejection using a trained logistic regression model.
- Displays the model’s confidence in its prediction.
- Provides an interactive web interface built with Streamlit.
- Includes a script to clean the data, train the model, and report evaluation metrics.

## Project Files

- `app.py` — Streamlit application for entering applicant details and viewing predictions.
- `train_model.py` — Cleans the dataset, trains and evaluates the model, and saves it as `loan_model.pkl`.
- `loan_data.csv` — Dataset used to train the model.
- `loan_model.pkl` — Saved trained model.
- `requirements.txt` — Python dependencies.

## Setup

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
