# Loan Approval Prediction System

A machine learning project that predicts whether a loan application may be approved based on applicant income, requested loan amount, credit history, and education. It includes a Streamlit web app, a model training script, and a sample dataset.

> **Educational use only:** This project is a demonstration. Its predictions are not real loan decisions and do not guarantee approval.

## Features

- Interactive web interface built with Streamlit
- Logistic regression model with standardized input features
- Displays an approval/rejection prediction and the model's confidence in its predicted class
- Training script reports accuracy, a classification report, and a confusion matrix

## Project files

- `app.py` — Streamlit application
- `train_model.py` — cleans the data, trains and evaluates the model, and saves it
- `loan_data.csv` — sample training data
- `loan_model.pkl` — saved trained model used by the app
- `requirements.txt` — Python package dependencies

## Requirements

- Python 3.8 or later
- pip

## Setup

Open a terminal in the project directory and create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

To train the model from `loan_data.csv` and generate/update `loan_model.pkl`, run:

```bash
python train_model.py
```

The script removes duplicate and incomplete rows, splits the data into training and test sets, trains a standardized logistic regression pipeline, and prints evaluation results.

## Run the app

Start the Streamlit app from the project directory:

```bash
streamlit run app.py
```

Enter the applicant details in the page and select **Predict Loan Status** to see the model's prediction and confidence.

## Input data

The CSV file is expected to include these columns:

| Column | Description |
|---|---|
| `Income` | Applicant's annual income in rupees |
| `Loan_Amount` | Requested loan amount in rupees |
| `Credit_History` | Credit history indicator (`1` = positive, `0` = negative) |
| `Education` | Education indicator (`1` = graduate, `0` = not graduate) |
| `Loan_Status` | Target label (`1` = approved, `0` = rejected) |

The model is only as reliable as its training data. The included dataset is intended for demonstration, not real-world lending decisions.

## License

No license has been specified yet. Add a `LICENSE` file if you want to make the project's reuse terms explicit.
