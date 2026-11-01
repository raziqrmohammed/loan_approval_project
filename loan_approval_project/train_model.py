import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("loan_data.csv")

# 2. Clean the dataset
df = df.drop_duplicates()
df = df.dropna()

features = [
    "Income",
    "Loan_Amount",
    "Credit_History",
    "Education"
]

X = df[features]
y = df["Loan_Status"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Create and train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# 5. Test the model
predictions = model.predict(X_test)

print("Accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 6. Save the trained model and feature names
with open("loan_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as loan_model.pkl")