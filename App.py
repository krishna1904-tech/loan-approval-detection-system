import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

st.title("🏦 Loan Approval Prediction System")
st.write("Machine Learning Project using Random Forest")

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("Dataset Shape :", df.shape)

# ----------------------------
# Missing Values
# ----------------------------

df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)

df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

# ----------------------------
# Label Encoding
# ----------------------------

le = LabelEncoder()

columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in columns:
    df[col] = le.fit_transform(df[col])

# Remove Loan_ID

df.drop("Loan_ID", axis=1, inplace=True)

st.success("Data Preprocessing Completed")

# ============================
# Train Machine Learning Model
# ============================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ============================
# Model Accuracy
# ============================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

st.subheader("Model Accuracy")

st.success(f"Accuracy : {accuracy*100:.2f}%")

# ============================
# Confusion Matrix
# ============================

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, prediction)

fig, ax = plt.subplots(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected","Approved"],
    yticklabels=["Rejected","Approved"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ============================
# Loan Approval Distribution
# ============================

st.subheader("Loan Approval Distribution")

fig, ax = plt.subplots()

df["Loan_Status"].value_counts().plot(
    kind="bar",
    color=["red","green"],
    ax=ax
)

ax.set_xticklabels(
    ["Rejected","Approved"],
    rotation=0
)

st.pyplot(fig)

# ============================
# Correlation Heatmap
# ============================

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# ============================
# Loan Prediction Form
# ============================

st.header("🏦 Predict Loan Approval")

st.write("Enter Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])

married = st.selectbox("Married", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

education = st.selectbox("Education", ["Graduate", "Not Graduate"])

self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=1,
    value=120
)

loan_term = st.number_input(
    "Loan Amount Term",
    min_value=12,
    value=360
)

credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Loan Status"):

    gender = 1 if gender == "Male" else 0

    married = 1 if married == "Yes" else 0

    education = 0 if education == "Graduate" else 1

    self_employed = 1 if self_employed == "Yes" else 0

    dep = {"0":0, "1":1, "2":2, "3+":3}
    dependents = dep[dependents]

    area = {
        "Rural":0,
        "Semiurban":1,
        "Urban":2
    }

    property_area = area[property_area]

    user_data = pd.DataFrame({
        "Gender":[gender],
        "Married":[married],
        "Dependents":[dependents],
        "Education":[education],
        "Self_Employed":[self_employed],
        "ApplicantIncome":[applicant_income],
        "CoapplicantIncome":[coapplicant_income],
        "LoanAmount":[loan_amount],
        "Loan_Amount_Term":[loan_term],
        "Credit_History":[credit_history],
        "Property_Area":[property_area]
    })

    result = model.predict(user_data)[0]

    probability = model.predict_proba(user_data)[0]

    st.subheader("Prediction Result")

    if result == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write("Approval Probability")

    st.progress(float(max(probability)))

    st.write(f"Rejected : {probability[0]*100:.2f}%")
    st.write(f"Approved : {probability[1]*100:.2f}%")
