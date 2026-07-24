import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

st.title("💳 Credit Card Fraud Detection")

# Load Dataset
df = pd.read_csv("creditcard.csv")

# ------------------ Graph ------------------
st.subheader("Fraud vs Normal Transactions")

fig, ax = plt.subplots()
df["Class"].value_counts().plot(kind="bar", color=["green", "red"], ax=ax)
ax.set_xticklabels(["Normal", "Fraud"], rotation=0)
st.pyplot(fig)

# ------------------ Prepare Data ------------------
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ Train Model ------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------ Accuracy ------------------
pred = model.predict(X_test)

st.success(f"Accuracy: {accuracy_score(y_test, pred)*100:.2f}%")

# ------------------ Heatmap ------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, pred)

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ------------------ Prediction ------------------
st.subheader("Predict Transaction")

row = st.number_input(
    "Enter Row Number (0 - {})".format(len(df)-1),
    min_value=0,
    max_value=len(df)-1,
    value=0
)

if st.button("Predict"):

    sample = X.iloc[[row]]

    result = model.predict(sample)[0]

    if result == 0:
        st.success("✅ Normal Transaction")
    else:
        st.error("🚨 Fraudulent Transaction")