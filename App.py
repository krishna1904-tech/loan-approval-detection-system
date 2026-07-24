import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

st.title("💳 Credit Card Fraud Detection")

import kagglehub
import pandas as pd
import os

# Download dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# Load CSV
df = pd.read_csv(os.path.join(path, "creditcard.csv"))

# ------------------ Prepare Data ------------------
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
st.header("🔍 Fraud Detection")

row = st.number_input(
    "Enter Transaction Row Number",
    min_value=0,
    max_value=len(df)-1,
    value=0
)

if st.button("Detect Fraud"):

    transaction = X.iloc[[row]]

    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0]

    if prediction == 0:
        st.success("✅ Normal Transaction")
    else:
        st.error("🚨 Fraudulent Transaction")

    st.write("Prediction Probability:")
    st.write(f"Normal: {probability[0]*100:.2f}%")
    st.write(f"Fraud : {probability[1]*100:.2f}%")

# ------------------ Graph ------------------
st.subheader("Fraud vs Normal Transactions")

fig, ax = plt.subplots()
df["Class"].value_counts().plot(kind="bar", color=["green", "red"], ax=ax)
ax.set_xticklabels(["Normal", "Fraud"], rotation=0)
st.pyplot(fig)
