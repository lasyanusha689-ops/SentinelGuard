
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SentinelGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #f0f2f6;
}
[data-testid="stMetric"] {
    background-color: #1b2230;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #30394b;
}
div.stButton > button {
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- HEADER ----------------
st.title("🛡️ SentinelGuard")
st.subheader("AI-Based Credit Card Fraud Detection System")
st.caption("Machine learning powered transaction analysis using Random Forest")

st.divider()

# ---------------- DATA UPLOAD ----------------
st.header("📂 Dataset")

uploaded_file = st.file_uploader(
    "Upload creditcard.csv to activate the dashboard",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload the credit card dataset to view transaction analytics "
        "and test sample predictions. Your dataset is not stored in this app."
    )
    st.markdown("""
    **Dataset required columns:**
    `Time`, `V1`–`V28`, `Amount`, `Class`

    Get the dataset from Kaggle:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """)
    st.stop()

# ---------------- READ DATA ----------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV file: {e}")
    st.stop()

required_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error("Dataset is missing required columns: " + ", ".join(missing_columns))
    st.stop()

# ---------------- DATA SUMMARY ----------------
total_transactions = len(df)
normal_transactions = int((df["Class"] == 0).sum())
fraud_transactions = int((df["Class"] == 1).sum())
fraud_rate = (fraud_transactions / total_transactions * 100) if total_transactions else 0

# ---------------- OVERVIEW ----------------
st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{total_transactions:,}")
c2.metric("Normal Transactions", f"{normal_transactions:,}")
c3.metric("Fraud Transactions", f"{fraud_transactions:,}")
c4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

st.divider()

# ---------------- CHARTS ----------------
st.subheader("Transaction Distribution")

fig, ax = plt.subplots()
ax.bar(
    ["Normal", "Fraud"],
    [normal_transactions, fraud_transactions]
)
ax.set_ylabel("Number of Transactions")
ax.set_title("Normal vs Fraud Transactions")
st.pyplot(fig)
plt.close(fig)

# ---------------- AMOUNT ANALYSIS ----------------
st.subheader("Transaction Amount Distribution")

fig2, ax2 = plt.subplots()
normal_amounts = df.loc[df["Class"] == 0, "Amount"]
fraud_amounts = df.loc[df["Class"] == 1, "Amount"]

ax2.hist(normal_amounts, bins=50, alpha=0.6, label="Normal")
ax2.hist(fraud_amounts, bins=50, alpha=0.6, label="Fraud")
ax2.set_xlabel("Transaction Amount")
ax2.set_ylabel("Frequency")
ax2.legend()
st.pyplot(fig2)
plt.close(fig2)

st.divider()

# ---------------- SAMPLE PREDICTION ----------------
st.header("🔍 Demo Prediction")

choice = st.selectbox(
    "Choose transaction type",
    ["Normal Transaction", "Fraud Transaction"]
)

if choice == "Normal Transaction":
    sample_data = df[df["Class"] == 0].head(1)
else:
    sample_data = df[df["Class"] == 1].head(1)

if not sample_data.empty:
    sample = sample_data.drop(columns=["Class"]).iloc[0]
    sample_df = pd.DataFrame([sample])

    prediction = model.predict(sample_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(sample_df)[0][1] * 100
    else:
        probability = 0

    if prediction == 1:
        st.error("🚨 Prediction: FRAUD")
    else:
        st.success("✅ Prediction: NORMAL")

    st.metric("Fraud Probability", f"{probability:.2f}%")
    st.caption("This prediction is generated from a sample transaction in the uploaded dataset.")
else:
    st.warning("No transaction sample available for the selected category.")

st.divider()

# ---------------- MANUAL PREDICTION ----------------
st.header("🧪 Manual Transaction Prediction")

with st.form("manual_prediction_form"):
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    time_value = st.number_input("Transaction Time", min_value=0.0, value=0.0)

    st.write("Enter feature values (V1–V28)")
    feature_values = {}

    with st.expander("V1 to V28 Features", expanded=False):
        for i in range(1, 29):
            feature_values[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                step=0.1,
                key=f"manual_V{i}"
            )

    predict_button = st.form_submit_button("Analyze Transaction")

if predict_button:
    input_data = {
        "Time": time_value,
        **feature_values,
        "Amount": amount
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[model.feature_names_in_] if hasattr(model, "feature_names_in_") else input_df

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1] * 100
    else:
        probability = None

    if prediction == 1:
        st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
    else:
        st.success("✅ TRANSACTION APPEARS NORMAL")

    if probability is not None:
        st.metric("Fraud Probability", f"{probability:.2f}%")

st.divider()

# ---------------- MODEL PERFORMANCE ----------------
st.header("📈 Model Performance")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", "99.94%")
m2.metric("Precision", "84.21%")
m3.metric("Fraud Recall", "81.63%")
m4.metric("ROC-AUC", "0.9782")

st.subheader("Confusion Matrix")

confusion_matrix_data = pd.DataFrame(
    [[56849, 15], [18, 80]],
    index=["Actual Normal", "Actual Fraud"],
    columns=["Predicted Normal", "Predicted Fraud"]
)

st.dataframe(confusion_matrix_data, use_container_width=True)
