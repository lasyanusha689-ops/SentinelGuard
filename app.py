
import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SentinelGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: white;
}

[data-testid="stMetric"] {
    background-color: #172235;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #263750;
}

h1, h2, h3 {
    color: #60a5fa;
}

.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATASET AND MODEL
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/fraud_model.pkl")


df = load_data()
model = load_model()

# ==========================================
# HEADER
# ==========================================

st.title("🛡️ SentinelGuard")

st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "An intelligent machine learning system "
    "for identifying suspicious financial transactions."
)

st.divider()

# ==========================================
# DATASET OVERVIEW
# ==========================================

total_transactions = len(df)

fraud_count = int(df["Class"].sum())

normal_count = total_transactions - fraud_count

fraud_percentage = (
    fraud_count / total_transactions
) * 100

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Normal Transactions",
    f"{normal_count:,}"
)

col3.metric(
    "Fraud Transactions",
    f"{fraud_count:,}"
)

col4.metric(
    "Fraud Rate",
    f"{fraud_percentage:.3f}%"
)

st.divider()

# ==========================================
# TRANSACTION PREDICTION
# ==========================================

st.header("🔍 Transaction Analysis")

st.sidebar.title("🛡️ SentinelGuard")

mode = st.sidebar.radio(
    "Select Prediction Mode",
    ["Demo Transactions", "Manual Input"]
)

def predict_transaction(transaction):

    prediction = model.predict(transaction)[0]

    probabilities = model.predict_proba(transaction)[0]

    fraud_probability = probabilities[1] * 100

    return prediction, fraud_probability


# ==========================================
# DEMO TRANSACTION MODE
# ==========================================

if mode == "Demo Transactions":

    st.subheader("🧪 Test Sample Transactions")

    sample_type = st.selectbox(
        "Choose Transaction Type",
        ["Normal Transaction", "Fraud Transaction"]
    )

    if sample_type == "Normal Transaction":

        transaction = (
            df[df["Class"] == 0]
            .drop("Class", axis=1)
            .iloc[[0]]
        )

        actual = "Normal"

    else:

        transaction = (
            df[df["Class"] == 1]
            .drop("Class", axis=1)
            .iloc[[0]]
        )

        actual = "Fraud"

    st.info(
        "This demo uses a sample from the historical "
        "credit card fraud dataset. It is not a live transaction."
    )

    if st.button(
        "🔎 Analyze Transaction",
        use_container_width=True
    ):

        prediction, fraud_probability = predict_transaction(
            transaction
        )

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("🚨 FRAUD DETECTED")

        else:

            st.success("✅ NORMAL TRANSACTION")

        col1, col2 = st.columns(2)

        col1.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

        col2.metric(
            "Actual Sample Category",
            actual
        )

        st.progress(
            min(100, max(0, int(fraud_probability)))
        )


# ==========================================
# MANUAL TRANSACTION MODE
# ==========================================

else:

    st.subheader("✍️ Manual Transaction Input")

    st.warning(
        "V1–V28 are anonymized dataset features. "
        "Manual values are only for demonstration, "
        "not real banking transactions."
    )

    with st.form("transaction_form"):

        time_value = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=0.0
        )

        amount_value = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0
        )

        features = {}

        with st.expander("Enter Features V1–V28"):

            for i in range(1, 29):

                features[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=0.0,
                    step=0.1,
                    key=f"v{i}"
                )

        submit = st.form_submit_button(
            "🔎 Analyze Transaction",
            use_container_width=True
        )

    if submit:

        transaction_data = {
            "Time": time_value,
            **features,
            "Amount": amount_value
        }

        transaction = pd.DataFrame(
            [transaction_data]
        )

        transaction = transaction[
            model.feature_names_in_
        ]

        prediction, fraud_probability = predict_transaction(
            transaction
        )

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("🚨 FRAUD DETECTED")

        else:

            st.success("✅ NORMAL TRANSACTION")

        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

        st.progress(
            min(100, max(0, int(fraud_probability)))
        )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.divider()

st.header("📈 Model Performance")

X = df.drop("Class", axis=1)

y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

# Display performance metrics

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

m2.metric(
    "Precision",
    f"{precision * 100:.2f}%"
)

m3.metric(
    "Fraud Recall",
    f"{recall * 100:.2f}%"
)

m4.metric(
    "ROC-AUC",
    f"{roc_auc:.4f}"
)

st.caption(
    "Metrics are calculated using the same held-out "
    "test split used during model evaluation."
)

# ==========================================
# TRANSACTION DISTRIBUTION CHART
# ==========================================

st.divider()

st.header("📊 Transaction Distribution")

chart_data = pd.DataFrame({
    "Transaction Type": [
        "Normal Transactions",
        "Fraud Transactions"
    ],

    "Count": [
        normal_count,
        fraud_count
    ]
})

st.bar_chart(
    chart_data.set_index("Transaction Type")
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

st.divider()

st.header("🎯 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,

    index=[
        "Actual Normal",
        "Actual Fraud"
    ],

    columns=[
        "Predicted Normal",
        "Predicted Fraud"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)

st.write(
    "**True Negatives:**", int(cm[0][0])
)

st.write(
    "**False Positives:**", int(cm[0][1])
)

st.write(
    "**False Negatives:**", int(cm[1][0])
)

st.write(
    "**True Positives:**", int(cm[1][1])
)

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "🛡️ SentinelGuard | Random Forest Machine Learning | "
    "AI Internship Portfolio Project"
)
