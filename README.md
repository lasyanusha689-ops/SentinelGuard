# SentinelGuard 🛡️
## AI-Based Credit Card Fraud Detection System

### 1. Project Overview
SentinelGuard is a machine learning-based project designed to detect fraudulent credit card transactions using the Random Forest algorithm.

The system analyzes transaction data and predicts whether a transaction is Normal or Fraudulent.

### 2. Objectives
- To identify fraudulent credit card transactions.
- To apply machine learning for fraud detection.
- To reduce financial losses caused by fraudulent transactions.
- To display prediction results through an interactive dashboard.

### 3. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

### 4. Machine Learning Algorithm
**Random Forest Classifier**

Random Forest is a supervised machine learning algorithm that uses multiple decision trees to make predictions.

### 5. Dataset
This project uses the Credit Card Fraud Detection dataset.

- Total transactions: 284,807
- Normal transactions: 284,315
- Fraud transactions: 492
- Total features: 30
- Target column: Class

**Dataset download:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Download `creditcard.csv` and place it inside the `data` folder.

### 6. Project Structure

```text
SentinelGuard/
├── data/
│   └── creditcard.csv
├── models/
│   └── fraud_model.pkl
├── reports/
│   ├── amount_distribution.png
│   └── fraud_class_distribution.png
├── src/
│   ├── analyze_data.py
│   ├── check_data.py
│   ├── test_prediction.py
│   └── train_model.py
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

### 7. Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 99.94% |
| Precision | 84.21% |
| Fraud Recall | 81.63% |
| ROC-AUC | 0.9782 |

### 8. Dashboard Features
- Dataset overview
- Normal and fraud transaction counts
- Fraud prediction using demo transactions
- Manual transaction input
- Fraud probability display
- Model performance metrics
- Confusion matrix

### 9. How to Run the Project

**Step 1: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 2: Train the model**

```bash
python src/train_model.py
```

**Step 3: Run the Streamlit dashboard**

```bash
python -m streamlit run app.py
```

### 10. Conclusion
SentinelGuard demonstrates how machine learning can be used to identify suspicious credit card transactions. The project provides an interactive dashboard to understand predictions and model performance.

### Disclaimer
This is an educational project developed for learning and demonstration purposes. It is not intended for real-world banking or financial decisions.