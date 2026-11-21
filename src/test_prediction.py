
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Recreate test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Select one normal and one fraud transaction
normal_index = y_test[y_test == 0].index[0]
fraud_index = y_test[y_test == 1].index[0]

samples = {
    "Normal Transaction": X_test.loc[[normal_index]],
    "Fraud Transaction": X_test.loc[[fraud_index]]
}

print("=" * 45)
print("   SENTINELGUARD PREDICTION TEST")
print("=" * 45)

for name, transaction in samples.items():

    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0]

    result = "FRAUD 🚨" if prediction == 1 else "NORMAL ✅"
    fraud_probability = probability[1] * 100

    print(f"\nTest: {name}")
    print("Predicted Result:", result)
    print(f"Fraud Probability: {fraud_probability:.2f}%")

print("\nPrediction testing completed!")
