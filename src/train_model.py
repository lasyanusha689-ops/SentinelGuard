import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("Loading dataset...")

df = pd.read_csv("data/creditcard.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================================
# 2. SPLIT DATA INTO TRAIN AND TEST
# ==========================================
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 3. CREATE MACHINE LEARNING MODEL
# ==========================================
print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# ==========================================
# 4. TRAIN MODEL
# ==========================================
print("\nTraining started... Please wait.")

model.fit(X_train, y_train)

print("Training completed successfully!")

# ==========================================
# 5. MAKE PREDICTIONS
# ==========================================
print("\nPredicting test transactions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# 6. EVALUATE MODEL
# ==========================================
print("\n========== MODEL EVALUATION ==========")

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Fraud"],
    zero_division=0
))

# ==========================================
# 7. SAVE TRAINED MODEL
# ==========================================
os.makedirs("models", exist_ok=True)

model_path = "models/fraud_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved location:", model_path)

print("\n===== SENTINELGUARD TRAINING COMPLETE =====")