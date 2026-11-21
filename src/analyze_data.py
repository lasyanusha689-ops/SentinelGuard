
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Create reports folder
os.makedirs("reports", exist_ok=True)

print("=" * 45)
print("       SENTINELGUARD DATA ANALYSIS")
print("=" * 45)

# 1. Transaction counts
counts = df["Class"].value_counts().sort_index()

print("\n1. TRANSACTION COUNTS")
print("Normal transactions:", counts[0])
print("Fraud transactions:", counts[1])

# 2. Calculate percentages
fraud_percentage = (counts[1] / len(df)) * 100
normal_percentage = (counts[0] / len(df)) * 100

print("\n2. TRANSACTION PERCENTAGES")
print(f"Normal: {normal_percentage:.4f}%")
print(f"Fraud: {fraud_percentage:.4f}%")

# 3. Amount statistics
print("\n3. TRANSACTION AMOUNT STATISTICS")
print(df["Amount"].describe())

# 4. Check missing values
missing_values = df.isnull().sum().sum()

print("\n4. MISSING VALUES")
print("Total missing values:", missing_values)

# 5. Fraud distribution chart
plt.figure(figsize=(8, 5))
counts.plot(kind="bar", logy=True)
plt.title("Normal vs Fraud Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Transaction Count (Log Scale)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/fraud_class_distribution.png")
plt.close()

# 6. Amount distribution chart
plt.figure(figsize=(8, 5))
df["Amount"].hist(bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("reports/amount_distribution.png")
plt.close()

print("\n5. CHARTS CREATED")
print("Fraud class chart: reports/fraud_class_distribution.png")
print("Amount chart: reports/amount_distribution.png")

print("\n" + "=" * 45)
print("     DATA ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 45)