import pandas as pd

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Display first 5 rows
print(df.head())

# Dataset information
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Check fraud and normal transactions
print("\nTransaction Counts:")
print(df["Class"].value_counts())