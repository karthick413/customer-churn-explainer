
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV file found in {DATA_DIR}")

df = pd.read_csv(csv_files[0])

print("=== DATASET OVERVIEW ===")
print("Shape:", df.shape)
print("\nData types:")
print(df.dtypes.to_string())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum().sort_values(ascending=False).to_string())

print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== NUMERIC SUMMARY ===")
print(df.describe().round(2).to_string())

print("\n=== CATEGORICAL VALUES ===")
for col in df.select_dtypes(include=["object", "category"]).columns:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts(dropna=False).head(10).to_dict())

print("\n=== TARGET BALANCE ===")
print(df["Churn"].value_counts(dropna=False))
print("\nChurn percentage:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))