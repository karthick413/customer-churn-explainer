from pathlib import Path
import pandas as pd

# Find the project root and data folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Find CSV files
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV file found in the data folder.")
    print(f"Place your dataset here: {DATA_DIR}")
else:
    file_path = csv_files[0]
    df = pd.read_csv(file_path)

    print("Dataset:", file_path.name)
    print("Shape:", df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\nTarget distribution:")
    if "Churn" in df.columns:
        print(df["Churn"].value_counts(dropna=False))
    else:
        print("No column named 'Churn' found.")