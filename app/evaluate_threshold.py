
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV dataset found.")

df = pd.read_csv(csv_files[0])

X = df.drop(
    columns=[
        "Churn",
        "tenure-binned",
        "MonthlyCharges-binned",
        "TotalCharges-binned",
    ]
)
y = df["Churn"].astype(int)

# Recreate the same test split used during baseline evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = joblib.load(MODEL_PATH)
probabilities = model.predict_proba(X_test)[:, 1]

print("Threshold | Precision | Recall | F1-score")
print("-" * 44)

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(
        f"{threshold:9.2f} | {precision:9.3f} | "
        f"{recall:6.3f} | {f1:8.3f}"
    )

print("\nConfusion matrix at threshold 0.40:")
predictions = (probabilities >= 0.40).astype(int)
print(confusion_matrix(y_test, predictions))
