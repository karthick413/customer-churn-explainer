
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# 1. Load data
csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV file found in {DATA_DIR}")

df = pd.read_csv(csv_files[0])

excluded_columns = [
    "Churn",
    "tenure-binned",
    "MonthlyCharges-binned",
    "TotalCharges-binned",
]

X = df.drop(columns=excluded_columns, errors="ignore").copy()
y = df["Churn"].copy()

# Support either numeric labels (0/1) or Yes/No labels.
if not pd.api.types.is_numeric_dtype(y):
    y = y.astype(str).str.strip().map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Churn contains unexpected labels or missing values.")

y = y.astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Expected Churn labels to be 0 and 1.")

# 2. Split into train (60%), validation (20%), and test (20%).
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

# 3. Fit preprocessing using training data only.
categorical_columns = X_train.select_dtypes(
    include=["str", "object", "category"]
).columns.tolist()

numeric_columns = [
    column for column in X_train.columns
    if column not in categorical_columns
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
        ("numeric", "passthrough", numeric_columns),
    ]
)

classifier = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

from sklearn.pipeline import Pipeline

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)

print("Training model...")
model.fit(X_train, y_train)

# 4. Choose the threshold on validation data, not test data.
val_probabilities = model.predict_proba(X_val)[:, 1]

threshold_results = []
for threshold in np.arange(0.10, 0.81, 0.01):
    val_predictions = (val_probabilities >= threshold).astype(int)

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, val_predictions, zero_division=0
        ),
    })

threshold_df = pd.DataFrame(threshold_results)

# Select the threshold with the best validation F1.
# If tied, prefer the higher-recall threshold.
best_row = threshold_df.sort_values(
    ["f1", "recall", "threshold"],
    ascending=[False, False, False],
).iloc[0]

best_threshold = float(best_row["threshold"])

threshold_df.to_csv(
    OUTPUT_DIR / "validation_thresholds.csv",
    index=False,
)

# 5. Evaluate once on the held-out test set.
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

metrics = {
    "selected_threshold": best_threshold,
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
    "average_precision": float(
        average_precision_score(y_test, test_probabilities)
    ),
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
}

print("\n=== VALIDATION THRESHOLD SELECTION ===")
print(f"Selected threshold: {best_threshold:.2f}")
print(f"Validation precision: {best_row['precision']:.3f}")
print(f"Validation recall: {best_row['recall']:.3f}")
print(f"Validation F1: {best_row['f1']:.3f}")

print("\n=== FINAL TEST RESULTS ===")
for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print("\n=== TEST CLASSIFICATION REPORT ===")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )
)

print("=== TEST CONFUSION MATRIX ===")
print(confusion_matrix(y_test, test_predictions))

# 6. Save the model, selected threshold, and metrics.
joblib.dump(model, MODEL_DIR / "churn_pipeline.joblib")

with open(OUTPUT_DIR / "selected_threshold.json", "w") as file:
    json.dump({"threshold": best_threshold}, file, indent=4)

with open(OUTPUT_DIR / "test_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nSaved:")
print("models/churn_pipeline.joblib")
print("outputs/validation_thresholds.csv")
print("outputs/selected_threshold.json")
print("outputs/test_metrics.json")
