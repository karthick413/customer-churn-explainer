
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data and trained pipeline
csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV file found in {DATA_DIR}")

df = pd.read_csv(csv_files[0])
model = joblib.load(MODEL_PATH)

# Select one customer who actually churned
excluded_columns = [
    "Churn",
    "tenure-binned",
    "MonthlyCharges-binned",
    "TotalCharges-binned",
]

X = df.drop(columns=excluded_columns)
customer_index = df.index[df["Churn"] == 1][0]
customer = X.loc[[customer_index]]

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Transform customer data and preserve feature names
transformed = preprocessor.transform(customer)
if hasattr(transformed, "toarray"):
    transformed = transformed.toarray()

feature_names = preprocessor.get_feature_names_out()
transformed_df = pd.DataFrame(
    transformed,
    columns=feature_names,
    index=customer.index,
)

# Calculate SHAP values
explainer = shap.TreeExplainer(classifier)
shap_values = explainer(transformed_df)

# Display the model prediction
probability = model.predict_proba(customer)[0, 1]

print("=== LOCAL CHURN EXPLANATION ===")
print(f"Customer row index: {customer_index}")
print(f"Actual outcome: {'Churn' if df.loc[customer_index, 'Churn'] == 1 else 'No churn'}")
print(f"Predicted churn probability: {probability:.1%}")

# Save the waterfall explanation
shap.plots.waterfall(shap_values[0], max_display=15, show=False)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_local_waterfall.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

print("Saved:", OUTPUT_DIR / "shap_local_waterfall.png")
