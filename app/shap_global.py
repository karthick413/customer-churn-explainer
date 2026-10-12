
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

# Load dataset and trained pipeline
csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV file found in {DATA_DIR}")

df = pd.read_csv(csv_files[0])
model = joblib.load(MODEL_PATH)

excluded_columns = [
    "Churn",
    "tenure-binned",
    "MonthlyCharges-binned",
    "TotalCharges-binned",
]

X = df.drop(columns=excluded_columns)

# Use a manageable sample for initial explanations
X_sample = X.sample(
    n=min(500, len(X)),
    random_state=42,
)

# Transform features using the fitted preprocessing step
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X_sample)
feature_names = preprocessor.get_feature_names_out()

# SHAP TreeExplainer for the fitted XGBoost model
explainer = shap.TreeExplainer(classifier)
shap_values = explainer(X_transformed)

# Global importance: mean absolute SHAP value per feature
importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": abs(shap_values.values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

print("\n=== TOP 15 GLOBAL FEATURES ===")
print(importance.head(15).to_string(index=False))

importance.to_csv(
    OUTPUT_DIR / "shap_global_importance.csv",
    index=False,
)

# Summary / beeswarm plot
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# Global bar plot
shap.plots.bar(shap_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_bar.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

print("\nSaved outputs to:", OUTPUT_DIR)
print("- shap_global_importance.csv")
print("- shap_summary.png")
print("- shap_bar.png")
