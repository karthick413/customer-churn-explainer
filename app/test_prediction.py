
from pathlib import Path
import pandas as pd

from predictive_churn import predict_churn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV dataset found in data/")

df = pd.read_csv(csv_files[0])

# Select a customer who actually churned, for demonstration
customer = df[df["Churn"] == 1].iloc[0]

# Keep only the features used to train the model
excluded_columns = [
    "Churn",
    "tenure-binned",
    "MonthlyCharges-binned",
    "TotalCharges-binned",
]
customer_data = customer.drop(labels=excluded_columns).to_dict()

result = predict_churn(customer_data, threshold=0.30)

print("=== CUSTOMER CHURN PREDICTION ===")
print("Actual outcome:", "Churned" if customer["Churn"] == 1 else "Stayed")
print(f"Predicted churn probability: {result['churn_probability']:.1%}")
print("Predicted outcome:", "Churn" if result["prediction"] else "Stayed")
print(f"Decision threshold: {result['threshold']:.0%}")
