
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"


def predict_churn(customer_data, threshold=0.30):
    """
    Predict churn using a configurable probability threshold.

    customer_data: dict containing the original model input features.
    threshold: probability cutoff between 0 and 1.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    model = joblib.load(MODEL_PATH)
    customer_df = pd.DataFrame([customer_data])

    churn_probability = model.predict_proba(customer_df)[0, 1]
    churn_prediction = int(churn_probability >= threshold)

    return {
        "churn_probability": float(churn_probability),
        "prediction": churn_prediction,
        "threshold": threshold,
    }


if __name__ == "__main__":
    print("Prediction module created successfully.")
    print("Next, we will test it with a sample customer.")
