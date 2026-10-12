from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st



PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

THRESHOLD_PATH = OUTPUT_DIR / "selected_threshold.json"
IMPORTANCE_PATH = OUTPUT_DIR / "shap_global_importance.csv"
SUMMARY_PATH = OUTPUT_DIR / "shap_summary.png"
BAR_PATH = OUTPUT_DIR / "shap_bar.png"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"

st.set_page_config(
    page_title="Customer Churn Explainer",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Churn Explainer")
st.caption("XGBoost + SHAP | Predict, Explain, and Explore Churn")


if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run: "
        "python -m app.evaluate_model"
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)


model = load_model()

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    st.error(f"No CSV dataset found in {DATA_DIR}")
    st.stop()

# Prefer the Telco dataset filename if present.
preferred_files = [
    path for path in csv_files
    if "telco" in path.name.lower()
]

DATA_PATH = preferred_files[0] if preferred_files else csv_files[0]
df = load_data(str(DATA_PATH))

if "Churn" not in df.columns:
    st.error("The dataset must contain a Churn column.")
    st.stop()


# Remove target and binned columns from prediction features.
excluded_columns = [
    "Churn",
    "tenure-binned",
    "MonthlyCharges-binned",
    "TotalCharges-binned",
]

X = df.drop(columns=excluded_columns, errors="ignore").copy()
FEATURES = list(X.columns)

# Read the threshold selected using validation data.
threshold = 0.35

if THRESHOLD_PATH.exists():
    try:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
            threshold = float(json.load(file)["threshold"])
    except (ValueError, KeyError, json.JSONDecodeError, OSError):
        st.warning("Could not read the saved threshold. Using 0.35.")

# Get feature types from the trained pipeline.
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

categorical_columns = list(preprocessor.transformers_[0][2])
numeric_columns = list(preprocessor.transformers_[1][2])



st.sidebar.header("Customer Details")
st.sidebar.write("Enter customer information below.")

with st.sidebar.form("customer_form"):
    customer = {}

    for column in FEATURES:
        if column in categorical_columns:
            choices = sorted(
                X[column].dropna().astype(str).unique().tolist()
            )

            if choices:
                customer[column] = st.selectbox(
                    column,
                    choices,
                )

        elif column in numeric_columns:
            numeric_values = pd.to_numeric(
                X[column],
                errors="coerce",
            ).dropna()

            if numeric_values.empty:
                customer[column] = 0
                continue

            minimum = float(numeric_values.min())
            maximum = float(numeric_values.max())
            default = float(numeric_values.median())

            if minimum == maximum:
                customer[column] = minimum

            elif column == "SeniorCitizen":
                choices = sorted(numeric_values.unique().tolist())
                default_index = (
                    choices.index(default)
                    if default in choices
                    else 0
                )

                customer[column] = st.selectbox(
                    column,
                    choices,
                    index=default_index,
                )

            elif column == "tenure":
                customer[column] = st.number_input(
                    column,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(default),
                    step=1,
                )

            else:
                customer[column] = st.number_input(
                    column,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=1.0,
                )

    decision_threshold = st.slider(
        "Churn decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(min(max(threshold, 0.10), 0.90)),
        step=0.01,
        help="Lower thresholds generally flag more customers.",
    )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True,
    )



st.header("Customer Churn Prediction")

prediction_succeeded = False
probability = None
prediction = None
customer_df = None

if submitted:
    customer_df = pd.DataFrame(
        [customer],
        columns=FEATURES,
    )

    try:
        probability = float(
            model.predict_proba(customer_df)[0, 1]
        )

        prediction = int(probability >= decision_threshold)
        prediction_succeeded = True

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Churn Probability",
            f"{probability:.1%}",
        )

        col2.metric(
            "Prediction",
            "Churn" if prediction == 1 else "No Churn",
        )

        col3.metric(
            "Decision Threshold",
            f"{decision_threshold:.0%}",
        )

        if prediction == 1:
            st.warning(
                "This customer is flagged as being at risk of churn."
            )
        else:
            st.success(
                "This customer is below the selected risk threshold."
            )



        st.subheader("Why did the model make this prediction?")

        try:
            transformed = preprocessor.transform(customer_df)

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            feature_names = preprocessor.get_feature_names_out()

            transformed_df = pd.DataFrame(
                transformed,
                columns=feature_names,
            )

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer(transformed_df)

            fig, ax = plt.subplots(figsize=(10, 7))

            shap.plots.waterfall(
                shap_values[0],
                max_display=15,
                show=False,
            )

            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Red features push the model output higher; blue features "
                "push it lower. SHAP contributions explain the model output "
                "and do not establish causal effects."
            )

        except Exception as error:
            st.error(
                f"Could not generate the SHAP plot: {error}"
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

else:
    st.info(
        "Choose customer details in the sidebar and click Predict Churn."
    )


st.divider()
st.header("Global Model Explainability")

if IMPORTANCE_PATH.exists():
    importance = pd.read_csv(IMPORTANCE_PATH)

    st.subheader("Top 10 Influential Features")

    st.dataframe(
        importance.head(10).round(4),
        use_container_width=True,
        hide_index=True,
    )

    if BAR_PATH.exists():
        st.subheader("Global Feature Importance")
        st.image(
            str(BAR_PATH),
            use_container_width=True,
        )

    if SUMMARY_PATH.exists():
        with st.expander("View SHAP Summary Plot"):
            st.image(
                str(SUMMARY_PATH),
                use_container_width=True,
            )

else:
    st.info(
        "Global SHAP files are missing. Run: "
        "python -m app.shap_global"
    )


st.divider()
st.header("Customer Retention Recommendations")

st.caption(
    "Suggested actions based on predicted risk and customer profile. "
    "These are heuristic recommendations, not causal predictions."
)

if prediction_succeeded:
    recommendations = []

    # Risk-based recommendation.
    if probability >= decision_threshold:
        recommendations.append(
            (
                "High priority: proactive outreach",
                "Contact the customer to understand their concerns "
                "and offer appropriate support.",
            )
        )
    else:
        recommendations.append(
            (
                "Maintain engagement",
                "Continue regular engagement and monitor the customer's "
                "service experience.",
            )
        )

    # Contract-based recommendation.
    if customer.get("Contract") == "Month-to-month":
        recommendations.append(
            (
                "Review contract options",
                "Explain available contract options and their benefits "
                "without pressuring the customer to switch.",
            )
        )

    # Tenure-based recommendation.
    if customer.get("tenure", 0) <= 12:
        recommendations.append(
            (
                "Strengthen the early customer experience",
                "Check onboarding, service setup, and whether the customer "
                "needs help using their services.",
            )
        )

    # Technical support recommendation.
    if customer.get("TechSupport") == "No":
        recommendations.append(
            (
                "Review support needs",
                "Ask whether the customer needs technical assistance and "
                "explain available support options.",
            )
        )

    # Online security recommendation.
    if customer.get("OnlineSecurity") == "No":
        recommendations.append(
            (
                "Explain security options",
                "Provide clear information about available online security "
                "features and their costs.",
            )
        )

    # Payment preference recommendation.
    if customer.get("PaymentMethod") == "Electronic check":
        recommendations.append(
            (
                "Review payment preferences",
                "Explain alternative payment methods, if available, "
                "and let the customer choose.",
            )
        )

    for title, description in recommendations:
        st.markdown(f"**{title}**")
        st.write(description)

else:
    st.info(
        "Enter customer details in the sidebar and click "
        "'Predict Churn' to generate retention recommendations."
    )


st.divider()
st.header("Model Performance")

if METRICS_PATH.exists():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        st.caption(
            "Results from the held-out test set. The decision threshold "
            "was selected using validation data."
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.1%}",
        )

        m2.metric(
            "Accuracy",
            f"{metrics['accuracy']:.1%}",
        )

        m3.metric(
            "Selected Threshold",
            f"{metrics.get('selected_threshold', threshold):.0%}",
        )

        m4, m5, m6 = st.columns(3)

        m4.metric(
            "Churn Precision",
            f"{metrics['precision']:.1%}",
        )

        m5.metric(
            "Churn Recall",
            f"{metrics['recall']:.1%}",
        )

        m6.metric(
            "Churn F1-score",
            f"{metrics['f1']:.1%}",
        )

        st.info(
            f"On the held-out test set, the model identified "
            f"{metrics['recall']:.1%} of customers who churned. "
            f"Precision was {metrics['precision']:.1%}, meaning some "
            f"customers flagged as at risk did not churn."
        )

    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        st.warning(f"Could not load model metrics: {error}")

else:
    st.warning(
        "Evaluation metrics not found. Run: "
        "python -m app.evaluate_model"
    )


# --------------------------------------------------
# 8. BUSINESS ANALYTICS
# --------------------------------------------------

st.divider()
st.header("Business Analytics")

st.caption(
    "Historical dataset statistics. These describe patterns, "
    "not causes of customer churn."
)

analytics_df = df.copy()

if pd.api.types.is_numeric_dtype(analytics_df["Churn"]):
    analytics_df["ChurnFlag"] = pd.to_numeric(
        analytics_df["Churn"],
        errors="coerce",
    )
else:
    analytics_df["ChurnFlag"] = (
        analytics_df["Churn"]
        .astype(str)
        .str.strip()
        .map({
            "Yes": 1,
            "No": 0,
            "1": 1,
            "0": 0,
        })
    )

analytics_df = analytics_df.dropna(subset=["ChurnFlag"])
analytics_df["ChurnFlag"] = analytics_df["ChurnFlag"].astype(int)

total_customers = len(analytics_df)
total_churners = int(analytics_df["ChurnFlag"].sum())

overall_churn_rate = (
    total_churners / total_customers
    if total_customers
    else 0
)

kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(
    "Total Customers",
    f"{total_customers:,}",
)

kpi2.metric(
    "Customers Churned",
    f"{total_churners:,}",
)

kpi3.metric(
    "Overall Churn Rate",
    f"{overall_churn_rate:.1%}",
)


st.subheader("Churn Rate by Contract Type")

if "Contract" in analytics_df.columns:
    contract_stats = (
        analytics_df.groupby("Contract")["ChurnFlag"]
        .agg(
            Customers="size",
            Churned="sum",
            Churn_Rate="mean",
        )
        .sort_values("Churn_Rate", ascending=False)
    )

    contract_stats["Churn Rate (%)"] = (
        contract_stats["Churn_Rate"] * 100
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Churn Rate (%)**")
        st.bar_chart(contract_stats["Churn Rate (%)"])

    with right:
        st.markdown("**Customer Counts**")
        st.bar_chart(contract_stats[["Customers", "Churned"]])

    st.dataframe(
        contract_stats[
            ["Customers", "Churned", "Churn Rate (%)"]
        ].round(2),
        use_container_width=True,
    )

st.subheader("Churn Rate by Tenure")

if "tenure" in analytics_df.columns:
    tenure_df = analytics_df.copy()

    tenure_df["Tenure Group"] = pd.cut(
        tenure_df["tenure"],
        bins=[-1, 12, 24, 48, 60, float("inf")],
        labels=[
            "0–12 months",
            "13–24 months",
            "25–48 months",
            "49–60 months",
            "61+ months",
        ],
    )

    tenure_stats = (
        tenure_df.groupby(
            "Tenure Group",
            observed=False,
        )["ChurnFlag"]
        .agg(
            Customers="size",
            Churned="sum",
            Churn_Rate="mean",
        )
    )

    tenure_stats["Churn Rate (%)"] = (
        tenure_stats["Churn_Rate"] * 100
    )

    st.bar_chart(tenure_stats["Churn Rate (%)"])

    st.dataframe(
        tenure_stats[
            ["Customers", "Churned", "Churn Rate (%)"]
        ].round(2),
        use_container_width=True,
    )


st.subheader("Monthly Charges by Actual Churn Outcome")

if "MonthlyCharges" in analytics_df.columns:
    charge_stats = (
        analytics_df.groupby("ChurnFlag")["MonthlyCharges"]
        .agg(
            Customers="count",
            Average="mean",
            Median="median",
        )
        .rename(index={
            0: "No Churn",
            1: "Churn",
        })
    )

    st.bar_chart(charge_stats[["Average", "Median"]])

    st.dataframe(
        charge_stats.rename(
            columns={
                "Average": "Average Monthly Charges",
                "Median": "Median Monthly Charges",
            }
        ).round(2),
        use_container_width=True,
    )

st.divider()

st.caption(
    "Customer Churn Explainer | XGBoost + SHAP | "
    "Predictions are estimates, not guarantees of future behavior."
)