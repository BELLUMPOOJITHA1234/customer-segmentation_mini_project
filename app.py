from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

st.title("Customer Segmentation")
st.write("Upload your customer dataset to predict customer segments.")

folder = Path(__file__).resolve().parent

# Load your saved model and preprocessing settings.
model_path = folder / "customer_model.pkl"

if not model_path.exists():
    st.error("customer_model.pkl was not found beside app.py.")
    st.stop()

bundle = joblib.load(model_path)
features = bundle["features"]

segment_order = [
    "Low-Value Customers",
    "Mid-Value Customers",
    "High-Value Customers"
]

display_order = {
    "Low-Value Customers": 0,
    "Mid-Value Customers": 1,
    "High-Value Customers": 2
}


def predict_segments(customers):
    """Predict using the saved feature order, limits and scaler."""
    prepared = customers[features].copy()

    for feature, (lower, upper) in bundle["bounds"].items():
        prepared[feature] = prepared[feature].clip(
            lower=lower,
            upper=upper
        )

    scaled = pd.DataFrame(
        bundle["scaler"].transform(prepared),
        columns=features,
        index=prepared.index
    )

    return bundle["model"].predict(scaled)


# Dataset upload
st.subheader("Upload Customer Dataset")

st.info(
    "Upload prepared numeric customer data. "
    "Cluster and Customer_Segment columns are not required. "
    "If present, they will be replaced with fresh predictions."
)

with st.expander("View required columns and input instructions"):
    st.write(pd.DataFrame({"Required column": features}))

    st.write(
        "Use the same units and rate scales as your training dataset."
    )

    st.write(
        "customer_tenure_days = reference date minus signup date."
    )

    st.write(
        "days_since_last_purchase = reference date minus "
        "last purchase date."
    )

    st.write(
        "Notebook reference date:",
        bundle["reference_date"]
    )

    st.write(
        "This version requires these calculated day columns. "
        "It does not convert raw signup_date or last_purchase_date columns."
    )

    st.download_button(
        label="Download Empty CSV Template",
        data=pd.DataFrame(columns=features).to_csv(
            index=False
        ).encode("utf-8"),
        file_name="customer_input_template.csv",
        mime="text/csv"
    )

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

# No dashboard or results before uploading.
if uploaded_file is None:
    st.info("Upload a dataset above to begin.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
    st.error("Could not read this file. Please upload a valid UTF-8 CSV.")
    st.stop()

if data.empty:
    st.error("The uploaded dataset has no customer rows.")
    st.stop()

missing_columns = [
    feature for feature in features
    if feature not in data.columns
]

if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

numeric_data = data[features].apply(
    pd.to_numeric,
    errors="coerce"
)

invalid_rows = ~np.isfinite(
    numeric_data.to_numpy(dtype=float)
).all(axis=1)

if invalid_rows.any():
    st.error(
        f"{int(invalid_rows.sum())} rows contain missing or invalid "
        "numbers in the required feature columns. "
        "Correct these rows and upload again."
    )

    st.dataframe(data.loc[invalid_rows].head(20))
    st.stop()

try:
    predicted_clusters = predict_segments(numeric_data)
except ValueError as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

# Replace any existing result labels with model predictions.
data["Cluster"] = predicted_clusters

data["Customer_Segment"] = data["Cluster"].map(
    bundle["segment_names"]
)

if data["Customer_Segment"].isna().any():
    st.error("The saved model contains a cluster without a segment name.")
    st.stop()

st.success(
    f"Successfully processed {len(data)} customer records."
)


# Dashboard
st.subheader("Customer Dashboard")

col1, col2 = st.columns(2)

col1.metric("Total Customers", len(data))

col2.metric(
    "Segments Found",
    data["Customer_Segment"].nunique()
)

st.subheader("Customers in Each Segment")

counts = (
    data["Customer_Segment"]
    .value_counts()
    .reindex(segment_order, fill_value=0)
)

counts.index = [
    f"{display_order[name]} - {name}"
    for name in counts.index
]

st.bar_chart(counts)


# Customer results
st.subheader("Customer Results")

display_data = data.copy()

display_data["Segment Number"] = (
    display_data["Customer_Segment"].map(display_order)
)

display_data = display_data.drop(
    columns=["Cluster"],
    errors="ignore"
)

display_data = display_data.sort_values(
    "Segment Number",
    kind="stable"
)

selected_segment = st.selectbox(
    "Filter customers by segment",
    ["All"] + segment_order,
    key="customer_filter"
)

if selected_segment == "All":
    filtered_data = display_data
else:
    filtered_data = display_data[
        display_data["Customer_Segment"] == selected_segment
    ]

st.write("Customers shown:", len(filtered_data))
st.dataframe(filtered_data)


# Download filtered results
st.subheader("Download Customer Results")

st.write(
    f"Download selection: {selected_segment} "
    f"— {len(filtered_data)} customers"
)

st.download_button(
    label="Download Results as CSV",
    data=filtered_data.to_csv(index=False).encode("utf-8"),
    file_name="customer_results.csv",
    mime="text/csv"
)


# Optional manual prediction
st.divider()
st.subheader("Predict One Customer Manually")

st.caption(
    "Default values are training-data medians. "
    "Replace them with the customer's actual details."
)

values = {}

with st.form("prediction_form"):
    columns = st.columns(2)

    for i, feature in enumerate(features):
        with columns[i % 2]:
            label = feature.replace("_", " ").title()

            if feature in [
                "is_premium_user",
                "discount_used",
                "refund_requested"
            ]:
                values[feature] = st.selectbox(
                    label + " (0 = No, 1 = Yes)",
                    [0, 1],
                    key=feature
                )
            else:
                values[feature] = st.number_input(
                    label,
                    value=float(bundle["defaults"][feature]),
                    key=feature
                )

    predict = st.form_submit_button("Predict Segment")

if predict:
    customer = pd.DataFrame(
        [values],
        columns=features
    )

    if not np.isfinite(customer.to_numpy(dtype=float)).all():
        st.error("Please enter valid finite numbers.")
    else:
        cluster = int(predict_segments(customer)[0])
        segment = bundle["segment_names"][cluster]

        st.success(f"Customer Segment: {segment}")
        st.write("Segment Number:", display_order[segment])
