from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

st.title("Customer Segmentation")

folder = Path(__file__).resolve().parent

data = pd.read_csv(
    folder / "customer_segmentation_final.csv"
)

bundle = joblib.load(folder / "customer_model.pkl")

# Display order only; original model cluster IDs stay unchanged.
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


def predict_segment(customer):
    """Apply saved preprocessing and predict a segment."""
    customer = customer[bundle["features"]].copy()

    for feature, (lower, upper) in bundle["bounds"].items():
        customer[feature] = customer[feature].clip(
            lower=lower,
            upper=upper
        )

    scaled_customer = pd.DataFrame(
        bundle["scaler"].transform(customer),
        columns=bundle["features"]
    )

    cluster = int(
        bundle["model"].predict(scaled_customer)[0]
    )

    return bundle["segment_names"][cluster]


# Dashboard
col1, col2 = st.columns(2)

col1.metric("Total Customers", len(data))

col2.metric(
    "Customer Segments",
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


# Prepare the customer table
st.subheader("Customer Data")

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


# Filter customers
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


# Manual prediction
st.divider()
st.subheader("Predict Customer Segment")

st.write("Enter the customer's actual details below.")

st.caption(
    "Default values are training-data medians. "
    "Use the same units and rate scales as your dataset."
)

st.caption(
    "Customer tenure days = reference date minus signup date. "
    "Days since last purchase = reference date minus last purchase date. "
    f"Notebook reference date: {bundle['reference_date']}"
)

values = {}

with st.form("prediction_form"):
    columns = st.columns(2)

    for i, feature in enumerate(bundle["features"]):
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
        columns=bundle["features"]
    )

    segment = predict_segment(customer)

    st.success(f"Customer Segment: {segment}")
    st.write("Segment Number:", display_order[segment])


# Test an existing customer
st.divider()
st.subheader("Test with an Existing Customer")

test_segment = st.selectbox(
    "Choose a segment to test",
    segment_order,
    key="test_segment"
)

if st.button("Test One Customer"):
    matching = data[
        data["Customer_Segment"] == test_segment
    ]

    if matching.empty:
        st.warning("No customers found in this segment.")
    else:
        row = matching.iloc[[0]]

        predicted_segment = predict_segment(row)

        st.write(
            "Customer ID:",
            str(row["customer_id"].iloc[0])
        )
        st.write("Saved segment:", test_segment)
        st.write("Model prediction:", predicted_segment)
        st.write(
            "Segment Number:",
            display_order[predicted_segment]
        )

        if predicted_segment == test_segment:
            st.success(
                "The prediction matches the saved segment!"
            )
        else:
            st.warning(
                "The prediction differs from the saved segment. "
                "We should check the model and dataset."
            )


# Download the filtered customer table
st.divider()
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