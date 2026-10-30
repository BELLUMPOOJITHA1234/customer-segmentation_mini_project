# Customer Segmentation Using K-Means

A machine-learning web application that groups customers into **Standard-Value Customers** and **High Lifetime-Value Customers** based on customer characteristics, purchasing behaviour and engagement.

The model is developed in Jupyter Notebook and deployed through a Streamlit web application.

## Live Application

[Open Customer Segmentation App](https://customer-segmentationminiproject-ws7tkv6p49d6udcmud5vrh.streamlit.app/)

## Project Objective

The project helps users explore customer groups and understand their differences. These insights can support customer retention, personalised marketing and customer relationship management.

## Customer Segments

| Display Number | Segment | Meaning |
|---|---|---|
| 0 | Standard-Value Customers | The cluster with the lower average lifetime value |
| 1 | High Lifetime-Value Customers | The cluster with the higher average lifetime value |

The model uses multiple customer features to create the groups. Segment names are assigned afterwards by comparing each cluster’s average lifetime value.

Display numbers are separate from the model’s internal cluster IDs. A high-value segment does not mean every customer in that group has a higher lifetime value than every customer in the standard-value group.

## Features

- Upload a customer dataset in CSV format.
- Automatically predict a segment for each customer.
- View customer counts and segment distribution.
- Filter customers by segment.
- Test an uploaded customer without entering values manually.
- Optionally enter customer details for an individual prediction.
- Download filtered prediction results as a CSV file.
- Download an empty CSV template containing the required columns.
- Validate missing columns and invalid numeric inputs.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Data processing and application logic |
| Pandas | Reading, cleaning and analysing datasets |
| NumPy | Numerical operations and input validation |
| Matplotlib | Data visualisation in the notebook |
| Seaborn | Exploratory visualisation in the notebook |
| Scikit-learn | Scaling, clustering, evaluation and PCA |
| Joblib | Saving and loading the trained model bundle |
| Jupyter Notebook | Model development and analysis |
| Streamlit | Web interface and application execution |
| GitHub | Source-code storage |
| Streamlit Community Cloud | Application hosting |

## Machine-Learning Approach

### K-Means Clustering

K-Means is an unsupervised learning algorithm that groups similar customers together.

This project uses:

- `n_clusters=2`
- `random_state=42`
- `n_init=10`

Customers are assigned to the nearest cluster centre using their scaled features.

Two clusters are selected to meet the project’s requirement. Elbow and silhouette analyses are available in the notebook to compare different cluster counts.

### StandardScaler

StandardScaler standardises the numeric features before clustering. This helps prevent features with larger numerical scales from dominating the distance calculation.

The website reuses the saved scaler without fitting it again.

### Outlier Treatment

Selected numeric features are capped using the Interquartile Range method:

- Lower limit: Q1 − 1.5 × IQR
- Upper limit: Q3 + 1.5 × IQR

The website applies the same saved limits used during training.

### Silhouette Score

The silhouette score evaluates how compact and separated the clusters are.

It is a clustering evaluation measure, not a classification accuracy percentage.

### Principal Component Analysis

PCA reduces the scaled features to two components for plotting the clusters in the notebook.

PCA is used for visualisation and is not part of the website’s prediction pipeline.

## Model Input Features

The model uses these 20 features:

| Feature | Description |
|---|---|
| `age` | Customer age |
| `is_premium_user` | Premium membership indicator |
| `total_visits` | Total customer visits |
| `avg_session_time` | Average session duration |
| `pages_per_session` | Average pages viewed per session |
| `email_open_rate` | Email opening rate |
| `email_click_rate` | Email link clicking rate |
| `total_spent` | Total customer spending |
| `avg_order_value` | Average order value |
| `discount_used` | Discount usage indicator |
| `support_tickets` | Number of support tickets |
| `refund_requested` | Refund request indicator |
| `delivery_delay_days` | Delivery delay in days |
| `satisfaction_score` | Customer satisfaction score |
| `nps_score` | Net Promoter Score |
| `marketing_spend_per_user` | Marketing spending per customer |
| `lifetime_value` | Customer lifetime value supplied in the dataset |
| `last_3_month_purchase_freq` | Purchase frequency during the last three months |
| `customer_tenure_days` | Days between signup and the notebook reference date |
| `days_since_last_purchase` | Days between the last purchase and the notebook reference date |

Lifetime value is an input to this project. The application predicts a customer segment, not a future lifetime-value amount.

## Project Workflow

### Model Development

1. Load the customer dataset.
2. Explore the data using statistics and visualisations.
3. Handle missing values.
4. Convert dates and calculate tenure and recency.
5. Prepare categorical data and select the clustering features.
6. Cap selected outliers.
7. Scale the features.
8. Compare different cluster counts.
9. Train the final two-cluster K-Means model.
10. Name the clusters using average lifetime value.
11. Evaluate and visualise the results.
12. Save the model and preprocessing settings.

### Website Prediction

1. The user uploads a prepared CSV file.
2. The application checks required columns and numeric values.
3. Features are selected in the saved training order.
4. Saved outlier limits are applied.
5. The saved scaler transforms the features.
6. The trained model predicts cluster IDs.
7. Cluster IDs are mapped to segment names.
8. Results appear in the dashboard and can be downloaded.

The website does not retrain the model when a dataset is uploaded.

## Frontend and Backend

### Frontend

Streamlit provides the user interface, including:

- CSV upload
- Dashboard metrics
- Segment distribution chart
- Customer results table
- Segment filter
- Customer testing controls
- Manual prediction form
- CSV download buttons

### Backend

Python handles:

- Loading the model bundle
- Reading and validating uploaded data
- Applying preprocessing
- Running predictions
- Preparing results for display and download

Both the interface and processing logic are implemented in `app.py`. The project does not use a separate Flask or FastAPI service.

## Important Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `customer_model.pkl` | Trained model and preprocessing bundle |
| `requirements.txt` | Application dependencies |
| `customer_segmentation_final.csv` | Exported customer dataset with segment labels |
| Jupyter notebook (`.ipynb`) | Data analysis and model development |

The model bundle contains the trained K-Means model, scaler, feature order, segment names, outlier limits, default input values and reference date.

The application requires `customer_model.pkl` beside `app.py`. The exported customer CSV is optional for deployment because users upload their own prepared data.

## Run Locally

Install the dependencies:

```bash
python -m pip install -r requirements.txt
