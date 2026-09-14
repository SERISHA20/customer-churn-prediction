import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Page settings
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")

# Load dataset
data = pd.read_csv(
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


data = data.drop("customerID", axis=1)

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)
data["TotalCharges"] = data["TotalCharges"].fillna(0)

X = data.drop("Churn", axis=1)

# Convert categorical data
X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

# Create scaler
scaler = StandardScaler()
scaler.fit(X)

# Load model
model = tf.keras.models.load_model(
    "churn_model.keras"
)

st.subheader("Customer Details")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

phone = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

stream_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

stream_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)

# Prediction button
if st.button("🔍 Predict Churn"):

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # Convert to same format as training data
    customer = pd.get_dummies(
        customer,
        drop_first=True,
        dtype=int
    )

    # Match training columns
    customer = customer.reindex(
        columns=X.columns,
        fill_value=0
    )

    # Scale
    customer = scaler.transform(customer)

    # Predict
    probability = model.predict(
        customer,
        verbose=0
    )[0][0]

    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: **{probability * 100:.2f}%**"
    )

    if probability >= 0.5:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to NOT CHURN")
