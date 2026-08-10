import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load model
model = joblib.load("house_price_model.pkl")


# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)


# Title
st.title("🏠 House Price Prediction")
st.write("Enter the house details to predict its price.")


# User inputs
bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1.0,
    max_value=10.0,
    value=2.0
)

sqft_living = st.number_input(
    "Living Area (sqft)",
    min_value=100,
    value=1500
)

sqft_lot = st.number_input(
    "Lot Area (sqft)",
    min_value=100,
    value=5000
)

floors = st.number_input(
    "Floors",
    min_value=1.0,
    max_value=4.0,
    value=1.0
)

waterfront = st.selectbox(
    "Waterfront",
    [0, 1]
)

view = st.number_input(
    "View",
    min_value=0,
    max_value=4,
    value=0
)

condition = st.number_input(
    "Condition",
    min_value=1,
    max_value=5,
    value=3
)

sqft_above = st.number_input(
    "Above Ground Area (sqft)",
    min_value=100,
    value=1500
)

sqft_basement = st.number_input(
    "Basement Area (sqft)",
    min_value=0,
    value=0
)

yr_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000
)

yr_renovated = st.number_input(
    "Year Renovated",
    min_value=0,
    max_value=2026,
    value=0
)


# Prediction
if st.button("Predict House Price"):

    new_data = pd.DataFrame([[
        bedrooms,
        bathrooms,
        sqft_living,
        sqft_lot,
        floors,
        waterfront,
        view,
        condition,
        sqft_above,
        sqft_basement,
        yr_built,
        yr_renovated
    ]], columns=[
        'bedrooms',
        'bathrooms',
        'sqft_living',
        'sqft_lot',
        'floors',
        'waterfront',
        'view',
        'condition',
        'sqft_above',
        'sqft_basement',
        'yr_built',
        'yr_renovated'
    ])


    # Predict log(price)
    prediction = model.predict(new_data)

    # Convert back to original price
    actual_price = np.expm1(prediction[0])


    st.success(
        f"🏠 Predicted House Price: ${actual_price:,.2f}"
    )