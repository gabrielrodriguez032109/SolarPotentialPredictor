import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.random_forest import (
    load_data,
    load_or_train_model,
    predict_with_model,
    validate_inputs,
)

st.set_page_config(page_title="Solar Potential Predictor", page_icon="☀️", layout="centered")

st.title("Solar Potential Predictor")
st.write("Estimate annual solar generation, carbon offset, and system size for a location using historical Project Sunroof data.")

with st.form("prediction_form"):
    latitude = st.number_input("Latitude", value=25.68, format="%.4f")
    longitude = st.number_input("Longitude", value=-80.31, format="%.4f")
    zip_code = st.text_input("ZIP Code (optional)", placeholder="e.g. 33156")
    orientation = st.selectbox("Roof Orientation", ["South", "West", "East", "North"])
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        validated = validate_inputs(latitude=latitude, longitude=longitude, zip_code=zip_code, orientation=orientation)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    df = load_data()
    model = load_or_train_model(df)

    if validated["zip_code"]:
        st.info("ZIP code input received; the app will use the nearest available census tract for the estimate.")

    prediction = predict_with_model(
        model,
        df,
        validated["latitude"] if validated["latitude"] is not None else 0.0,
        validated["longitude"] if validated["longitude"] is not None else 0.0,
        validated["orientation"],
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Annual Generation", f"{prediction['annual_generation_kwh']:,.0f} kWh")
    col2.metric("Carbon Offset", f"{prediction['carbon_offset_metric_tons']:.1f} tons")
    col3.metric("System Size", f"{prediction['recommended_system_kw']:.1f} kW")

    st.subheader("Prediction Summary")
    st.write(f"Best orientation: {max(prediction['orientation_rankings'], key=prediction['orientation_rankings'].get)}")
    st.write(f"Confidence: {prediction['confidence_level']}")
    st.write(prediction['confidence_message'])

    st.subheader("Nearest Tract")
    st.write(
        f"Lat: {prediction['nearest_tract']['lat_avg']:.3f} | Lng: {prediction['nearest_tract']['lng_avg']:.3f}"
    )

    st.subheader("Orientation Comparison")
    orientation_df = pd.DataFrame(
        {
            "Orientation": list(prediction["orientation_rankings"].keys()),
            "Estimated output": list(prediction["orientation_rankings"].values()),
        }
    )
    st.bar_chart(orientation_df.set_index("Orientation"))
