"""Streamlit application for nearby tract data and homeowner planning estimates.

The page loads a processed census-tract table, selects a nearby source record, and
either displays that tract's stored totals or performs the documented home-scale
calculation. It does not train or use a saved machine-learning model on submission.
"""

import os
import sys

import pandas as pd
import streamlit as st

# This file is run directly by Streamlit from `src/app`. Add the repository root so
# imports such as `src.models.random_forest` work without packaging the project first.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.random_forest import (
    load_data,
    predict_with_model,
    resolve_zip_code,
    validate_inputs,
)

# Configure the page before creating widgets, as required by Streamlit.
st.set_page_config(page_title="Solar Potential Predictor", page_icon="☀️", layout="centered")

# Everything above the form is static explanatory context. Streamlit reruns this file
# from top to bottom whenever a widget changes, so no data or network work happens yet.
st.title("Solar Potential Predictor")
st.write("Compare area-scale solar potential with a planning estimate for one home.")
st.caption("Community results use the nearest Project Sunroof census tract. Homeowner results add your roof-area input and are not a site design.")

st.info(
    "**Community Solar Potential** reports the nearby census tract's total potential. "
    "**Residential Solar Recommendation** estimates a single home's system from roof area and local solar yield."
)

# Keep selectors outside the form so changing either one immediately redraws the
# relevant fields instead of waiting for a form submission.
prediction_mode = st.radio(
    "Choose analysis type",
    ["community", "homeowner"],
    format_func=lambda mode: "Community Solar Potential Estimate" if mode == "community" else "Residential Solar Recommendation",
    horizontal=True,
)
location_method = st.radio(
    "Choose location input",
    ["Coordinates", "ZIP code"],
    horizontal=True,
    help="Coordinates select the closest tract to a specific point. ZIP codes use an approximate ZIP-centroid lookup.",
)

# Both modes require a location. The homeowner-only inputs appear only when they
# affect the calculation, keeping the community workflow focused on tract-level data.
with st.form("prediction_form"):
    latitude = None
    longitude = None
    zip_code = ""
    if location_method == "Coordinates":
        latitude = st.number_input("Latitude", value=25.68, format="%.4f")
        longitude = st.number_input("Longitude", value=-80.31, format="%.4f")
        st.caption("Coordinates provide the most specific nearest-tract match available in this app.")
    else:
        # A ZIP is resolved after validation because that lookup uses a network service.
        zip_code = st.text_input("US ZIP code", max_chars=5, placeholder="e.g. 33156")
        st.caption("ZIP lookup uses an approximate ZIP centroid. Use coordinates for a more specific result.")
    # Defaults preserve a valid call even in community mode, where these home-only
    # values are intentionally ignored by the calculation helper.
    orientation = "South"
    roof_area_sqft = 1800.0
    shading_level = "Unknown"
    monthly_electricity_kwh = None

    if prediction_mode == "homeowner":
        st.subheader("Home inputs")
        roof_area_sqft = st.number_input(
            "Approximate total roof area (sq ft)",
            min_value=250.0,
            value=1800.0,
            step=100.0,
            help="Sets the roof-area planning capacity. Use a rough total; the app does not measure usable roof geometry.",
        )
        orientation = st.selectbox(
            "Main usable roof orientation",
            ["South", "West", "East", "North"],
            help="Adjusts estimated home production. It does not affect the community result.",
        )
        shading_level = st.selectbox(
            "Broad shading around the usable roof (optional)",
            ["Unknown", "Minimal", "Moderate", "Significant"],
            help="Applies a broad production adjustment. Choose Unknown when you are not sure; it applies no extra adjustment.",
        )
        monthly_electricity_kwh = st.number_input(
            "Average monthly electricity use in kWh (optional)",
            min_value=1.0,
            value=None,
            step=25.0,
            help="When supplied, the recommended system is sized toward your annual electricity use, without exceeding the roof-area capacity.",
        )
        st.caption(
            "Without electricity use, the estimate uses roof area. With it, the estimate targets annual use; shading and orientation adjust expected production."
        )
    else:
        st.caption(
            "Community results use the nearest tract's stored totals. Home-specific roof, shading, and electricity-use inputs are intentionally not used."
        )
    submitted = st.form_submit_button("Predict")

if submitted:
    # Validate the form input first so the app can fail fast on bad values before any
    # data lookup or ZIP network request begins.
    try:
        validated = validate_inputs(
            latitude=latitude,
            longitude=longitude,
            zip_code=zip_code,
            orientation=orientation,
            shading_level=shading_level,
            monthly_electricity_kwh=monthly_electricity_kwh,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if validated["zip_code"]:
        # Resolve only a submitted ZIP. Coordinate users do not need a network request.
        try:
            resolved_latitude, resolved_longitude = resolve_zip_code(validated["zip_code"])
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        st.info(
            f"Using the approximate center of ZIP code {validated['zip_code']}. "
            "Use coordinates for a more specific tract match."
        )
    else:
        resolved_latitude = validated["latitude"]
        resolved_longitude = validated["longitude"]

    # Load one consistent processed table, then pass the final coordinates to the
    # source-record estimator. This avoids retraining or mutating data in the UI.
    df = load_data()
    prediction = predict_with_model(
        None,
        df,
        resolved_latitude,
        resolved_longitude,
        validated["orientation"],
        prediction_mode=prediction_mode,
        roof_area_sqft=roof_area_sqft,
        shading_level=validated["shading_level"],
        monthly_electricity_kwh=validated["monthly_electricity_kwh"],
    )

    # The payload has a common core for both modes; homeowner mode appends its own
    # calculation details. Render the shared title/message before branching on mode.
    st.subheader(prediction["prediction_title"])
    st.caption(prediction["mode_message"])
    col1, col2, col3 = st.columns(3)
    if prediction_mode == "homeowner":
        col1.metric("Estimated Annual Home Production", f"{prediction['annual_generation_kwh']:,.0f} kWh")
        col2.metric("Estimated Household Carbon Reduction", f"{prediction['carbon_offset_metric_tons']:.1f} tons")
        col3.metric("Recommended System Size", f"{prediction['recommended_system_kw']:.1f} kW")
        with st.expander("Homeowner assumptions"):
            st.write(f"Roof area used: {prediction['roof_area_sqft']:.0f} sq ft")
            st.write(f"Sizing basis: {prediction['sizing_basis'].capitalize()}")
            st.write(f"Shading input: {prediction['shading_level']} ({prediction['shading_multiplier']:.0%} production factor)")
            st.write(f"Estimated panel count: {prediction['estimated_panels']} panels")
            if prediction["annual_electricity_usage_kwh"] is not None:
                st.write(f"Annual electricity use entered: {prediction['annual_electricity_usage_kwh']:,.0f} kWh")
                st.write(f"Estimated usage offset: {prediction['estimated_usage_offset_percent']:.0f}%")
                if prediction["roof_area_limits_usage_target"]:
                    st.warning("The roof-area planning capacity is below the size needed to offset all entered annual use.")
    else:
        col1.metric("Potential Annual Energy Generation", f"{prediction['annual_generation_kwh']:,.0f} kWh")
        col2.metric("Potential Carbon Reduction", f"{prediction['carbon_offset_metric_tons']:.1f} tons")
        col3.metric("Potential Solar Capacity", f"{prediction['recommended_system_kw']:.1f} kW")

    # This section intentionally exposes tract proximity and directional context so a
    # user can see what source record informed the planning estimate.
    st.subheader("Prediction Summary")
    st.write(f"Best orientation: {max(prediction['orientation_rankings'], key=prediction['orientation_rankings'].get)}")
    st.write("Use this as a planning guide, not as a substitute for a site survey or engineering review.")
    st.write(f"Geographic match: {prediction['confidence_level']}")
    st.write(prediction['confidence_message'])

    st.subheader("Nearest Tract")
    st.write(
        f"Lat: {prediction['nearest_tract']['lat_avg']:.3f} | Lng: {prediction['nearest_tract']['lng_avg']:.3f} "
        f"| {prediction['nearest_distance_km']:.1f} km from the requested point"
    )

    st.subheader("Orientation Comparison")
    # Convert the dictionary to a labelled table because Streamlit's bar chart expects
    # a dataframe index for the category axis.
    orientation_df = pd.DataFrame(
        {
            "Orientation": list(prediction["orientation_rankings"].keys()),
            "Estimated output": list(prediction["orientation_rankings"].values()),
        }
    )
    st.bar_chart(orientation_df.set_index("Orientation"))
