"""Regression tests for the reusable estimation and ETL helpers.

The tests use deliberately tiny synthetic tracts. That keeps calculation expectations
exact and makes them independent of the large, versioned Project Sunroof export.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.etl.pipeline import transform
from src.models.random_forest import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    call_predict_with_model,
    engineer_features,
    format_prediction_summary,
    predict_with_model,
    resolve_zip_code,
    validate_inputs,
    convert_to_homeowner_estimate,
)


def test_format_prediction_summary_contains_user_facing_metrics():
    """The text helper should use community labels and readable units."""
    predictions = {
        "annual_generation_kwh": 15600,
        "carbon_offset_metric_tons": 7.2,
        "recommended_system_kw": 8.5,
        "orientation_rankings": {
            "South": 15600,
            "West": 13800,
            "East": 13500,
            "North": 10900,
        },
    }

    summary = format_prediction_summary(predictions, prediction_mode="community")

    assert "Potential annual energy generation" in summary
    assert "15,600 kWh/year" in summary
    assert "Potential carbon reduction" in summary
    assert "7.2 metric tons/year" in summary
    assert "Potential solar capacity" in summary
    assert "8.5 kW" in summary
    assert "South-facing placement" in summary


def test_format_prediction_summary_uses_homeowner_language():
    """The same helper should switch labels when rendering homeowner results."""
    predictions = {
        "annual_generation_kwh": 13200,
        "carbon_offset_metric_tons": 6.1,
        "recommended_system_kw": 7.8,
        "orientation_rankings": {
            "South": 13200,
            "West": 11800,
            "East": 11600,
            "North": 9500,
        },
    }

    summary = format_prediction_summary(predictions, prediction_mode="homeowner")

    assert "Estimated annual home production" in summary
    assert "Estimated household carbon reduction" in summary
    assert "Recommended system size" in summary


def test_validate_inputs_accepts_valid_latitude_and_longitude():
    """Coordinate submissions are normalized into the shared validation payload."""
    validated = validate_inputs(latitude=25.68, longitude=-80.31, zip_code="", orientation="South")

    assert validated["latitude"] == 25.68
    assert validated["longitude"] == -80.31
    assert validated["orientation"] == "South"


def test_validate_inputs_rejects_invalid_latitude():
    """Out-of-range geography fails before any database or network work occurs."""
    try:
        validate_inputs(latitude=95.0, longitude=-80.31, zip_code="", orientation="South")
    except ValueError as exc:
        assert "Latitude" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid latitude")


def test_validate_inputs_accepts_a_five_digit_zip_code():
    """ZIP validation defers coordinate lookup until the caller requests resolution."""
    validated = validate_inputs(zip_code="33156")

    assert validated["zip_code"] == "33156"
    assert validated["latitude"] is None
    assert validated["longitude"] is None


def test_resolve_zip_code_uses_the_returned_centroid():
    """Inject a fake HTTP client so ZIP parsing is tested without live networking."""
    class FakeResponse:
        def read(self):
            return b'{"places": [{"latitude": "25.679", "longitude": "-80.308"}]}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_urlopen(url, timeout):
        assert url.endswith("/33156")
        assert timeout == 5
        return FakeResponse()

    assert resolve_zip_code("33156", urlopen_fn=fake_urlopen) == (25.679, -80.308)


def test_validate_inputs_rejects_invalid_optional_homeowner_inputs():
    """Optional homeowner inputs still must obey the documented allowed values."""
    try:
        validate_inputs(
            latitude=25.68,
            longitude=-80.31,
            shading_level="Heavy",
            monthly_electricity_kwh=600,
        )
    except ValueError as exc:
        assert "Shading" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an invalid shading level")


def test_engineer_features_adds_useful_solar_features():
    """Directional values produce an additive total and south-to-north ratio."""
    source_df = pd.DataFrame(
        [
            {
                "yearly_sunlight_kwh_n": 1000,
                "yearly_sunlight_kwh_e": 1200,
                "yearly_sunlight_kwh_s": 1500,
                "yearly_sunlight_kwh_w": 1100,
            }
        ]
    )

    engineered = engineer_features(source_df)

    assert "sunlight_total_directional" in engineered.columns
    assert "south_to_north_ratio" in engineered.columns
    assert engineered.loc[0, "sunlight_total_directional"] == 4800
    assert engineered.loc[0, "south_to_north_ratio"] == 1.5


def test_predict_with_model_returns_prediction_payload():
    """Community mode returns direct values from the one available source tract."""
    reference_df = pd.DataFrame(
        [
            {
                "lat_avg": 25.68,
                "lng_avg": -80.31,
                "count_qualified": 120,
                "percent_covered": 65.0,
                "percent_qualified": 80.0,
                "yearly_sunlight_kwh_n": 1000,
                "yearly_sunlight_kwh_e": 1200,
                "yearly_sunlight_kwh_s": 1500,
                "yearly_sunlight_kwh_w": 1100,
                "yearly_sunlight_kwh_kw_threshold_avg": 1400,
                "yearly_sunlight_kwh_total": 15000,
                "carbon_offset_metric_tons": 7.0,
                "kw_total": 8.5,
            }
        ]
    )

    reference_df = engineer_features(reference_df)

    # Fit a tiny model only to prove the public helper accepts the historical argument;
    # the assertions below confirm it returns source values rather than model output.
    model = RandomForestRegressor(random_state=42)
    X = reference_df[FEATURE_COLUMNS].astype(float).to_numpy()
    y = reference_df[TARGET_COLUMNS].astype(float).to_numpy()
    model.fit(X, y)

    prediction = predict_with_model(None, reference_df, latitude=25.68, longitude=-80.31, orientation="North")

    assert prediction["annual_generation_kwh"] == 15000
    assert prediction["carbon_offset_metric_tons"] == 7.0
    assert prediction["recommended_system_kw"] == 8.5
    assert prediction["nearest_distance_km"] == 0.0
    assert "South" in prediction["orientation_rankings"]


def test_homeowner_prediction_uses_roof_area_and_local_solar_yield():
    """Homeowner mode converts one tract's yield into a roof-area-capped estimate."""
    reference_df = pd.DataFrame(
        [{
            "lat_avg": 25.68,
            "lng_avg": -80.31,
            "count_qualified": 120,
            "percent_covered": 65.0,
            "percent_qualified": 80.0,
            "yearly_sunlight_kwh_n": 1000,
            "yearly_sunlight_kwh_e": 1200,
            "yearly_sunlight_kwh_s": 1500,
            "yearly_sunlight_kwh_w": 1100,
            "yearly_sunlight_kwh_kw_threshold_avg": 1400,
            "yearly_sunlight_kwh_total": 15000,
            "carbon_offset_metric_tons": 7.0,
            "kw_total": 8.5,
        }]
    )

    prediction = predict_with_model(
        None,
        reference_df,
        latitude=25.68,
        longitude=-80.31,
        orientation="South",
        prediction_mode="homeowner",
        roof_area_sqft=1800,
    )

    assert prediction["recommended_system_kw"] == 7.2
    assert prediction["annual_generation_kwh"] == 10080.0
    assert prediction["carbon_offset_metric_tons"] == 4.704
    assert prediction["estimated_panels"] == 18


def test_homeowner_usage_and_shading_inputs_adjust_the_recommendation():
    """Usage sizing and broad shading change only the homeowner calculation."""
    reference_df = pd.DataFrame(
        [{
            "lat_avg": 25.68,
            "lng_avg": -80.31,
            "count_qualified": 120,
            "percent_covered": 65.0,
            "percent_qualified": 80.0,
            "yearly_sunlight_kwh_n": 1000,
            "yearly_sunlight_kwh_e": 1200,
            "yearly_sunlight_kwh_s": 1500,
            "yearly_sunlight_kwh_w": 1100,
            "yearly_sunlight_kwh_kw_threshold_avg": 1400,
            "yearly_sunlight_kwh_total": 15000,
            "carbon_offset_metric_tons": 7.0,
            "kw_total": 8.5,
        }]
    )

    prediction = predict_with_model(
        None,
        reference_df,
        latitude=25.68,
        longitude=-80.31,
        orientation="South",
        prediction_mode="homeowner",
        roof_area_sqft=1800,
        shading_level="Moderate",
        monthly_electricity_kwh=600,
    )

    assert prediction["sizing_basis"] == "annual electricity-use target"
    assert prediction["recommended_system_kw"] < 7.2
    assert prediction["annual_generation_kwh"] == 7200.0
    assert prediction["estimated_usage_offset_percent"] == 100.0
    assert prediction["shading_multiplier"] == 0.85


def test_community_prediction_ignores_homeowner_only_inputs():
    """Community source totals remain unchanged by household-specific arguments."""
    reference_df = pd.DataFrame(
        [{
            "lat_avg": 25.68,
            "lng_avg": -80.31,
            "count_qualified": 120,
            "percent_covered": 65.0,
            "percent_qualified": 80.0,
            "yearly_sunlight_kwh_n": 1000,
            "yearly_sunlight_kwh_e": 1200,
            "yearly_sunlight_kwh_s": 1500,
            "yearly_sunlight_kwh_w": 1100,
            "yearly_sunlight_kwh_kw_threshold_avg": 1400,
            "yearly_sunlight_kwh_total": 15000,
            "carbon_offset_metric_tons": 7.0,
            "kw_total": 8.5,
        }]
    )

    prediction = predict_with_model(
        None,
        reference_df,
        latitude=25.68,
        longitude=-80.31,
        prediction_mode="community",
        roof_area_sqft=500,
        shading_level="Significant",
        monthly_electricity_kwh=1500,
    )

    assert prediction["annual_generation_kwh"] == 15000
    assert prediction["recommended_system_kw"] == 8.5


def test_transform_keeps_required_columns_for_model_training():
    """ETL retains the minimum fields consumed by the app and evaluation paths."""
    raw_df = pd.DataFrame(
        [
            {
                "lat_avg": 25.68,
                "lng_avg": -80.31,
                "count_qualified": 120,
                "percent_covered": 65.0,
                "percent_qualified": 80.0,
                "yearly_sunlight_kwh_n": 1000,
                "yearly_sunlight_kwh_e": 1200,
                "yearly_sunlight_kwh_s": 1500,
                "yearly_sunlight_kwh_w": 1100,
                "yearly_sunlight_kwh_kw_threshold_avg": 1400,
                "yearly_sunlight_kwh_total": 15000,
                "carbon_offset_metric_tons": 7.0,
                "kw_total": 8.5,
            }
        ]
    )

    transformed = transform(raw_df)

    assert {
        "lat_avg",
        "lng_avg",
        "yearly_sunlight_kwh_total",
        "carbon_offset_metric_tons",
        "kw_total",
    }.issubset(transformed.columns)


def test_convert_to_homeowner_estimate_scales_down_area_potential():
    """A huge tract-scale capacity is capped to the helper's residential range."""
    estimate = convert_to_homeowner_estimate(
        annual_generation_kwh=12000000.0,
        carbon_offset_metric_tons=5000.0,
        recommended_system_kw=6800.0,
    )

    assert estimate["recommended_system_kw"] < 6800.0
    assert estimate["annual_generation_kwh"] > 0
    assert estimate["carbon_offset_metric_tons"] > 0


def test_convert_to_homeowner_estimate_uses_roof_area_for_a_more_realistic_home_result():
    """A normal roof produces a positive, home-scale panel and capacity result."""
    estimate = convert_to_homeowner_estimate(
        annual_generation_kwh=15000.0,
        carbon_offset_metric_tons=7.0,
        recommended_system_kw=8.5,
        roof_area_sqft=1800.0,
    )

    assert estimate["recommended_system_kw"] > 4.0
    assert estimate["recommended_system_kw"] < 8.5
    assert estimate["annual_generation_kwh"] > 0
    assert estimate["estimated_panels"] > 0


def test_call_predict_with_model_falls_back_for_older_signatures():
    """The compatibility wrapper can still call a pre-homeowner prediction helper."""
    def legacy_predict_with_model(model, training_df, latitude, longitude, orientation):
        return {"prediction_title": "legacy", "annual_generation_kwh": 1000.0}

    result = call_predict_with_model(
        model=None,
        training_df=None,
        latitude=25.68,
        longitude=-80.31,
        orientation="South",
        prediction_mode="homeowner",
        roof_area_sqft=1800.0,
        predict_fn=legacy_predict_with_model,
    )

    assert result["prediction_title"] == "legacy"
    assert result["annual_generation_kwh"] == 1000.0
