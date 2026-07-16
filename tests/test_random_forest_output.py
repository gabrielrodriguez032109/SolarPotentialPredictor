from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.random_forest import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    engineer_features,
    format_prediction_summary,
    predict_with_model,
    validate_inputs,
)


def test_format_prediction_summary_contains_user_facing_metrics():
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

    summary = format_prediction_summary(predictions)

    assert "Estimated annual generation" in summary
    assert "15,600 kWh/year" in summary
    assert "Carbon offset" in summary
    assert "7.2 metric tons/year" in summary
    assert "Recommended system size" in summary
    assert "8.5 kW" in summary
    assert "South-facing placement" in summary


def test_validate_inputs_accepts_valid_latitude_and_longitude():
    validated = validate_inputs(latitude=25.68, longitude=-80.31, zip_code="", orientation="South")

    assert validated["latitude"] == 25.68
    assert validated["longitude"] == -80.31
    assert validated["orientation"] == "South"


def test_validate_inputs_rejects_invalid_latitude():
    try:
        validate_inputs(latitude=95.0, longitude=-80.31, zip_code="", orientation="South")
    except ValueError as exc:
        assert "Latitude" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid latitude")


def test_engineer_features_adds_useful_solar_features():
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

    model = RandomForestRegressor(random_state=42)
    X = reference_df[FEATURE_COLUMNS].astype(float)
    y = reference_df[TARGET_COLUMNS].astype(float)
    model.fit(X, y)

    prediction = predict_with_model(model, reference_df, latitude=25.68, longitude=-80.31, orientation="South")

    assert prediction["annual_generation_kwh"] > 0
    assert prediction["carbon_offset_metric_tons"] > 0
    assert prediction["recommended_system_kw"] > 0
    assert "South" in prediction["orientation_rankings"]
