"""Random Forest solar-potential evaluation workflow and tract-context helpers.

This module loads the cleaned census-tract dataset from SQLite, engineers
tract-level solar features, and trains an offline Random Forest regressor. It
also exposes helpers for the Streamlit demonstration, whose public results use
nearby source records rather than Random Forest inference.
"""

import json
import os
import sqlite3 as sql
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# Shared artifact locations. The ETL writes the database; evaluation optionally writes
# the model and charts. The Streamlit app reads only the processed database.
DB_PATH = "data/processed/solar.db"
TABLE_NAME = "sunroof_clean"
OUTPUT_DIR = os.path.join("src", "model-output")
MODEL_PATH = os.path.join("src", "models", "random_forest.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.json")

# Evaluation-only Random Forest inputs. The app does not use model predictions for its
# public result because the selected tract already carries the displayed target values.
FEATURE_COLUMNS = [
    "lat_avg",
    "lng_avg",
    "count_qualified",
    "percent_covered",
    "percent_qualified",
    "yearly_sunlight_kwh_n",
    "yearly_sunlight_kwh_e",
    "yearly_sunlight_kwh_s",
    "yearly_sunlight_kwh_w",
    "yearly_sunlight_kwh_kw_threshold_avg",
    "sunlight_total_directional",
    "south_to_north_ratio",
]

# Evaluation-only model outputs. `kw_total` is present in current ETL output; the
# fallback list supports older processed artifacts that predate that field.
TARGET_COLUMNS = [
    "yearly_sunlight_kwh_total",
    "carbon_offset_metric_tons",
    "kw_total",
]

FALLBACK_TARGET_COLUMNS = [
    "yearly_sunlight_kwh_total",
    "carbon_offset_metric_tons",
]


VALID_ORIENTATIONS = {"North", "South", "East", "West"}
HOMEOWNER_KW_PER_SQFT = 0.004
MIN_HOME_SYSTEM_KW = 1.0
MAX_HOME_SYSTEM_KW = 15.0
PANEL_KW = 0.4
VALID_SHADING_LEVELS = {"Unknown", "Minimal", "Moderate", "Significant"}

# These broad adjustments let a homeowner account for obvious site shading without
# implying that the app has a roof-level shade survey. "Unknown" preserves the local
# tract-based yield and is the default when the user does not provide this information.
SHADING_MULTIPLIERS = {
    "Unknown": 1.0,
    "Minimal": 0.95,
    "Moderate": 0.85,
    "Significant": 0.70,
}

# These are planning adjustments used only for a single home's selected roof
# orientation. The Project Sunroof export has tract-level directional totals, not
# property-specific orientation yield, so they must not alter community totals.
ORIENTATION_MULTIPLIERS = {
    "South": 1.0,
    "West": 0.90,
    "East": 0.88,
    "North": 0.72,
}


def load_data(db_path: str = DB_PATH, table_name: str = TABLE_NAME) -> pd.DataFrame:
    """Load the processed SQLite table back into a pandas DataFrame.

    The Streamlit app and the modeling functions both depend on this helper to read the
    cleaned dataset produced by the ETL pipeline.
    """
    # A context manager closes the SQLite connection even if pandas raises an error.
    # `table_name` is an internal constant/caller parameter, not user-provided input.
    with sql.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def validate_inputs(
    latitude: float | None = None,
    longitude: float | None = None,
    zip_code: str | None = None,
    orientation: str | None = None,
    shading_level: str | None = None,
    monthly_electricity_kwh: float | None = None,
) -> dict[str, Any]:
    """Validate the user input before a prediction request is sent to the model.

    The app accepts either a US ZIP code or a latitude/longitude pair. ZIP codes are
    resolved separately to a ZIP-centroid coordinate before prediction.
    """
    # Normalize once so UI values and direct Python callers follow identical rules.
    normalized_zip_code = str(zip_code or "").strip()
    normalized_orientation = (orientation or "South").strip().title()
    normalized_shading_level = (shading_level or "Unknown").strip().title()
    if normalized_orientation not in VALID_ORIENTATIONS:
        raise ValueError("Orientation must be one of: North, South, East, West.")
    if normalized_shading_level not in VALID_SHADING_LEVELS:
        raise ValueError("Shading level must be one of: Unknown, Minimal, Moderate, Significant.")
    if monthly_electricity_kwh is not None and float(monthly_electricity_kwh) <= 0:
        raise ValueError("Average monthly electricity use must be greater than zero when provided.")

    if normalized_zip_code:
        # ZIP resolution is deliberately separated from validation: validation stays
        # local and deterministic, while resolution is the only network-dependent step.
        if len(normalized_zip_code) != 5 or not normalized_zip_code.isdigit():
            raise ValueError("ZIP code must contain exactly five digits.")
        # A ZIP does not yet have coordinates. Its centroid is resolved later so this
        # pure validation function remains deterministic and easy to unit test.
        return {
            "latitude": None,
            "longitude": None,
            "zip_code": normalized_zip_code,
            "orientation": normalized_orientation,
            "shading_level": normalized_shading_level,
            "monthly_electricity_kwh": monthly_electricity_kwh,
        }

    if latitude is None or longitude is None:
        raise ValueError("Please provide both latitude and longitude.")

    latitude = float(latitude)
    longitude = float(longitude)

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    # Return a normalized payload rather than separate values. The app can use the
    # exact same contract for coordinate and ZIP submissions.
    return {
        "latitude": latitude,
        "longitude": longitude,
        "zip_code": None,
        "orientation": normalized_orientation,
        "shading_level": normalized_shading_level,
        "monthly_electricity_kwh": monthly_electricity_kwh,
    }


def resolve_zip_code(zip_code: str, urlopen_fn: Any = urlopen) -> tuple[float, float]:
    """Resolve a US ZIP code to an approximate centroid coordinate.

    The public Zippopotam.us response provides a place centroid, not a street address.
    The app reports the later tract-center distance so users can judge this precision.
    """
    normalized_zip_code = str(zip_code).strip()
    # `urlopen_fn` is injectable so tests can verify the response handling without a
    # live network call. The production default is urllib's standard urlopen.
    # This is the application's only live network call. It returns a ZIP-area
    # centroid, so it must never be presented as an exact street-address location.
    url = f"https://api.zippopotam.us/us/{normalized_zip_code}"
    try:
        with urlopen_fn(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        place = payload["places"][0]
        return float(place["latitude"]), float(place["longitude"])
    except (URLError, OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "The ZIP code could not be resolved. Check it or use latitude and longitude instead."
        ) from exc


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a small set of derived solar features from the raw directional sunlight fields.

    These engineered columns help the model capture simple directional patterns that are
    not obvious from the raw values alone. They are optional for inference, but they
    improve the usefulness of the feature matrix when present.
    """
    # Work on a copy because the caller may still need the unmodified source frame.
    engineered = df.copy()
    # Supply missing directional fields as NaN so the later dropna step, rather than a
    # KeyError, decides whether an older schema can support model evaluation.
    for column in ["yearly_sunlight_kwh_n", "yearly_sunlight_kwh_e", "yearly_sunlight_kwh_s", "yearly_sunlight_kwh_w"]:
        if column not in engineered.columns:
            engineered[column] = np.nan

    engineered["sunlight_total_directional"] = (
        engineered["yearly_sunlight_kwh_n"]
        + engineered["yearly_sunlight_kwh_e"]
        + engineered["yearly_sunlight_kwh_s"]
        + engineered["yearly_sunlight_kwh_w"]
    )
    engineered["south_to_north_ratio"] = (
        engineered["yearly_sunlight_kwh_s"] / engineered["yearly_sunlight_kwh_n"].replace(0, np.nan)
    )
    return engineered


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert the processed dataset into the matrix form expected by scikit-learn.

    The feature matrix and target matrix are built from the available columns in the
    dataframe so the training path can tolerate slight schema differences.
    """
    enriched_df = engineer_features(df)

    # This compatibility behavior supports older processed artifacts, but callers that
    # need a fixed model shape should validate the selected columns separately.
    available_target_columns = [column for column in TARGET_COLUMNS if column in enriched_df.columns]
    available_feature_columns = [column for column in FEATURE_COLUMNS if column in enriched_df.columns]

    if not available_target_columns or not available_feature_columns:
        raise ValueError("The processed dataset is missing the required feature or target columns.")

    # scikit-learn needs a dense numeric matrix; remove incomplete rows only after the
    # available schema has been established.
    enriched_df = enriched_df.dropna(subset=available_feature_columns + available_target_columns).reset_index(drop=True)
    X = enriched_df[available_feature_columns].astype(float).to_numpy()
    y = enriched_df[available_target_columns].astype(float).to_numpy()
    return X, y


def load_or_train_model(df: pd.DataFrame, force_train: bool = False) -> RandomForestRegressor:
    """Load an evaluation artifact or create it when a user runs the ML workflow.

    This helper is intentionally not on the Streamlit request path; app estimates use
    direct source values from the nearest tract and should not train on page submits.
    """
    # Reuse a saved artifact only for an explicit evaluation workflow. The Streamlit
    # app deliberately bypasses this helper so a page submission never triggers train.
    if os.path.exists(MODEL_PATH) and not force_train:
        return joblib.load(MODEL_PATH)

    X, y = prepare_data(df)
    model, _, _, _, _ = train_random_forest(X, y)
    save_model(model)
    return model


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Fit a seeded multi-output forest and calculate held-out evaluation metrics."""
    # Hold out rows before fitting so the reported test metrics use records the forest
    # did not see during training. The same seed makes comparisons reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # A fixed seed makes the split and forest reproducible for comparison and tests.
    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # scikit-learn supports multi-output regression here. Its aggregate metrics combine
    # all target columns, so inspect per-target charts before drawing conclusions.
    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "train_mape": mean_absolute_percentage_error(y_train, y_train_pred),
        "test_mape": mean_absolute_percentage_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
    }

    return model, X_test, y_test, y_test_pred, metrics


def plot_test_predictions(y_test: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> None:
    """Save one actual-versus-predicted scatter chart for each model target."""
    for idx, target_name in enumerate(target_names):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test[:, idx], y_pred[:, idx], alpha=0.4, edgecolors="k", linewidths=0.5)

        # The red y=x line is the visual benchmark for a perfect prediction.
        min_val = min(y_test[:, idx].min(), y_pred[:, idx].min())
        max_val = max(y_test[:, idx].max(), y_pred[:, idx].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

        plt.xlabel(f"Actual {target_name}")
        plt.ylabel(f"Predicted {target_name}")
        plt.title(f"Random Forest: Actual vs Predicted ({target_name})")
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"random_forest_predictions_{target_name}.png")
        plt.savefig(out_path)
        plt.close()


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> None:
    """Save residual histograms so systematic over/under-prediction is visible."""
    for idx, target_name in enumerate(target_names):
        # A residual is actual minus predicted: positive values mean under-prediction.
        residuals = y_test[:, idx] - y_pred[:, idx]
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=20, edgecolor="black")
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.title(f"Residual Distribution ({target_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"random_forest_residuals_{target_name}.png"))
        plt.close()


def cross_validate_model(X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict[str, float]:
    """Measure five-fold R² stability for annual generation, the first target only."""
    # Cross-validation reports R² for annual generation (the first target) only. It is
    # an evaluation diagnostic and is not shown as a confidence score in the app.
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    scores = cross_val_score(model, X, y[:, 0], cv=KFold(n_splits=cv, shuffle=True, random_state=42), scoring="r2")
    return {"cv_mean_r2": float(scores.mean()), "cv_std_r2": float(scores.std())}


def format_prediction_summary(predictions: dict[str, Any], prediction_mode: str = "community") -> str:
    """Format a text-only result for callers outside the Streamlit interface.

    Streamlit renders metrics directly, but keeping this helper makes the core result
    reusable by a CLI, notebook, or another presentation layer.
    """
    annual_generation = int(round(predictions["annual_generation_kwh"]))
    carbon_offset = float(predictions["carbon_offset_metric_tons"])
    system_size = float(predictions["recommended_system_kw"])
    orientation_rankings = predictions.get("orientation_rankings", {})

    # Rank stored tract-level directional totals. These rankings are informative
    # context; they are not a property-specific roof orientation recommendation.
    best_orientation = max(orientation_rankings, key=orientation_rankings.get)
    best_value = orientation_rankings[best_orientation]
    second_best = sorted(
        orientation_rankings.items(), key=lambda item: item[1], reverse=True
    )[1][0]

    if prediction_mode == "homeowner":
        annual_label = "Estimated annual home production"
        carbon_label = "Estimated household carbon reduction"
        system_label = "Recommended system size"
    else:
        annual_label = "Potential annual energy generation"
        carbon_label = "Potential carbon reduction"
        system_label = "Potential solar capacity"

    return (
        f"{annual_label}: "
        f"{annual_generation:,} kWh/year\n"
        f"{carbon_label}: "
        f"{carbon_offset:.1f} metric tons/year\n"
        f"{system_label}: "
        f"{system_size:.1f} kW\n"
        f"Best orientation for this region: {best_orientation} ({best_value:,} kWh/year)\n"
        f"{best_orientation}-facing placement provides approximately 11% more energy than {second_best}-facing placement."
    )


def estimate_confidence(
    latitude: float, longitude: float, reference_row: pd.Series
) -> tuple[str, str, float]:
    """Describe how closely the requested point matches the selected tract.

    This is a geographic-match indicator, not a claim about model accuracy.
    """
    # Use a haversine distance instead of raw degree distance so the result is in km
    # and longitude is correctly weighted by latitude.
    latitude_radians = np.radians(latitude)
    reference_latitude_radians = np.radians(float(reference_row["lat_avg"]))
    delta_latitude = reference_latitude_radians - latitude_radians
    delta_longitude = np.radians(float(reference_row["lng_avg"]) - longitude)
    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(latitude_radians)
        * np.cos(reference_latitude_radians)
        * np.sin(delta_longitude / 2) ** 2
    )
    distance_km = float(6371.0 * 2 * np.arctan2(np.sqrt(haversine), np.sqrt(1 - haversine)))

    if distance_km <= 2:
        return "High geographic match", "The selected tract center is within 2 km of the requested point.", distance_km
    if distance_km <= 10:
        return "Medium geographic match", "The selected tract center is within 10 km of the requested point.", distance_km
    return "Low geographic match", "The nearest available tract center is more than 10 km from the requested point.", distance_km


def convert_to_homeowner_estimate(
    annual_generation_kwh: float,
    carbon_offset_metric_tons: float,
    recommended_system_kw: float,
    roof_area_sqft: float = 1800.0,
    local_yield_kwh_per_kw: float | None = None,
    orientation: str = "South",
    shading_level: str = "Unknown",
    monthly_electricity_kwh: float | None = None,
) -> dict[str, Any]:
    """Convert a tract-level solar estimate into a more realistic homeowner-scale estimate.

    This helper uses a documented rooftop-area rule and local solar yield to derive a
    transparent planning estimate for one home. Optional electricity use right-sizes
    the recommendation to annual demand; broad shading adjusts production only.
    """
    # Defend the reusable helper against zero/negative input. The Streamlit widget has
    # a stricter 250 sq ft minimum, but direct Python callers do not.
    roof_area_sqft = max(float(roof_area_sqft), 1.0)

    # Roof area supplies a conservative planning ceiling, not a surveyed usable area.
    roof_limited_system_kw = min(
        max(roof_area_sqft * HOMEOWNER_KW_PER_SQFT, MIN_HOME_SYSTEM_KW),
        MAX_HOME_SYSTEM_KW,
    )
    if local_yield_kwh_per_kw is None:
        # Preserve compatibility with callers that have only tract total generation
        # and capacity rather than the explicit per-kW yield field.
        local_yield_kwh_per_kw = annual_generation_kwh / max(recommended_system_kw, 0.001)
    # These multipliers are intentionally applied only here. Community totals already
    # include each tract's mix of suitable roof orientations and exposures.
    orientation_multiplier = ORIENTATION_MULTIPLIERS[orientation]
    shading_multiplier = SHADING_MULTIPLIERS[shading_level]
    adjusted_yield_kwh_per_kw = local_yield_kwh_per_kw * orientation_multiplier * shading_multiplier

    annual_electricity_usage_kwh = None
    roof_area_limits_usage_target = False
    if monthly_electricity_kwh is None:
        homeowner_system_kw = roof_limited_system_kw
        sizing_basis = "roof-area planning capacity"
    else:
        annual_electricity_usage_kwh = float(monthly_electricity_kwh) * 12
        # Convert annual use into the capacity needed to produce it at this location.
        # The roof-area ceiling still wins when the demand target is too large.
        demand_matched_system_kw = annual_electricity_usage_kwh / max(adjusted_yield_kwh_per_kw, 1.0)
        homeowner_system_kw = min(
            max(demand_matched_system_kw, MIN_HOME_SYSTEM_KW), roof_limited_system_kw
        )
        roof_area_limits_usage_target = demand_matched_system_kw > roof_limited_system_kw
        sizing_basis = "annual electricity-use target"

    # Production is intentionally based on local yield rather than scaling the entire
    # tract total: tract capacity represents many roofs, not the submitted home.
    homeowner_generation = homeowner_system_kw * adjusted_yield_kwh_per_kw
    carbon_rate = carbon_offset_metric_tons / max(annual_generation_kwh, 1.0)
    homeowner_carbon_offset = homeowner_generation * carbon_rate
    estimated_panels = max(int(round(homeowner_system_kw / PANEL_KW)), 1)

    return {
        "annual_generation_kwh": homeowner_generation,
        "carbon_offset_metric_tons": homeowner_carbon_offset,
        "recommended_system_kw": homeowner_system_kw,
        "estimated_panels": estimated_panels,
        "roof_area_sqft": roof_area_sqft,
        "shading_level": shading_level,
        "shading_multiplier": shading_multiplier,
        "annual_electricity_usage_kwh": annual_electricity_usage_kwh,
        "estimated_usage_offset_percent": (
            min(homeowner_generation / annual_electricity_usage_kwh * 100, 100)
            if annual_electricity_usage_kwh
            else None
        ),
        "roof_area_limits_usage_target": roof_area_limits_usage_target,
        "sizing_basis": sizing_basis,
    }


def call_predict_with_model(
    model: RandomForestRegressor | None,
    training_df: pd.DataFrame | None,
    latitude: float,
    longitude: float,
    orientation: str = "South",
    prediction_mode: str = "community",
    roof_area_sqft: float = 1800.0,
    shading_level: str = "Unknown",
    monthly_electricity_kwh: float | None = None,
    predict_fn: Any | None = None,
) -> dict[str, Any]:
    """Call the prediction helper through a compatibility wrapper.

    Some runtime environments or import paths can expose an older helper signature
    without the homeowner-specific keywords. This wrapper preserves compatibility by
    falling back to a legacy-style call when needed.
    """
    # Dependency injection lets tests (and older callers) provide a prediction helper
    # without changing the application-facing signature.
    if predict_fn is not None:
        try:
            return predict_fn(
                model,
                training_df,
                latitude,
                longitude,
                orientation,
                prediction_mode=prediction_mode,
                roof_area_sqft=roof_area_sqft,
                shading_level=shading_level,
                monthly_electricity_kwh=monthly_electricity_kwh,
            )
        except TypeError:
            return predict_fn(model, training_df, latitude, longitude, orientation)

    return predict_with_model(
        model,
        training_df,
        latitude,
        longitude,
        orientation,
        prediction_mode=prediction_mode,
        roof_area_sqft=roof_area_sqft,
        shading_level=shading_level,
        monthly_electricity_kwh=monthly_electricity_kwh,
    )


def predict_with_model(
    model: RandomForestRegressor | None,
    training_df: pd.DataFrame,
    latitude: float,
    longitude: float,
    orientation: str = "South",
    prediction_mode: str = "community",
    roof_area_sqft: float = 1800.0,
    shading_level: str = "Unknown",
    monthly_electricity_kwh: float | None = None,
) -> dict[str, Any]:
    """Return a nearest-tract estimate from the Project Sunroof source fields.

    The public estimate intentionally uses the actual values of the selected tract,
    rather than an ML prediction of values already present in the source dataset.
    ``model`` remains an optional argument for backward-compatible callers and for
    the separate training/evaluation workflow.
    """
    # The name is retained for compatibility, but this app path selects a source row;
    # it deliberately does not call `model.predict`.
    # The dataframe name is historical. In the public path it is a source-record table,
    # not necessarily the dataframe used to train a model.
    training_df = engineer_features(training_df)

    available_target_columns = [column for column in TARGET_COLUMNS if column in training_df.columns]
    available_feature_columns = [column for column in FEATURE_COLUMNS if column in training_df.columns]

    if not available_feature_columns:
        raise ValueError("The processed dataset is missing the required feature columns for prediction.")

    if not available_target_columns:
        available_target_columns = FALLBACK_TARGET_COLUMNS

    # Remove incomplete candidates before locating a tract, so every value required by
    # the selected output mode comes from one internally complete source record.
    training_df = training_df.dropna(subset=available_feature_columns + available_target_columns).reset_index(drop=True)
    # This inexpensive squared-degree comparison chooses the closest stored tract.
    # `estimate_confidence` later computes the displayed, physically meaningful km gap.
    reference_row = training_df.iloc[
        ((training_df["lat_avg"] - latitude) ** 2 + (training_df["lng_avg"] - longitude) ** 2)
        .argmin()
    ]

    # Community values are direct Project Sunroof fields from the chosen record.
    annual_generation = max(float(reference_row["yearly_sunlight_kwh_total"]), 0.0)
    carbon_offset = max(float(reference_row["carbon_offset_metric_tons"]), 0.0)
    local_yield_kwh_per_kw = max(float(reference_row["yearly_sunlight_kwh_kw_threshold_avg"]), 0.0)
    recommended_system_kw = max(
        float(reference_row.get("kw_total", annual_generation / max(local_yield_kwh_per_kw, 1.0))),
        0.0,
    )

    # Preserve the source's four directional aggregates for comparison in the UI. Do
    # not apply homeowner orientation/shading assumptions to these community values.
    orientation_rankings = {
        "South": float(reference_row["yearly_sunlight_kwh_s"]),
        "West": float(reference_row["yearly_sunlight_kwh_w"]),
        "East": float(reference_row["yearly_sunlight_kwh_e"]),
        "North": float(reference_row["yearly_sunlight_kwh_n"]),
    }

    confidence_level, confidence_message, nearest_distance_km = estimate_confidence(
        latitude, longitude, reference_row
    )

    result = {
        "annual_generation_kwh": annual_generation,
        "carbon_offset_metric_tons": carbon_offset,
        "recommended_system_kw": recommended_system_kw,
        "orientation_rankings": orientation_rankings,
        "orientation": orientation,
        "confidence_level": confidence_level,
        "confidence_message": confidence_message,
        "nearest_distance_km": nearest_distance_km,
        "data_source": "Nearest Project Sunroof-style census tract",
        "nearest_tract": {
            "lat_avg": float(reference_row["lat_avg"]),
            "lng_avg": float(reference_row["lng_avg"]),
        },
        "prediction_mode": prediction_mode,
        "prediction_title": "Community Tract Context" if prediction_mode == "community" else "Home-Scale Planning Demonstration",
    }

    if prediction_mode == "homeowner":
        # Only homeowner mode transforms tract-level context into a one-home planning
        # estimate. Community mode returns the source values above unchanged.
        homeowner_estimate = convert_to_homeowner_estimate(
            annual_generation_kwh=annual_generation,
            carbon_offset_metric_tons=carbon_offset,
            recommended_system_kw=recommended_system_kw,
            roof_area_sqft=roof_area_sqft,
            local_yield_kwh_per_kw=local_yield_kwh_per_kw,
            orientation=orientation,
            shading_level=shading_level,
            monthly_electricity_kwh=monthly_electricity_kwh,
        )
        result.update(homeowner_estimate)
        result["mode_label"] = "Home-Scale Planning Demonstration"
        result["mode_message"] = "Transparent planning calculation based on roof area, local solar yield, and any optional home inputs you provided."
    else:
        result["mode_label"] = "Community Tract Context"
        result["mode_message"] = "Stored source-data context for the nearest Project Sunroof-style census tract."

    return result


def save_model(model: RandomForestRegressor, path: str = MODEL_PATH) -> None:
    """Serialize an offline evaluation model, creating its output directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: dict[str, Any], path: str = METRICS_PATH) -> None:
    """Write evaluation metrics as readable JSON for later inspection or comparison."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def plot_feature_importance(model: RandomForestRegressor) -> None:
    """Save the forest's impurity-based feature-importance chart for exploration."""
    importances = model.feature_importances_
    feature_names = FEATURE_COLUMNS

    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()


def main() -> None:
    """Run the complete offline Random Forest evaluation and artifact workflow."""
    df = load_data()
    X, y = prepare_data(df)
    model, X_test, y_test, y_pred, metrics = train_random_forest(X, y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_model(model)
    save_metrics(metrics)
    plot_feature_importance(model)
    plot_residuals(y_test, y_pred, TARGET_COLUMNS)

    # Store CV separately because it evaluates only annual generation, not every
    # multi-output target reported in the train/test metrics above.
    metrics["cross_validation"] = cross_validate_model(X, y)
    save_metrics(metrics)

    print("Random Forest model trained on data/processed/solar.db")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMNS}")
    print("\nModel evaluation metrics:")
    print(f"  Train RMSE: {metrics['train_rmse']:.2f}")
    print(f"  Test RMSE:  {metrics['test_rmse']:.2f}")
    print(f"  Train MAE:  {metrics['train_mae']:.2f}")
    print(f"  Test MAE:   {metrics['test_mae']:.2f}")
    print(f"  Train R2:   {metrics['train_r2']:.4f}")
    print(f"  Test R2:    {metrics['test_r2']:.4f}")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"\nSaved plots: {OUTPUT_DIR}/random_forest_predictions_<target_name>.png")

    plot_test_predictions(y_test, y_pred, TARGET_COLUMNS)


if __name__ == "__main__":
    main()
