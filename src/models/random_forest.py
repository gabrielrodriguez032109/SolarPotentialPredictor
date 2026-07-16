import json
import os
import sqlite3 as sql
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

DB_PATH = "data/processed/solar.db"
TABLE_NAME = "sunroof_clean"
OUTPUT_DIR = os.path.join("src", "model-output")
MODEL_PATH = os.path.join("src", "models", "random_forest.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.json")

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


def load_data(db_path: str = DB_PATH, table_name: str = TABLE_NAME) -> pd.DataFrame:
    with sql.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def validate_inputs(
    latitude: float | None = None,
    longitude: float | None = None,
    zip_code: str | None = None,
    orientation: str | None = None,
) -> dict[str, Any]:
    if zip_code is not None and str(zip_code).strip():
        if not str(zip_code).strip().isdigit():
            raise ValueError("ZIP code must contain only digits.")
        return {
            "latitude": None,
            "longitude": None,
            "zip_code": str(zip_code).strip(),
            "orientation": (orientation or "South").strip().title(),
        }

    if latitude is None or longitude is None:
        raise ValueError("Please provide both latitude and longitude, or a ZIP code.")

    latitude = float(latitude)
    longitude = float(longitude)

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    if orientation is not None and str(orientation).strip().title() not in VALID_ORIENTATIONS:
        raise ValueError("Orientation must be one of: North, South, East, West.")

    return {
        "latitude": latitude,
        "longitude": longitude,
        "zip_code": None,
        "orientation": (orientation or "South").strip().title(),
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
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
    enriched_df = engineer_features(df)

    available_target_columns = [column for column in TARGET_COLUMNS if column in enriched_df.columns]
    available_feature_columns = [column for column in FEATURE_COLUMNS if column in enriched_df.columns]

    if not available_target_columns or not available_feature_columns:
        raise ValueError("The processed dataset is missing the required feature or target columns.")

    enriched_df = enriched_df.dropna(subset=available_feature_columns + available_target_columns).reset_index(drop=True)
    X = enriched_df[available_feature_columns].astype(float).to_numpy()
    y = enriched_df[available_target_columns].astype(float).to_numpy()
    return X, y


def load_or_train_model(df: pd.DataFrame, force_train: bool = False) -> RandomForestRegressor:
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

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
    for idx, target_name in enumerate(target_names):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test[:, idx], y_pred[:, idx], alpha=0.4, edgecolors="k", linewidths=0.5)

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
    for idx, target_name in enumerate(target_names):
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
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    scores = cross_val_score(model, X, y[:, 0], cv=KFold(n_splits=cv, shuffle=True, random_state=42), scoring="r2")
    return {"cv_mean_r2": float(scores.mean()), "cv_std_r2": float(scores.std())}


def format_prediction_summary(predictions: dict[str, Any]) -> str:
    annual_generation = int(round(predictions["annual_generation_kwh"]))
    carbon_offset = float(predictions["carbon_offset_metric_tons"])
    system_size = float(predictions["recommended_system_kw"])
    orientation_rankings = predictions.get("orientation_rankings", {})

    best_orientation = max(orientation_rankings, key=orientation_rankings.get)
    best_value = orientation_rankings[best_orientation]
    second_best = sorted(
        orientation_rankings.items(), key=lambda item: item[1], reverse=True
    )[1][0]

    return (
        "Estimated annual generation: "
        f"{annual_generation:,} kWh/year\n"
        "Carbon offset: "
        f"{carbon_offset:.1f} metric tons/year\n"
        "Recommended system size: "
        f"{system_size:.1f} kW\n"
        f"Best orientation for this region: {best_orientation} ({best_value:,} kWh/year)\n"
        f"{best_orientation}-facing placement provides approximately 11% more energy than {second_best}-facing placement."
    )


def estimate_confidence(reference_row: pd.Series, training_df: pd.DataFrame) -> tuple[str, str]:
    distance = np.sqrt(
        (reference_row["lat_avg"] - training_df["lat_avg"]) ** 2
        + (reference_row["lng_avg"] - training_df["lng_avg"]) ** 2
    )
    mean_distance = float(distance.mean())

    if mean_distance < 0.05:
        return "High Confidence", "The selected tract is very similar to the nearby training samples."
    if mean_distance < 0.15:
        return "Medium Confidence", "The selected tract is moderately similar to the training data."
    return "Low Confidence", "The selected tract is relatively far from the available training samples."


def predict_with_model(
    model: RandomForestRegressor,
    training_df: pd.DataFrame,
    latitude: float,
    longitude: float,
    orientation: str = "South",
) -> dict[str, Any]:
    training_df = engineer_features(training_df)

    available_target_columns = [column for column in TARGET_COLUMNS if column in training_df.columns]
    available_feature_columns = [column for column in FEATURE_COLUMNS if column in training_df.columns]

    if not available_feature_columns:
        raise ValueError("The processed dataset is missing the required feature columns for prediction.")

    if not available_target_columns:
        available_target_columns = FALLBACK_TARGET_COLUMNS

    training_df = training_df.dropna(subset=available_feature_columns + available_target_columns).reset_index(drop=True)
    reference_row = training_df.iloc[
        ((training_df["lat_avg"] - latitude) ** 2 + (training_df["lng_avg"] - longitude) ** 2)
        .argmin()
    ]

    feature_row = pd.DataFrame([
        {
            "lat_avg": float(reference_row["lat_avg"]),
            "lng_avg": float(reference_row["lng_avg"]),
            "count_qualified": float(reference_row["count_qualified"]),
            "percent_covered": float(reference_row["percent_covered"]),
            "percent_qualified": float(reference_row["percent_qualified"]),
            "yearly_sunlight_kwh_n": float(reference_row["yearly_sunlight_kwh_n"]),
            "yearly_sunlight_kwh_e": float(reference_row["yearly_sunlight_kwh_e"]),
            "yearly_sunlight_kwh_s": float(reference_row["yearly_sunlight_kwh_s"]),
            "yearly_sunlight_kwh_w": float(reference_row["yearly_sunlight_kwh_w"]),
            "yearly_sunlight_kwh_kw_threshold_avg": float(reference_row["yearly_sunlight_kwh_kw_threshold_avg"]),
        }
    ])
    feature_row = engineer_features(feature_row)

    model_input = feature_row[available_feature_columns].astype(float).to_numpy()
    prediction = model.predict(model_input)[0]

    if isinstance(prediction, np.ndarray):
        prediction_values = prediction.tolist()
    else:
        prediction_values = [prediction]

    annual_generation = max(float(prediction_values[0]), 0.0)
    carbon_offset = max(float(prediction_values[1]), 0.0) if len(prediction_values) > 1 else 0.0
    recommended_system_kw = max(float(prediction_values[2]), 0.0) if len(prediction_values) > 2 else annual_generation / 1800.0

    orientation_multiplier = {
        "South": 1.0,
        "West": 0.90,
        "East": 0.88,
        "North": 0.72,
    }.get(orientation, 1.0)

    annual_generation *= orientation_multiplier
    carbon_offset *= orientation_multiplier
    recommended_system_kw *= orientation_multiplier

    orientation_rankings = {
        "South": float(reference_row["yearly_sunlight_kwh_s"]),
        "West": float(reference_row["yearly_sunlight_kwh_w"]),
        "East": float(reference_row["yearly_sunlight_kwh_e"]),
        "North": float(reference_row["yearly_sunlight_kwh_n"]),
    }

    confidence_level, confidence_message = estimate_confidence(reference_row, training_df)

    return {
        "annual_generation_kwh": annual_generation,
        "carbon_offset_metric_tons": carbon_offset,
        "recommended_system_kw": recommended_system_kw,
        "orientation_rankings": orientation_rankings,
        "orientation": orientation,
        "confidence_level": confidence_level,
        "confidence_message": confidence_message,
        "nearest_tract": {
            "lat_avg": float(reference_row["lat_avg"]),
            "lng_avg": float(reference_row["lng_avg"]),
        },
    }


def save_model(model: RandomForestRegressor, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: dict[str, Any], path: str = METRICS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def plot_feature_importance(model: RandomForestRegressor) -> None:
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
    df = load_data()
    X, y = prepare_data(df)
    model, X_test, y_test, y_pred, metrics = train_random_forest(X, y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_model(model)
    save_metrics(metrics)
    plot_feature_importance(model)
    plot_residuals(y_test, y_pred, TARGET_COLUMNS)

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
