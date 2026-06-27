import sqlite3 as sql
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

DB_PATH = "data/processed/solar.db"
TABLE_NAME = "sunroof_clean"

# Directory where model outputs (plots, artifacts) are stored
# Changed to keep outputs inside the `src/` tree as requested
OUTPUT_DIR = os.path.join("src", "model-output")

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
]
TARGET_COLUMNS = [
    "yearly_sunlight_kwh_total",
    "carbon_offset_metric_tons",
]


def load_data(db_path: str = DB_PATH, table_name: str = TABLE_NAME) -> pd.DataFrame:
    # Load cleaned solar data from the processed SQLite database.
    with sql.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    return df


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Select features and target values, dropping any rows with missing values.
    df = df.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).reset_index(drop=True)
    X = df[FEATURE_COLUMNS].astype(float).to_numpy()
    y = df[TARGET_COLUMNS].astype(float).to_numpy()
    return X, y


def train_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # Train a simple linear regression model and evaluate it on a holdout set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
    }

    return model, X_test, y_test, y_test_pred, metrics


def plot_test_predictions(y_test: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> None:
    """Create scatter plots comparing actual and predicted target values on the test set."""
    for idx, target_name in enumerate(target_names):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test[:, idx], y_pred[:, idx], alpha=0.4, edgecolors="k", linewidths=0.5)
        min_val = min(y_test[:, idx].min(), y_pred[:, idx].min())
        max_val = max(y_test[:, idx].max(), y_pred[:, idx].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        plt.xlabel(f"Actual {target_name}")
        plt.ylabel(f"Predicted {target_name}")
        plt.title(f"Linear Regression: Actual vs Predicted ({target_name})")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"linear_regression_predictions_{target_name}.png")
        plt.savefig(out_path)
        plt.close()


def main() -> None:
    df = load_data()
    X, y = prepare_data(df)
    model, X_test, y_test, y_pred, metrics = train_linear_regression(X, y)
    # Ensure output directory exists for plots and other artifacts
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Linear Regression model trained on data/processed/solar.db")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMNS}")
    print("\nModel evaluation metrics:")
    print(f"  Train RMSE: {metrics['train_rmse']:.2f}")
    print(f"  Test RMSE:  {metrics['test_rmse']:.2f}")
    print(f"  Train R2:   {metrics['train_r2']:.4f}")
    print(f"  Test R2:    {metrics['test_r2']:.4f}")
    print(f"\nSaved plots: {OUTPUT_DIR}/linear_regression_predictions_<target_name>.png")

    plot_test_predictions(y_test, y_pred, TARGET_COLUMNS)


if __name__ == "__main__":
    main()