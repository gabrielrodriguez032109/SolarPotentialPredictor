import sqlite3 as sql
import os

# numpy handles numerical arrays and math operations used during evaluation.
import numpy as np

# pandas loads the SQLite table into a DataFrame and prepares tabular data.
import pandas as pd

# matplotlib creates and saves the actual-vs-predicted scatter plots.
import matplotlib.pyplot as plt

# scikit-learn provides the regression model, train/test split, and metrics.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Location of the processed SQLite database created by the ETL pipeline.
DB_PATH = "data/processed/solar.db"

# Name of the cleaned table inside the SQLite database.
TABLE_NAME = "sunroof_clean"

# Directory where model outputs, such as plots, are stored.
OUTPUT_DIR = os.path.join("src", "model-output")

# Input columns used by the model to make predictions.
# These describe location, qualified roof counts, coverage, and sunlight by direction.
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

# Output columns the model is trying to predict.
# This model predicts both total yearly sunlight and carbon offset at the same time.
TARGET_COLUMNS = [
    "yearly_sunlight_kwh_total",
    "carbon_offset_metric_tons",
]


def load_data(db_path: str = DB_PATH, table_name: str = TABLE_NAME) -> pd.DataFrame:
    # Open a connection to the SQLite database. The context manager closes it automatically.
    with sql.connect(db_path) as conn:
        # Read the full cleaned table into a pandas DataFrame.
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    return df


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Remove rows that are missing any feature or target value, because scikit-learn
    # cannot train LinearRegression with NaN values.
    df = df.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).reset_index(drop=True)

    # X is the input matrix: one row per census tract, one column per feature.
    X = df[FEATURE_COLUMNS].astype(float).to_numpy()

    # y is the target matrix: one row per census tract, one column per target.
    y = df[TARGET_COLUMNS].astype(float).to_numpy()

    return X, y


def train_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # Split the data into training and testing sets.
    # The model learns from the training set, then gets evaluated on unseen test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create the linear regression model.
    model = LinearRegression()

    # Fit the model so it learns coefficients that map features to target values.
    model.fit(X_train, y_train)

    # Predict on both train and test sets so we can compare fit vs generalization.
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # RMSE measures average prediction error in the same units as the target.
    # R2 measures how much variance the model explains; closer to 1 is better.
    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
    }

    return model, X_test, y_test, y_test_pred, metrics


def plot_test_predictions(y_test: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> None:
    """Create scatter plots comparing actual and predicted target values on the test set."""
    # Build one plot per target column.
    for idx, target_name in enumerate(target_names):
        # Start a new figure so each target gets its own saved image.
        plt.figure(figsize=(8, 6))

        # Each point compares an actual value from the test set against the model prediction.
        plt.scatter(y_test[:, idx], y_pred[:, idx], alpha=0.4, edgecolors="k", linewidths=0.5)

        # Use the min and max values to draw a perfect-prediction reference line.
        min_val = min(y_test[:, idx].min(), y_pred[:, idx].min())
        max_val = max(y_test[:, idx].max(), y_pred[:, idx].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

        # Label the plot so the saved image is understandable on its own.
        plt.xlabel(f"Actual {target_name}")
        plt.ylabel(f"Predicted {target_name}")
        plt.title(f"Linear Regression: Actual vs Predicted ({target_name})")
        plt.tight_layout()

        # Save the plot to the model output directory, then close it to free memory.
        out_path = os.path.join(OUTPUT_DIR, f"linear_regression_predictions_{target_name}.png")
        plt.savefig(out_path)
        plt.close()


def main() -> None:
    # Load cleaned data from the processed SQLite database.
    df = load_data()

    # Convert the DataFrame into model-ready input and target arrays.
    X, y = prepare_data(df)

    # Train the model and collect predictions plus evaluation metrics.
    model, X_test, y_test, y_pred, metrics = train_linear_regression(X, y)

    # Ensure the output directory exists before saving plots.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Print a short summary of what ran and how the model performed.
    print("Linear Regression model trained on data/processed/solar.db")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMNS}")
    print("\nModel evaluation metrics:")
    print(f"  Train RMSE: {metrics['train_rmse']:.2f}")
    print(f"  Test RMSE:  {metrics['test_rmse']:.2f}")
    print(f"  Train R2:   {metrics['train_r2']:.4f}")
    print(f"  Test R2:    {metrics['test_r2']:.4f}")
    print(f"\nSaved plots: {OUTPUT_DIR}/linear_regression_predictions_<target_name>.png")

    # Save actual-vs-predicted plots for each target column.
    plot_test_predictions(y_test, y_pred, TARGET_COLUMNS)


# Run main() only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()
