"""ETL helpers for preparing the Project Sunroof dataset for the app and model.

The pipeline reads the raw census-tract CSV, keeps only the columns needed for solar
potential modeling, removes rows with incomplete values, and writes a cleaned SQLite
copy that the rest of the project can load reliably.
"""

import pandas as pd
from sqlalchemy import create_engine


def extract() -> pd.DataFrame:
    """Load the raw Project Sunroof CSV from disk.

    Returns:
        A pandas DataFrame containing the unprocessed census-tract data.
    """
    df = pd.read_csv("data/raw/sunroof_solar_potential_by_censustract.csv")
    return df


def transform(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and narrow the raw dataset to the columns used by the model.

    This step removes incomplete rows and keeps the solar-related fields that are
    meaningful for prediction. The selected columns form the contract between the ETL
    step and the model/inference code.
    """
    df = pd.DataFrame(data)
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Keep only the columns that the model and app need. This makes the processed data
    # easier to reason about and prevents downstream code from depending on optional
    # columns that may not exist in every data export.
    selected_columns = [
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
        "yearly_sunlight_kwh_total",
        "carbon_offset_metric_tons",
    ]

    available_columns = [column for column in selected_columns if column in df.columns]
    df = df[available_columns].reset_index(drop=True)

    return df

def load(df: pd.DataFrame) -> None:
    """Persist the cleaned dataframe as both CSV and SQLite artifacts.

    The CSV file is useful for inspection and manual review, while the SQLite table is
    the runtime format used by the model and the Streamlit application.
    """
    # Save cleaned CSV for easy inspection and manual debugging.
    df.to_csv(
        "data/processed/sunroof_clean.csv",
        index=False
    )

    # Create SQLite database file (auto-created)
    engine = create_engine("sqlite:///data/processed/solar.db")

    # Load dataframe into SQLite table (Save table)
    df.to_sql(
        "sunroof_clean",
        engine,
        if_exists="replace",
        index=False
    )

    print("Data loaded into SQLite database: data/processed/solar.db")

# Run the ETL pipeline when this file is executed directly.
df = extract()
df = transform(df)
load(df)
