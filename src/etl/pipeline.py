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
    # Keep the raw file untouched. Every downstream artifact is regenerated from it,
    # which lets an analyst return to the original export if cleaning rules change.
    df = pd.read_csv("data/raw/sunroof_solar_potential_by_censustract.csv")
    return df


def transform(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and narrow the raw dataset to the columns used by the model.

    This step removes incomplete rows and keeps the solar-related fields that are
    meaningful for prediction. The selected columns form the contract between the ETL
    step and the model/inference code.
    """
    # Constructing a new DataFrame gives this function its own working copy, so callers
    # can safely compare the raw and transformed versions after this function returns.
    df = pd.DataFrame(data)
    # Drop incomplete source records before selecting columns so every persisted row is
    # internally complete, including the source fields displayed by community mode.
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Keep only the columns that the model and app need. This makes the processed data
    # easier to reason about and prevents downstream code from depending on optional
    # columns that may not exist in every data export.
    # This list is the processed-data contract. Changes here must be reflected in the
    # model feature/target lists and in the app's direct-source prediction fields.
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
        "kw_total",
    ]

    # Keep the transform tolerant of a slightly older raw export. Note that this does
    # not guarantee that later app/model code can run: those layers require specific
    # fields and will raise an error if a required field is absent.
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

    # SQLite is the runtime source of truth: the Streamlit app and evaluation workflows
    # read the same `sunroof_clean` table rather than independently parsing the CSV.
    # The database file is created automatically when it does not already exist.
    engine = create_engine("sqlite:///data/processed/solar.db")

    # Replace, rather than append to, the table so it always matches this ETL run.
    df.to_sql(
        "sunroof_clean",
        engine,
        if_exists="replace",
        index=False
    )

    print("Data loaded into SQLite database: data/processed/solar.db")

if __name__ == "__main__":
    # Run the ETL pipeline only when this file is executed directly. Importing the
    # transform helper in tests must not overwrite the processed data artifacts.
    # Keep the three ETL stages explicit so a learner can run or inspect each one in
    # an interactive session without relying on hidden orchestration.
    df = extract()
    df = transform(df)
    load(df)
