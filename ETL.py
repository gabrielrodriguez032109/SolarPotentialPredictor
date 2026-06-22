# ETL.py - a simple extract, transform, load pipeline for solar potential data.
# Install pandas with: pip install pandas

import pandas as pd
from pathlib import Path


def extract(csv_filename: str = "project-sunroof-city-09082017.csv") -> pd.DataFrame:
    """Read the raw solar dataset from a CSV file."""
    base_dir = Path(__file__).parent
    csv_path = base_dir / csv_filename
    print(f"Extracting data from {csv_path}")
    return pd.read_csv(csv_path)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the useful columns, clean values, and add a new score."""
    print("Transforming data")

    keep_columns = [
        "region_name",
        "state_name",
        "lat_avg",
        "lng_avg",
        "yearly_sunlight_kwh_total",
        "percent_qualified",
        "carbon_offset_metric_tons",
        "existing_installs_count",
    ]

    df = df[keep_columns].copy()

    df.columns = [
        "region",
        "state",
        "latitude",
        "longitude",
        "sunlight_kwh_total",
        "qualified_percent",
        "carbon_offset_tons",
        "existing_installs_count",
    ]

    numeric_columns = [
        "latitude",
        "longitude",
        "sunlight_kwh_total",
        "qualified_percent",
        "carbon_offset_tons",
        "existing_installs_count",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["region", "state", "sunlight_kwh_total"])

    df["solar_score"] = df["sunlight_kwh_total"] * df["qualified_percent"] / 100

    df = df.sort_values(by="solar_score", ascending=False).reset_index(drop=True)

    return df


def load(df: pd.DataFrame, output_filename: str = "solar_data_cleaned.csv") -> Path:
    """Save the cleaned data to a new CSV file."""
    print(f"Loading data to {output_filename}")
    output_path = Path(__file__).parent / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_etl() -> Path:
    raw_data = extract()
    clean_data = transform(raw_data)
    output_path = load(clean_data)

    print("ETL complete")
    print(f"Rows extracted: {len(raw_data)}")
    print(f"Rows loaded: {len(clean_data)}")
    print(f"Cleaned file saved at: {output_path}")
    print("Sample cleaned rows:")
    print(clean_data.head(10).to_string(index=False))

    return output_path


if __name__ == "__main__":
    run_etl()