import pandas as pd
from sqlalchemy import create_engine

# EXTRACT
def extract():
    df = pd.read_csv("data/raw/sunroof_solar_potential_by_censustract.csv")
    return df

# TRANSFORM
def transform(data):
    df = pd.DataFrame(data)
    df = df.dropna()
    df = df.reset_index(drop=True)

    df = df[
        [
            # Region metadata - not included 

            # Location
            "lat_avg",
            "lng_avg"

            # Solar potential characteristics
            "count_qualified",
            "percent_covered",
            "percent_qualified",

            # System characteristics - not included

            # Orientation-specific sunlight
            "yearly_sunlight_kwh_n",
            "yearly_sunlight_kwh_e",
            "yearly_sunlight_kwh_s",
            "yearly_sunlight_kwh_w",

            # Average sunlight intensity
            "yearly_sunlight_kwh_kw_threshold_avg",

            # Targets
            "yearly_sunlight_kwh_total",
            "carbon_offset_metric_tons"
        ]
    ]

    return df

# LOAD
def load(df):
    # Save cleaned CSV
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

# RUN PIPELINE
df = extract()
df = transform(df)
load(df)
