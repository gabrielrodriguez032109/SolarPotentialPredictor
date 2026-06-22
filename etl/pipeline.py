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
            "region_name",
            "lat_avg",
            "lng_avg",
            "yearly_sunlight_kwh_kw_threshold_avg",
            "count_qualified",
            "percent_covered",
            "percent_qualified",
            "state_name",
            "yearly_sunlight_kwh_total"
        ]
    ]

    return df

# LOAD
def load(df):
    engine = create_engine(
        "postgresql://username:password@localhost:5432/mydatabase"
    )

    df.to_sql(
        "sunroof_clean",
        engine,
        if_exists="replace",
        index=False
    )

    print("Data loaded successfully!")

# RUN PIPELINE
df = extract()
df = transform(df)
load(df)
