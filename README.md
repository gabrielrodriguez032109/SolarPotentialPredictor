# SolarPotentialPredictor

This project contains a simple ETL (Extract, Transform, Load) pipeline for solar potential data.

## What this pipeline does

- **Extract**: reads the raw CSV file into a pandas DataFrame
- **Transform**: keeps useful columns, cleans numeric values, drops bad rows, and calculates a `solar_score`
- **Load**: saves the cleaned data back to a new CSV file

## Run the ETL pipeline

1. Install the dependency:

```bash
pip install pandas
```

2. Run the script:

```bash
python ETL.py
```

3. The cleaned output file will be saved as `solar_data_cleaned.csv` in the same folder.

## File to edit

- `ETL.py` contains the complete pipeline and is written to be easy to follow for Python beginners.