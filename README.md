# Solar Potential Predictor

Solar Potential Predictor is an end-to-end Python project that estimates solar potential for a user-specified location using a cleaned Project Sunroof-style census-tract dataset. The project combines data processing, feature engineering, machine learning, and a Streamlit-based web application into a single workflow.

## Overview

The project takes either latitude/longitude or a US ZIP code and provides an approximate planning estimate for:

- annual solar generation in kilowatt-hours
- carbon offset in metric tons
- recommended system size in kilowatts
- orientation-based comparison for North, South, East, and West exposure

The app now supports two interpretation modes:

- Community / tract-level estimate: displays the selected tract's source values
- Homeowner estimate: calculates a roof-area-based planning estimate using local solar yield, optional broad shading, and optional electricity use

The workflow includes:

1. Loading the raw solar dataset
2. Cleaning and selecting the most relevant columns
3. Saving the processed data to CSV and SQLite
4. Optionally training a Random Forest regressor for evaluation
5. Serving nearest-tract source values and homeowner planning calculations through Streamlit

## Project structure

```text
SolarPotentialPredictor/
├── data/
│   ├── raw/
│   │   └── sunroof_solar_potential_by_censustract.csv
│   └── processed/
│       ├── sunroof_clean.csv
│       └── solar.db
├── src/
│   ├── etl/
│   │   └── pipeline.py
│   ├── models/
│   │   ├── random_forest.py
│   │   └── linear_regression.py
│   └── app/
│       └── app.py
├── tests/
│   └── test_random_forest_output.py
├── PLAN.txt
├── project-print/
│   ├── project-overview.txt
│   ├── technical-blueprint.md
│   └── debugging-notes.txt
├── requirements.txt
└── README.md
```

## Key components

- ETL pipeline: prepares the raw dataset into a clean, model-ready format
- Model module: trains and evaluates the Random Forest regressor and runs predictions
- Streamlit app: provides a small interactive interface for end users
- Tests: validate core prediction, validation, and ETL behavior

## Technologies used

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- SQLAlchemy
- joblib
- Streamlit
- pytest

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Running the project

### Run the ETL pipeline

This creates the cleaned CSV and SQLite database outputs.

```powershell
python src/etl/pipeline.py
```

### Train or refresh the model

This trains the Random Forest model and saves the trained artifact for later use.

```powershell
python src/models/random_forest.py
```

### Launch the Streamlit app

```powershell
python -m streamlit run src/app/app.py
```

After launching, open the local URL provided by Streamlit in your browser.

## Testing

Run the test suite with:

```powershell
python -m pytest -q
```

## Notes on the current implementation

- The processed database is the main runtime data source used by both the model and the app.
- The prediction flow uses the nearest available census tract to the requested coordinates.
- The app accepts coordinates or a US ZIP code. ZIP lookup uses an approximate ZIP centroid; coordinates are more specific.
- Community outputs come directly from the selected tract, rather than re-predicting source fields.
- Homeowner outputs use documented roof-area and orientation assumptions.
- Optional monthly electricity use right-sizes the homeowner recommendation; broad shading adjusts its expected production.
- Results are best treated as approximate planning estimates rather than site-verified engineering numbers.

## Documentation

Additional documentation is available in:

- [PLAN.txt](PLAN.txt)
- [project-print/project-overview.txt](project-print/project-overview.txt)
- [project-print/technical-blueprint.md](project-print/technical-blueprint.md)
- [project-print/debugging-notes.txt](project-print/debugging-notes.txt)

These files explain the architecture, data flow, model workflow, and project structure in more detail.
