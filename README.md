# Solar Potential Predictor

This project builds an end-to-end machine learning application for estimating solar potential from Google Project Sunroof data.

## What the project does

- Cleans and processes the Project Sunroof dataset
- Stores the cleaned data in SQLite
- Trains a Random Forest regressor
- Saves the trained model with joblib
- Provides a Streamlit app for interactive predictions

## Project structure

- src/etl/ - ETL pipeline
- src/models/ - model training, persistence, and prediction helpers
- src/app/ - Streamlit dashboard
- tests/ - regression tests

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Run the ETL pipeline

```powershell
python src/etl/pipeline.py
```

## Train the model

```powershell
python src/models/random_forest.py
```

## Run the Streamlit app

```powershell
python -m streamlit run src/app/app.py
```

## Testing

```powershell
python -m pytest -q
```