Solar Potential Predictor – Full Project Blueprint
================================================

Purpose of this document
------------------------
This file is a complete walkthrough of the Solar Potential Predictor project. It explains every major process in the project from start to finish, including where data comes from, how it is transformed, how the model uses it, how the app displays results, and how the files in the repository work together.

This document is meant to be read by someone who wants to understand the project in detail, not just at a high level.

1. What the project is
-----------------------
The Solar Potential Predictor is a Python-based machine learning project that estimates solar potential for a given location using solar-related census tract data.

It takes a location (latitude and longitude, or an optional ZIP code), uses historical solar data from the dataset, and produces a prediction for:
- annual solar generation in kilowatt-hours
- carbon offset in metric tons
- recommended system size in kilowatts

The project is built as an end-to-end example of a data science workflow:
1. Raw data is read from disk.
2. The data is cleaned and prepared.
3. Useful features are created.
4. A machine learning model is trained.
5. A user-facing app uses the trained model to make predictions.

2. The main problem the project solves
--------------------------------------
The project solves a practical forecasting problem: given a location, estimate how much solar energy that area could potentially generate if a system were installed.

Instead of manually analyzing a dataset by hand, the project uses historical solar characteristics to learn patterns and then predicts future estimates based on similar known locations.

This is useful because:
- solar installers can estimate system size quickly
- users can see a rough energy potential before installation
- the project demonstrates how a tabular machine learning workflow can be turned into an application

3. The repository structure
---------------------------
The repository is organized into a small set of folders and files:

- data/
  This folder contains the raw input data and the processed outputs.
  - data/raw/sunroof_solar_potential_by_censustract.csv
    This is the original dataset used as input.
  - data/processed/sunroof_clean.csv
    This is the cleaned version of the dataset written out by the ETL pipeline.
  - data/processed/solar.db
    This is the SQLite database version of the cleaned data used by the application and model.

- src/
  This folder contains the actual application logic.
  - src/etl/
    Contains the ETL pipeline that extracts, cleans, and loads data.
  - src/models/
    Contains the machine learning workflow including training and prediction.
  - src/app/
    Contains the Streamlit web app.

- tests/
  This folder contains regression tests that validate key project behavior.

- README.md
  A short high-level overview of the project.

- requirements.txt
  Lists all Python dependencies needed for the project.

- PLAN.txt
  A general documentation summary of the project.

- blueprint.txt
  This file. It provides a more detailed technical walkthrough.

4. How the data flows through the project
-----------------------------------------
The project uses a straightforward end-to-end flow.

Step 1: raw data is loaded
The starting point is the raw CSV file:
- data/raw/sunroof_solar_potential_by_censustract.csv

This file contains one row per geographic tract and many columns describing solar characteristics.

Step 2: data is cleaned
The ETL pipeline reads that raw file and prepares a cleaned version.

The cleaning process does the following:
- reads the CSV into a pandas DataFrame
- removes incomplete rows with missing values
- selects only the columns needed by the modeling workflow
- resets the row index so the table is clean and consistent

Step 3: cleaned data is saved
The cleaned dataframe is stored in two places:
- a CSV file: data/processed/sunroof_clean.csv
- a SQLite database file: data/processed/solar.db

The database is especially important because both the model and the Streamlit app read from it.

Step 4: the model uses the cleaned data
The model module reads the SQLite table and converts it into a feature matrix.

The model uses columns such as:
- latitude and longitude
- count of qualified roof space
- percent covered
- percent qualified
- sunlight values by direction
- average sunlight threshold values

These values help the model learn relationships between the input features and expected solar outcomes.

Step 5: the app makes predictions
The Streamlit app takes user input, calls the prediction helper, and produces a solar estimate.

5. How every major file works
-----------------------------
5.1 data/raw/sunroof_solar_potential_by_censustract.csv
This is the source dataset.

It is the original unprocessed file that the ETL step reads.

It is important because everything else depends on it. If the raw data changes, the cleaned data and predictions will also change.

5.2 src/etl/pipeline.py
This file is the ETL entry point.

Its job is to take the raw dataset and turn it into a cleaned, usable form.

It contains three main stages:
- extract(): reads the raw CSV file from disk
- transform(): cleans and selects the needed columns
- load(): writes the cleaned dataframe out as CSV and SQLite

How it works in detail:
- extract() loads the CSV using pandas.read_csv
- transform() converts the data into a DataFrame, removes rows with missing values, and keeps only the relevant columns
- load() writes the data to data/processed/sunroof_clean.csv and creates/overwrites the SQLite table in data/processed/solar.db

Why this file exists:
The rest of the project assumes that cleaned data exists in a predictable format. This file creates that format.

5.3 src/models/random_forest.py
This is the heart of the machine learning workflow.

It contains everything needed to:
- load the processed data
- validate user input
- engineer new features
- train the model
- save the model to disk
- evaluate the model
- make predictions from the app

Key functions inside this file:

- load_data()
  Reads the processed SQLite table into a pandas DataFrame.
  This is the bridge between the database and the model logic.

- validate_inputs()
  Checks the values provided by the user before the model is used.
  It rejects invalid latitude/longitude values and bad ZIP code values.

- engineer_features()
  Creates new derived columns from existing ones.
  Examples:
  - sunlight_total_directional
  - south_to_north_ratio
  These features are created because they can capture broad directional solar patterns that the raw directional columns alone might not express clearly.

- prepare_data()
  Converts the processed dataframe into the array format expected by scikit-learn.
  This step is necessary because machine learning models need numeric arrays rather than raw table rows.

- load_or_train_model()
  Checks whether a saved model already exists. If it does, it loads that file. If not, it trains a new model.

- train_random_forest()
  Creates a Random Forest Regressor and fits it to the training data. It also computes evaluation metrics.

- plot_test_predictions()
  Saves prediction plots comparing actual and predicted values.

- plot_residuals()
  Saves residual plots to inspect error distribution.

- cross_validate_model()
  Uses cross-validation to estimate model stability.

- format_prediction_summary()
  Turns the raw prediction result into a well-formatted summary string for the app or logs.

- estimate_confidence()
  Computes a simple confidence estimate based on how close the selected tract is to the training data.

- predict_with_model()
  This is the most important inference function for the app.
  It takes the trained model, the training data, and a user location and returns a prediction payload.

How predict_with_model works:
1. It engineers features on the dataset.
2. It selects available feature columns and target columns from the dataframe.
3. It drops rows with missing values for the relevant columns.
4. It finds the nearest tract to the input latitude and longitude.
5. It builds a single feature row from that tract.
6. It passes that row to the trained model.
7. It returns metrics like annual generation, carbon offset, recommended system size, orientation rankings, confidence, and nearest tract details.

5.4 src/app/app.py
This is the user-facing Streamlit application.

Its job is to translate user input into a prediction result and display it clearly.

Its flow is:
1. It defines a form with input fields for latitude, longitude, ZIP code, and orientation.
2. When the user submits the form, it validates the input.
3. It loads the processed data from SQLite.
4. It loads or trains the model.
5. It calls predict_with_model.
6. It displays the prediction with metrics in the UI.

The app does not train the model on each request. It loads a pre-trained model from disk if one exists.

This is important because it makes the app faster and more practical for regular use.

5.5 tests/test_random_forest_output.py
This file contains regression tests that ensure the project continues to behave correctly.

The tests cover:
- prediction summary formatting
- input validation
- feature engineering
- prediction payload creation
- ETL transform behavior

These tests protect the project from breaking when the code is changed later.

6. How the model is trained
---------------------------
Training begins from the cleaned data stored in the processed SQLite database.

The training process is as follows:
1. Load the dataframe from the database.
2. Build a feature matrix from columns that describe location and solar characteristics.
3. Build a target matrix from solar outcome columns.
4. Split the data into training and testing subsets.
5. Fit a Random Forest Regressor.
6. Generate metrics to measure performance.

The model is trained to map input solar features to output solar outcomes.

Why this is useful:
Instead of manually calculating solar potential from raw metrics, the model learns patterns from the historical data and uses those patterns to estimate new values.

7. How predictions are made
---------------------------
When the app receives a user location, the prediction path works like this:

1. The app validates the input.
2. The app loads the processed dataset.
3. The model module finds the nearest tract in the dataset based on coordinates.
4. That tract becomes the reference row for the prediction.
5. The reference row is converted into a feature vector.
6. The trained model predicts the solar outcome values.
7. The output is adjusted slightly based on orientation.
8. The result is returned as a structured dictionary with several values.

This allows the app to produce a meaningful and user-friendly prediction instead of just returning raw model output.

8. What data is used by the model
--------------------------------
The model uses both base data and engineered data.

Base columns:
- lat_avg
- lng_avg
- count_qualified
- percent_covered
- percent_qualified
- yearly_sunlight_kwh_n
- yearly_sunlight_kwh_e
- yearly_sunlight_kwh_s
- yearly_sunlight_kwh_w
- yearly_sunlight_kwh_kw_threshold_avg

Engineered columns:
- sunlight_total_directional
- south_to_north_ratio

The targets are:
- yearly_sunlight_kwh_total
- carbon_offset_metric_tons
- kw_total when available

The model uses these values because they represent the physical and directional characteristics of the solar potential dataset.

9. Why the project uses SQLite
------------------------------
SQLite is used because it is simple and convenient for a local project.

It provides a lightweight way to store the cleaned data in a structured table that can be read by Python without needing a larger database system.

The reason the app and model read from SQLite rather than directly from the CSV is that SQLite gives the project a consistent runtime storage format.

10. Why the app uses Streamlit
------------------------------
Streamlit is a lightweight framework for turning Python scripts into interactive web apps.

It is ideal for this project because the app only needs a simple form and a few output metrics.

The Streamlit app makes the machine learning workflow accessible to non-technical users without requiring a full web framework or frontend setup.

11. How to run the project manually
----------------------------------
The project can be executed in the following order:

Step 1: install dependencies
Open a terminal in the project root and run:
- python -m pip install -r requirements.txt

Step 2: run the ETL pipeline
- python src/etl/pipeline.py

Step 3: train or refresh the model
- python src/models/random_forest.py

Step 4: launch the app
- python -m streamlit run src/app/app.py

12. Important implementation notes
---------------------------------
The project is intentionally simple and educational rather than a fully production-grade system.

This means:
- it focuses on correctness and clarity
- it demonstrates an end-to-end workflow
- it uses a straightforward model and app architecture
- it should be easy for a new developer to read and extend

A few practical notes:
- The ETL step creates the database and the cleaned CSV artifacts.
- The model step expects those artifacts to exist.
- The app depends on the model and the processed dataset being available.
- If the raw dataset or column names change, the ETL and model code may need to be updated together.

13. Where errors are most likely to occur
----------------------------------------
The most common failure points are:
- missing columns in the processed dataframe
- a missing processed database file
- an outdated model file that does not match the current data schema
- invalid user input values

This is why the project includes validation and column-aware logic.

14. Suggested future improvements
--------------------------------
Possible improvements include:
- add a more advanced geospatial prediction method
- improve feature engineering further
- compare multiple models such as gradient boosting or XGBoost
- add more robust error handling and logging
- increase documentation for each module
- add richer visualizations and explanation tools
- add support for real ZIP-code lookup instead of only coordinate-based nearest tract matching

15. Final summary
-----------------
The Solar Potential Predictor is a compact end-to-end machine learning project that transforms a census-tract solar dataset into a useful prediction application.

Its major flow is:
1. Read raw solar data.
2. Clean and standardize it.
3. Save it to processed files.
4. Train a Random Forest model.
5. Use the model to estimate solar metrics for a location.
6. Display the result through a Streamlit app.

Every part of the project connects to this flow, and this blueprint explains the purpose of each part and how they work together.
