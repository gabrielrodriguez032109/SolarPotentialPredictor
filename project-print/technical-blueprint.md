# Tract-Level Rooftop Solar Potential Estimation

## Technical Research Report and Reproducibility Guide

## Abstract

This repository presents a reproducible applied machine-learning workflow for
studying tract-level rooftop solar potential with a cleaned Project
Sunroof-style census-tract dataset. The central question is whether annual solar
generation potential can be estimated from geographic location, qualified-roof
measures, coverage measures, and directional sunlight variables. The workflow
starts with a 31-column raw CSV, applies a documented transformation, and
produces a 13-column analysis table with 48,664 complete tract records. A
multi-output Random Forest estimates annual sunlight total, carbon-offset
potential, and total solar capacity; a Linear Regression model supplies a simpler
baseline for the overlapping generation and carbon targets.

The contribution is the transparent, inspectable modeling workflow rather than a
production prediction service. The Streamlit application is a demonstration and
communication interface: it retrieves a nearby tract's stored source values and,
when requested, performs a separate assumption-based homeowner calculation. It
does not use the Random Forest for user-facing results. The study is therefore
best understood as an exploratory tract-level predictive-modeling exercise, with
explicit limitations around target-feature relationships, spatial validation, data
provenance, and property-level interpretation.

## 1. Introduction

Rooftop solar potential is geographically heterogeneous: it varies with the
amount and orientation of available roof area, local sunlight, and the scale of
qualified rooftops within an area. This project uses tract-level aggregates to
study those relationships as a tabular regression problem. It combines a
rebuildable data-preparation pipeline, documented feature engineering,
deterministic model configuration, and diagnostic evaluation artifacts in one
repository.

The unit of analysis is a census tract, not an individual building. Accordingly,
the study does not claim to estimate a surveyed property's production or provide
an engineering design. Its scientific value lies in making the data contract,
modeling assumptions, and evaluation boundaries visible enough for another
analyst to reproduce and extend the experiment.

## 2. Problem Statement and Research Questions

The primary research question is:

> To what extent can tract-level annual rooftop solar generation potential be
> estimated from geospatial location, qualified-roof and coverage measures, and
> directional sunlight features?

The repository also supports two narrower questions:

1. Which available tract-level roof and sunlight variables are most informative
   within the Random Forest workflow?
2. How does a nonlinear Random Forest compare with a Linear Regression baseline
   for the generation and carbon targets that both workflows model?

These questions are intentionally limited to the supplied tract-level dataset.
They do not imply a claim about causal effects, future deployment performance, or
unseen property-level predictions.

## 3. Data and Feature Representation

### 3.1 Source data and unit of analysis

The repository contains `data/raw/sunroof_solar_potential_by_censustract.csv`, a
Project Sunroof-style export with 48,722 rows and 31 fields. The data represent
aggregate census-tract solar quantities. The repository does not contain source
download metadata, a dataset version, a license record, or additional provenance
documentation; the analysis should therefore identify the file as the supplied
Project Sunroof-style export rather than make stronger source claims.

### 3.2 Analysis table

The ETL pipeline retains the following 13 fields. They form the contract between
data preparation, model evaluation, and the demonstration interface.

| Feature group | Fields | Role in the study |
| --- | --- | --- |
| Geographic context | `lat_avg`, `lng_avg` | Tract-center coordinates and model inputs. |
| Roof opportunity | `count_qualified`, `percent_covered`, `percent_qualified` | Aggregate indicators of qualified roof availability and coverage. |
| Directional sunlight | `yearly_sunlight_kwh_n`, `yearly_sunlight_kwh_e`, `yearly_sunlight_kwh_s`, `yearly_sunlight_kwh_w` | Direction-specific tract-level sunlight measures. |
| Yield context | `yearly_sunlight_kwh_kw_threshold_avg` | Annual sunlight yield per kW threshold average. |
| Modeling outcomes | `yearly_sunlight_kwh_total`, `carbon_offset_metric_tons`, `kw_total` | Annual total, carbon-offset potential, and aggregate capacity targets. |

The transformed table has 48,664 rows and no missing values in these selected
columns. It is an area-level analytical dataset: its values should not be
interpreted as measurements for one roof or one household.

### 3.3 Engineered predictors

For Random Forest evaluation, the code derives two additional predictors from the
four directional sunlight fields:

```text
sunlight_total_directional = north + east + south + west
south_to_north_ratio       = south / north
```

North values of zero are converted to missing values before calculating the
ratio, and incomplete feature/target rows are excluded before fitting. Together
with the ten retained location, roof, directional, and yield inputs, these
derived variables create a 12-feature Random Forest design matrix.

## 4. Reproducible Data Preparation

`src/etl/pipeline.py` provides the repository's extract-transform-load sequence:

```text
raw census-tract CSV
  -> pandas read
  -> remove raw rows with any missing value
  -> retain the 13-field analytical schema
  -> data/processed/sunroof_clean.csv
  -> SQLite table: data/processed/solar.db / sunroof_clean
```

The raw export contains 58 rows with at least one missing field. Because the
current transformation calls `dropna()` before selecting the 13 retained fields,
all 58 are removed, even when a missing value may be outside the eventual
analytical schema. This behavior is reproducible and documented, but it is a
data-cleaning choice rather than an assertion that every omitted variable is
scientifically necessary.

The SQLite table is regenerated with `if_exists="replace"`; the CSV and SQLite
database are both current derived artifacts. Model and demonstration code read the
same SQLite table, which reduces the risk of comparing results from separate
preprocessing paths.

## 5. Methodology and Experimental Design

The study treats each complete tract record as one observation in a supervised,
multi-output regression task. The Random Forest implementation uses a seeded
80/20 train-test split (`random_state=42`) and fits 200 trees. Its three targets
are:

```text
yearly_sunlight_kwh_total
carbon_offset_metric_tons
kw_total
```

The fixed seed makes the split and forest configuration reproducible. A separate
five-fold shuffled cross-validation routine (`KFold`, `shuffle=True`,
`random_state=42`) measures R² for annual generation, the first target only.
This is a conventional tabular evaluation design in the codebase; it is not a
spatial holdout design.

The study also includes a Linear Regression baseline in
`src/models/linear_regression.py`. It uses the ten unengineered location, roof,
directional sunlight, and yield fields, the same seeded 80/20 split, and two
outcomes: `yearly_sunlight_kwh_total` and `carbon_offset_metric_tons`. It is an
interpretive baseline, not a component of the Streamlit interface.

## 6. Model Design

### 6.1 Random Forest

`RandomForestRegressor` is used as a nonlinear, multi-output model. The workflow
loads the processed SQLite table, engineers the two directional features, removes
incomplete observations for the selected feature/target matrix, fits the forest,
and can save a serialized model, a JSON metrics file, feature-importance chart,
residual histograms, and actual-versus-predicted plots. These outputs are created
when the script is run; they are not required by the application.

The model's feature-importance plot is an exploratory diagnostic. It can help
rank the contribution of features within this fitted forest, but it should not be
read as a causal attribution or a universal ranking beyond this data and model
specification.

### 6.2 Linear Regression baseline

The baseline supplies a simpler linear relationship for the two shared outcomes.
It reports training and held-out RMSE and R², and writes actual-versus-predicted
plots. It does not serialize a model or alter application results. Its role is to
make the nonlinear Random Forest workflow easier to contextualize, not to claim a
comprehensive model comparison.

## 7. Evaluation

The Random Forest workflow reports held-out RMSE, MAE, MAPE, and R² for its
multi-output target array, together with five-fold R² for annual generation.
Diagnostic plots include actual-versus-predicted values, residual distributions,
and impurity-based feature importance. The Linear Regression workflow reports its
own train/test RMSE and R² and produces two prediction plots.

Two interpretation rules are important:

1. The Random Forest's aggregate error metrics combine outcomes whose physical
   scales and units differ. They are implementation diagnostics, not a single
   unit-specific estimate of solar-generation error.
2. The predictors include directional sunlight variables and their sum while one
   target is annual sunlight total. These are substantively related quantities.
   Strong random-split fit would demonstrate recovery of structure already present
   in the tract aggregates, not proof of an independent, property-level forecast.

The test suite contains 16 synthetic unit/regression tests. It exercises input
validation, ZIP-response parsing, feature engineering, nearest-tract output,
homeowner calculations, ETL column selection, and a compatibility wrapper. It
does not currently run a full raw-data-to-database integration test, a real-data
model evaluation assertion, or a Streamlit smoke test.

## 8. Results and Interpretation

The repository is designed to produce reproducible evaluation outputs, but it
does not retain a versioned Random Forest metrics artifact that should be quoted as
a permanent benchmark. For that reason, this report does not present unverified
numeric performance claims. Re-running the documented workflow on the supplied
processed data produces the held-out metrics and diagnostic figures for a specific
environment and data artifact.

The appropriate interpretation is therefore methodological. The project
establishes a complete tract-level experiment with real geospatial, roof-related,
and sunlight features, a nonlinear model, and a linear reference model. It can
show whether the selected variables recover variation in the provided aggregate
outcomes under the stated split. It cannot, without stronger validation, establish
accuracy for a new geography, a future data release, or a single rooftop.

## 9. Demonstration and Communication Interface

`src/app/app.py` is intentionally secondary to the modeling study. It is a
Streamlit demonstration that helps users inspect nearby tract context and
communicate the difference between area-scale and home-scale quantities.

For the community view, the application selects the nearest available tract by a
squared latitude/longitude comparison and displays the stored source values for
annual generation, carbon offset, and capacity. It does not call
`model.predict()`. A displayed haversine distance describes the gap from the
requested point to the selected tract center; it is a geographic-match indicator,
not model confidence.

For the homeowner view, the application applies a transparent planning formula to
the selected tract's local yield and carbon rate. Roof area is converted to a
capacity ceiling of `0.004 kW/sq ft`, bounded from 1 to 15 kW. Orientation and
broad shading multipliers then adjust yield; optional monthly electricity use can
size toward annual demand without exceeding that roof-area ceiling. These values
are demonstration assumptions, not research targets or a site-specific system
design.

## 10. Limitations and Threats to Validity

- **Aggregate target relationships.** Directional sunlight fields are closely
  related to the annual-total target, limiting the strength of any claim that the
  model predicts novel information.
- **Validation design.** The implementation uses random train/test and shuffled
  folds rather than spatial, temporal, or external validation. Nearby tracts may
  appear in both training and test partitions.
- **Geographic granularity.** Coordinates represent tract centers and no processed
  tract identifier is retained. Tract aggregates do not resolve roof geometry,
  tilt, azimuth, condition, panel technology, or site-specific shading.
- **Data provenance.** The supplied file lacks source-version, download, license,
  and collection-metadata records in this repository.
- **ETL contract.** All-source-column null removal happens before field selection;
  expected source schema, type, range, duplicate, and spatial-bound checks are not
  currently enforced.
- **Interface boundaries.** The nearest-tract selection uses squared degrees while
  displayed distance uses haversine distance. The interface's home estimate is a
  separate rule-based calculation and must not be reported as Random Forest
  inference.

## 11. Conclusion

This project is best framed as a reproducible applied ML study of tract-level
rooftop solar potential. It makes an explicit research question operational with a
cleaned geospatial dataset, documented feature construction, a seeded Random
Forest experiment, a linear baseline, and diagnostic evaluation routines. The
project's strongest contribution is its transparent workflow and clear boundary
between model evaluation and demonstration. Future work should add source
provenance, schema validation, spatial holdouts, per-target reporting, and
property-level ground truth before making stronger predictive claims.

## 12. Project Summary

| Component | Research role |
| --- | --- |
| `data/raw/` | Supplied tract-level Project Sunroof-style CSV. |
| `src/etl/pipeline.py` | Rebuildable transformation into the 13-field analytical table. |
| `data/processed/` | Clean CSV and SQLite source shared by evaluation and demo workflows. |
| `src/models/random_forest.py` | Primary Random Forest experiment, diagnostics, and reusable tract helpers. |
| `src/models/linear_regression.py` | Linear baseline for two overlapping outcomes. |
| `tests/test_random_forest_output.py` | Synthetic regression coverage for core behavior. |
| `src/app/app.py` | User-facing demonstration and interpretation interface, not the central model contribution. |

To reproduce the repository workflow from the project root:

```powershell
python -m pip install -r requirements.txt
python src/etl/pipeline.py
python src/models/random_forest.py
python src/models/linear_regression.py
python -m pytest -q
python -m streamlit run src/app/app.py
```

The model scripts create local artifacts under `src/model-output/` and, for the
Random Forest, `src/models/random_forest.pkl`. The Streamlit interface can run
without the serialized model because it reads the processed tract table directly.

## 13. Suggested Research Title Options

1. **Estimating Tract-Level Rooftop Solar Potential with Geospatial and Roof-Related Features**
2. **A Reproducible Machine Learning Workflow for Census-Tract Solar Potential Estimation**
3. **Modeling Annual Rooftop Solar Generation from Tract-Level Sunlight and Roof Indicators**
4. **Exploratory Prediction of Census-Tract Solar Potential Using Random Forest Regression**
