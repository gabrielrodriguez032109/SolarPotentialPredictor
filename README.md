# Tract-Level Rooftop Solar Potential Estimation

This repository is a reproducible applied machine-learning study of whether
tract-level rooftop solar potential can be estimated from geospatial,
roof-related, and directional sunlight features. It uses a cleaned Project
Sunroof-style census-tract dataset, an explicit ETL pipeline, and offline
regression workflows to examine annual solar-generation potential and related
tract-level outcomes.

The project is not a property-level solar design or a production prediction
service. Its primary contribution is the transparent workflow: raw data,
rebuildable preprocessing, documented features, seeded model evaluation, and
tests. The Streamlit app is a supporting demonstration interface for exploring
nearby tract context and communicating the distinction between aggregate and
home-scale estimates.

## Research focus

**Primary question:** Can tract-level annual rooftop solar generation potential be
estimated from tract location, qualified-roof and coverage measures, and
directional sunlight features?

The workflow also makes it possible to explore which inputs are influential
within a Random Forest and to compare that nonlinear model with a Linear
Regression baseline for their shared outcomes.

## Study design at a glance

```text
raw Project Sunroof-style tract CSV
  -> ETL: remove incomplete rows and retain the analytical schema
  -> processed CSV + SQLite table
  -> Random Forest and Linear Regression evaluation workflows
  -> diagnostic metrics and plots

                                -> Streamlit research demonstration
                                   (nearby source record, not RF inference)
```

The supplied raw CSV has 48,722 rows and 31 columns. The current ETL output has
48,664 complete tract records and 13 selected fields:

- tract-center latitude and longitude;
- qualified-roof count, covered percentage, and qualified percentage;
- north, east, south, and west annual sunlight measures;
- annual sunlight yield per kW threshold average;
- annual sunlight total, carbon-offset potential, and total capacity.

The Random Forest adds directional-sum and south-to-north-ratio features, yielding
12 inputs. It uses a seeded 80/20 split (`random_state=42`), 200 trees, and three
outputs: `yearly_sunlight_kwh_total`, `carbon_offset_metric_tons`, and `kw_total`.
The Linear Regression baseline uses the ten unengineered inputs and the first two
outcomes.

## Repository structure

```text
SolarPotentialPredictor/
├── data/
│   ├── raw/                    # supplied tract-level source CSV
│   └── processed/              # reproducible CSV and SQLite analysis artifacts
├── src/
│   ├── etl/pipeline.py         # extract, transform, and load workflow
│   ├── models/
│   │   ├── random_forest.py    # primary RF experiment and diagnostics
│   │   └── linear_regression.py # baseline experiment
│   └── app/app.py              # supporting Streamlit demonstration
├── tests/test_random_forest_output.py
└── project-print/              # research report and supporting documentation
```

## Reproduce the workflow

Create and activate a virtual environment, then install the documented
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

From the repository root, rebuild the analytical data and run the evaluation
workflows:

```powershell
python src/etl/pipeline.py
python src/models/random_forest.py
python src/models/linear_regression.py
python -m pytest -q
```

The Random Forest script writes a local serialized model, metrics JSON, and
diagnostic plots. The Linear Regression script writes actual-versus-predicted
plots. These artifacts are generated for evaluation; they are not required for
the application.

## Demonstration interface

Launch the optional Streamlit interface with:

```powershell
python -m streamlit run src/app/app.py
```

The community view selects the nearest available census tract and displays that
record's stored aggregate values. It does **not** call the trained Random Forest.
The homeowner view is a separate, transparent planning calculation using user
roof area, local tract yield, orientation, broad shading, and optional electricity
use. Both views are communication aids and should not be interpreted as a
site-specific engineering assessment.

## Interpretation and limitations

- The model evaluates tract aggregates, not individual roofs or households.
- Directional sunlight variables are closely related to the annual-total target;
  evaluation should be interpreted as held-out reconstruction within this dataset,
  not proof of independent property-level forecasting.
- The code uses random train/test and shuffled cross-validation splits, not spatial
  or temporal holdouts.
- The repository does not retain source-version, download, or license metadata for
  the supplied Project Sunroof-style file.
- The current test suite provides synthetic unit/regression coverage; it does not
  yet include full raw-data ETL, real-data model, or Streamlit integration tests.

## Documentation

- [Technical research report and reproducibility guide](project-print/technical-blueprint.md)
- [Research project overview](project-print/project-overview.txt)
- [Future work and research roadmap](FUTURE_WORK.md)
- [Documentation guide](project-print/navigate.txt)
- [Operational troubleshooting notes](project-print/debugging-notes.txt)
