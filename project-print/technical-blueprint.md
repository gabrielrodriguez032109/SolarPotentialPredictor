# Solar Potential Predictor — Technical Blueprint

## 1. System purpose and scope

This repository is a local Python/Streamlit planning application built around a
Project Sunroof-style **census-tract** CSV. It answers two different questions:

1. **Community Solar Potential Estimate** — what are the stored aggregate solar
   figures for the census tract nearest to a requested point?
2. **Residential Solar Recommendation** — given an approximate roof area and the
   nearest tract's yield/carbon context, what is a deliberately simple, capped
   one-home planning estimate?

It is not property-level geospatial analysis, an engineering design, an installer
quote, a financial model, a permit calculation, or a live solar-data service. The
app's public prediction path intentionally does not use machine-learning inference:
it retrieves a nearby source record. The Random Forest and Linear Regression modules
are separate evaluation/learning workflows.

## 2. Repository inventory

| Path | Role | Runtime/tracking notes |
|---|---|---|
| `README.md` | Setup and high-level product guide | References a nonexistent `PLAN.txt`; update if documentation is reorganized. |
| `requirements.txt` | Python dependencies | `kagglehub` and `plotly` are not imported by current code. |
| `data/raw/sunroof_solar_potential_by_censustract.csv` | Source census-tract export | 48,722 rows, 31 columns, ~30.3 MB. |
| `src/etl/pipeline.py` | Extract/transform/load script | Regenerates both processed artifacts. |
| `data/processed/sunroof_clean.csv` | Narrowed inspection artifact | 48,664 rows, 13 columns, ~9.9 MB. |
| `data/processed/solar.db` | Runtime SQLite source | Table `sunroof_clean`, 48,664 rows, 13 columns. |
| `src/models/random_forest.py` | Estimation helpers and RF evaluation workflow | Public app path uses its nearest-row helper, not `model.predict`. |
| `src/models/linear_regression.py` | Linear-regression evaluation workflow | Writes two tracked PNG charts. |
| `src/app/app.py` | Streamlit UI | Executes at import/run time; no `main()` wrapper. |
| `tests/test_random_forest_output.py` | Unit/regression tests | Synthetic-data focused; no app or real-artifact integration test. |
| `src/model-output/*.png` | Existing Linear Regression charts | Generated outputs currently committed. |
| `notebooks/A-Placeholder.ipynb`, `D-Placeholder.ipynb` | Empty files | Zero bytes and not valid notebooks. |
| `notebooks/G-Placeholder.ipynb` | Empty valid notebook shell | No cells; Python 3.13.11 metadata. |
| `project-print/*.txt/.md` | Human and agent documentation | `agent-planning.md` is maintained as the operational handoff. |

Git currently tracks the raw data, processed CSV/database, and Linear Regression
PNGs. `.gitignore` ignores future `*.db`, `src/model-output/*.png`, JSON metrics,
and saved model files, so this is an inconsistent but currently working artifact
policy.

## 3. Data contract

### 3.1 Raw source

The raw CSV contains these 31 fields:

```text
carbon_offset_metric_tons, count_qualified, existing_installs_count,
install_size_kw_buckets, kw_median, kw_total, lat_avg, lat_max, lat_min,
lng_avg, lng_max, lng_min, number_of_panels_e, number_of_panels_f,
number_of_panels_median, number_of_panels_n, number_of_panels_s,
number_of_panels_total, number_of_panels_w, percent_covered,
percent_qualified, region_name, state_name, yearly_sunlight_kwh_e,
yearly_sunlight_kwh_f, yearly_sunlight_kwh_kw_threshold_avg,
yearly_sunlight_kwh_median, yearly_sunlight_kwh_n, yearly_sunlight_kwh_s,
yearly_sunlight_kwh_total, yearly_sunlight_kwh_w
```

It has 58 rows with at least one empty field. `transform` removes those rows before
column selection, producing 48,664 rows. This means a null in an unused raw field
also discards a record; it is a conscious current behavior, not a selected-column-only
null check. No source metadata, download process, license, or dataset version is held
in the repository.

### 3.2 Processed schema

`pipeline.transform` keeps this ordered 13-field contract:

| Field | Meaning/use |
|---|---|
| `lat_avg`, `lng_avg` | Census-tract center; nearest-record lookup. |
| `count_qualified` | Model feature. |
| `percent_covered`, `percent_qualified` | Model features. |
| `yearly_sunlight_kwh_n/e/s/w` | Directional aggregate values; RF features and orientation comparison. |
| `yearly_sunlight_kwh_kw_threshold_avg` | Local annual yield per kW used in homeowner calculation. |
| `yearly_sunlight_kwh_total` | Community annual generation figure; RF/LR target. |
| `carbon_offset_metric_tons` | Community carbon figure; RF/LR target. |
| `kw_total` | Community capacity figure; RF target. |

The clean CSV is written without an index. SQLite is created using SQLAlchemy at
`sqlite:///data/processed/solar.db`; `to_sql(..., if_exists="replace")` replaces the
whole `sunroof_clean` table on every ETL run. There are no indexes, constraints,
primary keys, or migrations. The current SQLite column types are FLOAT except
`count_qualified` (BIGINT).

## 4. ETL lifecycle

```text
raw CSV
  -> pandas.read_csv (extract)
  -> DataFrame copy, drop any raw row containing null, reset index (transform)
  -> select fields that happen to exist in the raw export (transform)
  -> processed CSV + replacement SQLite table (load)
  -> Streamlit and model workflows read SQLite
```

Run it from repository root with `python src/etl/pipeline.py`. Relative paths make
the root working directory mandatory. Importing the module is side-effect free;
execution happens only under `if __name__ == "__main__"`.

The selection is tolerant of missing source columns: absent names are silently omitted.
That makes ETL itself succeed with an older export, but app/model code may subsequently
fail with a less-focused missing-column error. There is no validation of expected
schema, types, physical bounds, duplicate tracts, or raw/processed row count.

## 5. Application request flow

```text
Streamlit form
  -> validate_inputs
     -> coordinate path: validate numeric latitude/longitude
     -> ZIP path: validate five digits -> resolve_zip_code (Zippopotam.us HTTPS)
  -> load_data: SELECT * FROM SQLite sunroof_clean
  -> predict_with_model(None, dataframe, final coordinate, UI choices)
     -> engineer directional columns
     -> choose nearest source row
     -> community source values OR homeowner calculation
  -> render metrics, tract/proximity details, and directional bar chart
```

### 5.1 Inputs and validation

- Coordinates require both values; latitude is -90..90 and longitude is -180..180.
- A nonempty ZIP takes precedence over coordinates, must be exactly five digits, and
  is resolved with `https://api.zippopotam.us/us/{ZIP}` and a five-second timeout.
- ZIP resolution catches network/parse/shape errors and returns one generic UI-safe
  `ValueError`; it uses the first returned place centroid, not an address.
- Orientation normalizes to title case and must be North/South/East/West.
- Shading normalizes to title case and must be Unknown/Minimal/Moderate/Significant.
- Monthly electricity, when supplied, must be positive. The app’s number input also
  enforces a 1 kWh minimum.
- The UI permits roof area from 250 sq ft upward. The library helper accepts any
  number and coerces it to at least 1 sq ft.

### 5.2 Nearest tract and geographic match

`predict_with_model` engineers two columns, drops incomplete rows, then selects the
minimum of `(lat_avg - lat)^2 + (lng_avg - lng)^2`. It does **not** call the passed
model; the parameter remains for compatibility. The displayed distance uses a
haversine calculation and is labelled High (<=2 km), Medium (<=10 km), or Low (>10
km) geographic match. It communicates proximity to the stored tract center only,
not accuracy/confidence.

The selection metric is a fast squared-degree approximation, while the displayed
metric is spherical distance. They can choose different rows, especially at high
latitudes or across a large longitude span; use haversine for selection if geographic
correctness becomes important.

### 5.3 Community mode

The selected source record supplies, unchanged:

| UI metric | Field |
|---|---|
| Potential Annual Energy Generation | `yearly_sunlight_kwh_total` |
| Potential Carbon Reduction | `carbon_offset_metric_tons` |
| Potential Solar Capacity | `kw_total` (or generation/yield fallback) |

Home-only inputs are ignored. The directional bar chart is always based on the four
directional source fields, and the UI labels the largest as “Best orientation.” These
are tract-level directional aggregates; they are not a one-home production forecast.

### 5.4 Homeowner mode

The selected tract still supplies local yield and carbon rate, but the outputs are
calculated. Let `A` be roof area, `Y` the tract’s
`yearly_sunlight_kwh_kw_threshold_avg`, `O` the orientation factor, `S` the shading
factor, and optional `U` monthly usage.

```text
roof ceiling kW       = clamp(A * 0.004, 1.0, 15.0)
adjusted yield        = Y * O * S
usage-target kW       = (U * 12) / max(adjusted yield, 1.0)
system kW             = roof ceiling, without U
                      = min(max(usage-target kW, 1.0), roof ceiling), with U
annual production     = system kW * adjusted yield
tract carbon rate     = carbon_offset_metric_tons / max(yearly_sunlight_kwh_total, 1.0)
annual carbon         = annual production * tract carbon rate
panel count           = max(round(system kW / 0.4), 1)
usage offset percent  = min(production / (U * 12) * 100, 100), if U exists
```

Factors are South 1.00, West 0.90, East 0.88, North 0.72; shading is Unknown 1.00,
Minimal 0.95, Moderate 0.85, Significant 0.70. They are static planning assumptions.
The code does not model usable roof geometry, azimuth/tilt, roof condition, panel
choice, battery, tariff, incentives, exports, site-specific shade, or degradation.

## 6. Model/evaluation workflows

### Random Forest

`random_forest.py` defines 12 canonical features: five location/coverage fields,
five raw yield fields, plus engineered directional sum and south:north ratio. It
targets annual total, carbon tons, and `kw_total`. `engineer_features` adds the sum
and computes the ratio after replacing north=0 with NaN.

`prepare_data` dynamically uses the available canonical fields, drops incomplete rows,
and returns NumPy arrays. `train_random_forest` uses an 80/20 seeded split (42) and a
200-tree `RandomForestRegressor`. It reports aggregate multi-output RMSE, MAE, MAPE,
and R². Five-fold shuffled CV reports R² only for target 0 (annual generation).

Running `python src/models/random_forest.py` writes ignored `random_forest.pkl`,
`model_metrics.json`, feature-importance, residual, and actual-vs-predicted PNGs.
Existing model output does not include these artifacts. `load_or_train_model` exists,
but Streamlit does not invoke it.

### Linear Regression

`linear_regression.py` uses the first 10 unengineered RF features and two targets
(annual total and carbon). It uses the same seeded 80/20 split, trains multi-output
`LinearRegression`, reports RMSE/R², and saves one actual-vs-predicted PNG per target.
It does not serialize a model or participate in application behavior.

### Important implementation caveats

- Both model sets include directional yield components while predicting a total closely
  related to them. Metrics should be interpreted as reconstruction performance, not as
  evidence of a useful unseen-property predictor.
- The RF dynamic-schema fallback is incomplete: `main()` always passes the fixed
  three-name target list to plotting, and `plot_feature_importance()` always uses the
  fixed 12-name feature list. An older partial schema can therefore train then fail
  during reporting due to length/index mismatch.
- `format_prediction_summary` is currently unused by Streamlit. Its fixed claim that
  the best orientation is “approximately 11%” better is not calculated from values.

## 7. Tests and verification boundary

The single test module contains 13 tests. They cover validation, ZIP parsing with an
injected fake request, feature engineering, nearest-row payloads, homeowner sizing,
community/homeowner separation, ETL selected columns, and legacy signature fallback.
They use small synthetic frames. They do not run ETL against raw data, inspect the
actual database, start Streamlit, test real ZIP networking, train/evaluate full data,
or test invalid/missing SQLite artifacts.

## 8. Runbook

From repository root:

```powershell
python -m pip install -r requirements.txt
python src/etl/pipeline.py
python src/models/random_forest.py          # optional evaluation artifacts
python src/models/linear_regression.py      # optional comparison charts
python -m streamlit run src/app/app.py
python -m pytest -q
```

The current default interpreter observed during documentation review was Python 3.14
at `E:\Coding\Tools\Python\python.exe`, and it did not have pandas installed. Use the
project virtual environment (or install the requirements) before claiming tests or
scripts passed. No test execution result is implied by this document.

## 9. Primary maintenance risks and recommended direction

1. Make artifact policy explicit: either regenerate processed data in setup/CI and
   stop tracking it, or track it deliberately and version the source/schema.
2. Validate required raw columns before transformation and limit null dropping to the
   selected contract fields (unless all-field completeness is intentionally required).
3. Use a single geodesic nearest-neighbor implementation for both selection and
   reported distance; consider spatial indexing for growth.
4. Rename `predict_with_model` or split source lookup from ML evaluation to eliminate
   the misleading public API and unused model argument.
5. Add actual-data ETL/schema tests, deterministic integration tests for SQLite and
   prediction, and a UI smoke test; record model metrics with data version/date.
6. Fix stale README links and decide whether placeholder notebooks should be removed
   or replaced with useful, valid analyses.
