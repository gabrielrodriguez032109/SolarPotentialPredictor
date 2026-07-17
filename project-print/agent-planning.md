# Agent Planning and Handoff Notes

## Current state — 2026-07-17

The product is implemented as a small local Streamlit application over a processed
Project Sunroof-style census-tract dataset. It currently has two user-visible modes:
community (direct nearby tract totals) and homeowner (a transparent roof-area/yield
calculation). Do not describe the public results as Random Forest predictions. The
app imports `random_forest.py`, but calls `predict_with_model(None, ...)`, which
selects a source row and never calls `model.predict`.

The complete architecture, schema, formulas, execution paths, caveats, and runbook
are in `project-print/technical-blueprint.md`. Read that first, then use this file as
the active work ledger and decision context.

## Working model of the codebase

```text
raw 31-column CSV (48,722 rows)
  -> src/etl/pipeline.py: drop all rows with any null; select 13 fields
  -> clean CSV + SQLite sunroof_clean (48,664 rows)
  -> Streamlit app loads SQLite per submitted request
  -> random_forest.predict_with_model selects nearest tract
       -> community: returns source annual/carbon/capacity values
       -> homeowner: uses roof area, tract yield/carbon rate, orientation, shading,
          and optional monthly consumption

Separate, non-app workflows: Random Forest evaluation and Linear Regression evaluation
```

### Main files and ownership boundaries

- `src/etl/pipeline.py`: only code permitted to rewrite processed CSV/SQLite during
  normal operation. Relative paths assume repository-root CWD.
- `src/models/random_forest.py`: input validation, ZIP resolution, feature engineering,
  nearest-source lookup, homeowner formulas, RF training/evaluation/artifacts.
- `src/app/app.py`: Streamlit widgets and rendering. It should remain thin and should
  not acquire data/model calculations that can be tested in the model module.
- `src/models/linear_regression.py`: independent baseline/evaluation script only.
- `tests/test_random_forest_output.py`: 13 synthetic unit tests; extend here or split
  by responsibility when adding integration/UI coverage.
- `data/raw/`: immutable supplied source. `data/processed/` is derivable, though it is
  currently committed.

## Non-negotiable domain boundaries

- A tract aggregate is not one home's potential. Community output must stay source
  data and must not be altered by orientation, shade, or household use.
- Homeowner output is a first-pass planning estimate, not a surveyed design. Preserve
  clear limitations around usable roof area, shading, roof geometry, equipment,
  economics, and engineering.
- A ZIP is an approximate centroid lookup through Zippopotam.us; coordinates are more
  specific. The geographic label communicates tract-center distance, not model
  certainty.
- Keep the Random Forest workflow clearly separate from public source-record results
  until there is a legitimate predictive use case and proper validation plan.

## Exact homeowner calculation contract

Constants live in `random_forest.py`:

```text
HOMEOWNER_KW_PER_SQFT = 0.004; min/max system = 1/15 kW; panel = 0.4 kW
orientation: South 1.00, West 0.90, East 0.88, North 0.72
shade: Unknown 1.00, Minimal 0.95, Moderate 0.85, Significant 0.70
```

Roof capacity is `clamp(area * 0.004, 1, 15)`. Without usage, it is the proposed
size. With monthly kWh, proposed size targets annual use at adjusted local yield but
cannot exceed that roof capacity. Production is proposed kW × local annual kWh/kW ×
orientation × shade. Carbon is production × tract carbon tons/kWh. Panels are rounded
from kW/0.4. Change formulas/constants only with updated copy, tests, blueprint, and
explicit product agreement.

## Known issues / improvement queue

Prioritize based on the requested scope rather than changing these opportunistically.

1. **Geographic correctness:** selection uses squared raw degrees; display uses
   haversine. Use one geodesic selection metric, ideally with a scalable spatial index.
2. **ETL contract:** raw `dropna()` occurs before selecting 13 needed fields and
   silently tolerates missing required columns. Validate schema/types and define the
   intended null policy.
3. **Artifact lifecycle:** generated DB/CSV/PNGs are committed but the ignore rules
   ignore future equivalent outputs. Pick a reproducible policy and add a data version.
4. **Model validity:** targets are highly related to model inputs. Clarify evaluation
   purpose or redesign the modeling problem before presenting metrics as predictive
   performance.
5. **RF compatibility paths:** dynamic schema support conflicts with fixed plotting
   names in `main`; either make reporting dynamic or remove unsupported fallback.
6. **Docs hygiene:** README points at missing `PLAN.txt`; zero-byte A/D notebooks are
   invalid; G is an empty valid shell. Resolve rather than imply notebook analysis.
7. **Test depth:** add real raw-to-DB ETL/schema tests, deterministic nearest-tract
   tests with geographic edge cases, and a Streamlit smoke test. The current 13 tests
   are not an end-to-end safety net.
8. **Configuration:** paths, remote ZIP endpoint, constants, and geographic thresholds
   are hard-coded. Introduce configuration only if deployment/environments require it.

## Safe change checklist

1. Read `technical-blueprint.md` and inspect `git status`; preserve unrelated user
   changes. `agent-planning.md` began as an untracked empty file and is now the
   intended agent handoff document.
2. When editing ETL fields, update `selected_columns`, both model feature/target sets,
   app behavior, tests, documentation, and regenerate outputs deliberately.
3. When editing formulas, add exact numeric unit tests and align UI descriptions and
   product-overview text.
4. When editing app behavior, keep network calls only on submit, retain friendly
   validation errors, and avoid training/rebuilding artifacts in the UI request path.
5. Validate with `python -m pytest -q`; run ETL only when changes warrant generated
   artifact updates. Exercise both coordinate and ZIP paths manually if UI changes.
6. State whether generated data/model artifacts were refreshed and which interpreter
   was used. The default Python observed here is 3.14 without pandas, so prepare a
   virtual environment before verification.

## Decisions already encoded in the implementation

- SQLite is the runtime data source; CSV is inspection/debug output.
- The nearest tract is chosen at request time; there is no persisted prediction cache.
- ZIP lookup is the only network dependency and is injectible in unit tests.
- Community `kw_total` falls back to annual generation/local yield for older rows.
- The app does not require `random_forest.pkl`; training is optional.
- The prediction wrapper exists solely for legacy callable signatures. New work should
  avoid adding more compatibility layers without a concrete external consumer.

## Suggested next task sequence (if product hardening is requested)

1. Confirm raw data provenance/version and decide generated-artifact policy.
2. Add strict ETL schema validation plus data-quality summary, then update tests.
3. Correct nearest-tract selection and add geodesic regression tests.
4. Split source lookup/homeowner calculator from RF evaluation under clearer names.
5. Add integration and UI smoke coverage; only then report verified model/app health.

No production deployment, CI pipeline, environment file, secrets, database migration,
API server, authentication, external analytics, or persistent user data exists in the
current repository.
