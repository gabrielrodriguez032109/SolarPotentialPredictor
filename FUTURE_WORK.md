# Future Work and Research Roadmap

This document records the next steps for strengthening Solar Potential Predictor as
a credible, reproducible applied machine-learning study. It is a planning document:
items are not completed results or publication claims.

## Guiding principles

- Keep the unit of analysis explicit: the study concerns census-tract aggregates,
  not individual roofs or households.
- Keep offline model evaluation separate from the Streamlit demonstration. The app
  currently retrieves nearby tract source values and runs a transparent
  home-scale planning calculation; it does not serve Random Forest inference.
- Prefer reproducible evidence over presentation: record data identity, model
  configuration, and per-target outcomes before making stronger claims.
- Do not silently change cleaning rules, features, or assumptions. Document the
  reason, update tests, and regenerate derived artifacts deliberately.

## Priority 1 — Establish a defensible data record

- [ ] Add a data card describing the supplied Project Sunroof-style export, its
  fields, unit of analysis, known coverage, and intended use.
- [ ] Record source version, download date, license, and provenance details if they
  can be verified from the original data source.
- [ ] Define a formal ETL schema contract: required columns, expected types, valid
  coordinate ranges, and duplicate-handling policy.
- [ ] Decide whether missingness should be assessed on all raw fields or only the
  retained analytical fields. The current pipeline drops rows before column
  selection.
- [ ] Report and investigate data-quality flags, including values outside expected
  percentage ranges, without silently removing observations.

## Priority 2 — Make model evaluation research-ready

- [ ] Publish one versioned results artifact for a declared data snapshot:
  per-target RMSE, MAE, and R²; split seed; model configuration; package versions;
  and a hash of the raw and processed data files.
- [ ] Report Random Forest metrics separately for annual sunlight total, carbon
  offset, and capacity. Do not rely only on aggregate multi-output metrics that
  mix units.
- [ ] Compare the Random Forest and Linear Regression baseline on the same rows and
  same holdout partition for their shared outcomes.
- [ ] Add an ablation or sensitivity analysis that tests the contribution of
  directional sunlight features. Interpret the result as reconstruction analysis,
  because these fields are closely related to annual sunlight total.
- [ ] Add spatial validation, such as geographically grouped holdouts, before
  claiming generalization beyond nearby tracts in a random split.
- [ ] Retain actual-versus-predicted, residual, and feature-importance figures with
  each declared evaluation run. Treat feature importance as model-specific rather
  than causal.

## Priority 3 — Improve reproducibility and software quality

- [ ] Define a clear artifact policy for the raw CSV, processed CSV, SQLite
  database, model files, metrics, and figures: what is tracked, regenerated, and
  versioned.
- [ ] Pin or record dependency versions for a declared research run.
- [ ] Add real-data integration tests for raw-to-processed ETL, SQLite loading, and
  deterministic model evaluation.
- [ ] Add data-quality assertions that fail loudly when required fields are missing
  or when the processed schema changes.
- [ ] Add a Streamlit smoke test or a small integration check for the demonstration
  path.
- [ ] Add continuous integration to run the test suite on every change.

## Priority 4 — Clarify implementation boundaries

- [ ] Split or rename `predict_with_model` so the API distinguishes nearest-tract
  source lookup from offline machine-learning evaluation.
- [ ] Use a single geodesic method for nearest-tract selection and displayed
  distance; the current selection uses squared degrees while the display uses
  haversine distance.
- [ ] Keep homeowner assumptions in a clearly named, independently tested planning
  helper. Do not present them as model outputs.
- [ ] Move hard-coded paths, thresholds, and planning constants into documented
  configuration only when multiple environments or scenarios require it.

## Priority 5 — Strengthen research communication

- [ ] Add a concise model card covering purpose, inputs, targets, evaluation design,
  intended use, excluded use, and limitations.
- [ ] Add a results section or report appendix that links each figure/table to a
  specific data and code configuration.
- [ ] Use a short analysis report or reproducible script for exploratory work if
  needed; avoid empty placeholder notebooks.
- [ ] Keep the README, technical report, and Streamlit text aligned whenever the
  study design changes.

## Definition of a credible first research release

The project is ready for a stronger project-report or portfolio claim when it has:

1. A documented and identifiable data snapshot.
2. A validated ETL schema and data-quality report.
3. Per-target held-out results with a reproducible configuration record.
4. A baseline comparison on equivalent data partitions.
5. At least one spatial or grouped validation experiment.
6. Integration tests plus automated test execution.
7. Clear language separating tract-level modeling, home-scale assumptions, and the
   demonstration interface.

## Out of scope unless new evidence is added

Do not claim property-level production forecasts, installer recommendations,
financial savings, permit calculations, causal effects, or peer-reviewed
publication status. Those claims require additional data, validation, or external
review that this repository does not currently contain.

