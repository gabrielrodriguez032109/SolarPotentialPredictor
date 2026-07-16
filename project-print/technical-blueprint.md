# Solar Potential Predictor technical blueprint

## Architecture

```text
raw census-tract CSV
        |
        v
ETL: extract -> drop incomplete rows -> select solar fields
        |
        +--> processed CSV
        +--> SQLite: sunroof_clean
                    |
                    v
           Streamlit nearest-tract estimate
                    |
         +----------+----------+
         |                     |
 community source values   homeowner planning calculation
```

`src/etl/pipeline.py` is the data boundary. It produces a stable table with location,
coverage, directional sunlight, annual yield, total annual potential, carbon offset,
and total installed-capacity fields. It runs only under `__main__`, so importing its
helpers has no persistence side effect.

## User-facing calculation path

`predict_with_model` finds the smallest squared latitude/longitude distance in the
cleaned table and uses that record as the reference tract. Its `model` parameter is
kept for caller compatibility; public estimates intentionally do not use an ML model
to re-predict target columns that exist in the selected source record.

Community mode returns these direct source values:

| Display metric | Source field |
|---|---|
| Potential Annual Energy Generation | `yearly_sunlight_kwh_total` |
| Potential Carbon Reduction | `carbon_offset_metric_tons` |
| Potential Solar Capacity | `kw_total` |

Homeowner mode derives a planning result. It uses total roof area `A`, selected
orientation `O`, and broad shading factor `S`. Optional monthly electricity use `U`
changes the recommended capacity target, but never exceeds the roof-area capacity:

```text
roof capacity kW = clamp(A * 0.004, 1, 15)
system kW = roof capacity, or min(roof capacity, U * 12 / adjusted yield) when U is supplied
annual kWh = system kW * yearly_sunlight_kwh_kw_threshold_avg * O * S
carbon tons = annual kWh * (carbon_offset_metric_tons / yearly_sunlight_kwh_total)
panels = round(system kW / 0.4)
```

Orientation factors are South 1.00, West 0.90, East 0.88, and North 0.72. They apply
only to homeowner production and carbon output. Community totals are not adjusted,
because the source tract totals already combine its suitable roofs.

Shading is optional: Unknown is 1.00, Minimal is 0.95, Moderate is 0.85, and
Significant is 0.70. These are broad planning factors, not a shade survey. Home square
footage, roof condition, panel model, battery selection, and incentives are not inputs:
the project has no property-specific data or calculation model for them.

## Location lookup and geographic-match message

The app accepts coordinates or a five-digit US ZIP code. A ZIP is resolved through a
public ZIP-centroid response before the nearest-tract lookup; it is less specific than
coordinates and is labelled as such in the interface. The app calculates haversine
distance from the resulting coordinate to the selected tract center. At most 2 km is
high geographic match, at most 10 km is medium, and larger distances are low. This
describes source proximity only; it is not an accuracy or confidence claim.

## Evaluation workflow

The Random Forest workflow remains available for learning and evaluation. It builds
the feature matrix, performs a train/test split, records RMSE, MAE, MAPE, and R², and
can write model/plot artifacts. It should be treated as a separate analytical workflow
from the app's source-record estimates.

## Operational constraints

- A five-digit US ZIP code may be used instead of coordinates. ZIP lookup requires an
  available network connection to resolve the public ZIP centroid.
- Results are planning information, not a site survey or system design.
- Raw-data schema changes require an ETL rebuild before using the app or training.
