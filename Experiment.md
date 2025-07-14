# Experiment & Development Log – F1 Race Predictor

## Date Started: July 2025  
Author: Parakram Nain

---

## Baseline Model

- Model: RandomForestRegressor
- Features: Qualifying time, grid position
- Tracks used: Silverstone (2022–2025)
- Metrics:
  - Mean Absolute Error (MAE): ~3.6

---

## Experiment 1: Added Weather Features

- Added AirTemp, TrackTemp, and Rainfall as features
- Result: Slight drop in R² and small change in MAE
- Interpretation: Likely due to lack of strong variability or correlation

---

## Experiment 2: Year-based Sample Weights

- Motivation: Emphasize recent data
- Weights:
  - 2022: 0.5
  - 2023: 1.5
  - 2024: 3.0
- Result: Slight performance gain in regression accuracy

---

---

## Experiment 3: Ergast API for Points Lookup

- Objective: Retrieve live driver/constructor standings using Ergast API
- Method: Used `requests` to query season schedule and round number
- Issue: Repeated timeout errors from `ergast.com`, failed connection
- Result: Abandoned external API due to unreliability
- Solution: Switched to internal point calculation using FastF1 race results

---

## Experiment 4: Driver & Team Points

- Computed cumulative points up to prediction race using FastF1 results
- Mapped to drivers and teams
- Result: MAE dropped significantly (from ~3.39 to ~3.06), R² increased to 0.51
- Interpretation: Captures driver form and team performance well

---

## Experiment 5: Gradient Boosting Models

- Models tested: GradientBoostingRegressor (sklearn), XGBoost
- Result: Performed slightly worse than tuned RandomForest
- Conclusion: Kept Random Forest as default model

---

## Experiment 6: Top 10 Finish Classification

- Converted problem to binary classification: Is driver in Top 10?
- Model: RandomForestClassifier
- Features: Same as regression
- Metrics:
  - Accuracy: 81.3%
  - F1 Score: 0.79
- Benefit: More interpretable, less sensitive to mid-field noise



## Final Feature Set

- QualiTime(s)
- GridPosition
- DriverEncoded
- TeamEncoded
- Trackencoded
- AirTemp
- TrackTemp
- Rainfall
- DriverPoints
- TeamPoints

---

## Future Plans

- Add Streamlit UI for real-time prediction
- Expand track coverage (e.g., Monaco, Zandvoort, Brazil)
- Add driver consistency score and DNF prediction
- Visualize prediction confidence and error bounds
