# F1_RaceResult_Predictor
Machine learning model to predict finishing order and Top 10 finishers
# 🏎 F1 Race Result Predictor

This project uses machine learning to predict Formula 1 race outcomes, including:
-  Final race finishing position (regression)
-  Whether a driver will finish in the Top 10 (classification)

Built using multi-season data from the [FastF1](https://theoehrly.github.io/Fast-F1/) API, with qualifying times, weather data, and performance history across multiple tracks.

---

##  Features

- Multi-track training data from 2022–2024 (Silverstone, Spain, Japan, Bahrain, Austria)
- Rich feature set: grid position, driver/team form, track/weather data
- Random Forest Regressor for race position prediction
- Random Forest Classifier for Top 10 finish prediction
- Automatically computes driver/team points from race results
- MAE (regression): ~3.06
- F1 Score (Top 10 classifier): ~0.79

---

##  How It Works

- Uses FastF1 to fetch qualifying and race data for multiple seasons
- Builds features like:
  - `QualiTime(s)` (converted from FastF1 lap times)
  - `GridPosition` (based on qualifying rank)
  - Weather: AirTemp, TrackTemp, Rainfall
  - `DriverPoints` and `TeamPoints` (season progress till prediction round)
- Trains two models:
  - `RandomForestRegressor` for predicting final finish position
  - `RandomForestClassifier` for Top 10 prediction

---

##  Project Structure

```
f1-race-predictor/
├── train_predict.py         # Main script: data collection, model training, prediction
├── models/                  # Saved models (race_model.pkl, top10_classifier.pkl)
├── outputs/                 # Optional prediction CSVs or visualizations
├── requirements.txt         # Python dependencies
└── README.md                # Project overview (this file)
```

---

##  Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/Parakram_Code/f1-race-result-predictor.git
cd f1-race-result-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Predictor
```bash
python train_predict.py
```

This will:
- Pull qualifying and race results for 2022–2025 (selected tracks)
- Train the regression and classification models
- Predict results for the 2025 Silverstone GP
- Output Top 10 probabilities

---

##Sample Output

```
 2025 Predicted Race Results (Random Forest):
| Driver | GridPosition | PredictedFinish | Top10_Prob | Top10_Pred |
|--------|--------------|------------------|-------------|-------------|
| VER    | 2            | 1.0              | 0.97        | 1          |
| NOR    | 3            | 2.0              | 0.93        | 1          |
| STR    | 11           | 13.0             | 0.12        | 0          |
```

---

##  Tech Stack

- Python 3.11+
- [FastF1](https://theoehrly.github.io/Fast-F1/) for race data
- scikit-learn (ML models)
- pandas (data handling)
- matplotlib (optional visualizations)
- joblib (model saving)

---

##  Future Enhancements

-  Add a Streamlit or Flask UI for live predictions
-  Improve feature engineering (driver consistency, tire strategy, DNF risk)
-  Support more tracks and full 2025 calendar
-  Interactive charts (feature importance, quali vs finish delta)

---

##  License

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

##  Acknowledgements

- [FastF1](https://github.com/theOehrly/Fast-F1) for open access to F1 telemetry and results data
- scikit-learn for the core ML engine
  

---

Made by [Parakram Nain](https://github.com/ParakramCode)
