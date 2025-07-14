import fastf1
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Enable FastF1 cache
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

# Track and year setup
tracks = ['Silverstone', 'Spain', 'Japan', 'Bahrain', 'Austria']
all_data = []
predict_year = 2025
predict_track = 'Silverstone'
POINTS_PER_POSITION = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Calculate standings using FastF1

def calculate_standings(year, tracks, until_track):
    driver_standings = {}
    team_standings = {}

    for current_track in tracks:
        session = fastf1.get_session(year, current_track, 'R')
        session.load()
        results = session.results[['Abbreviation', 'TeamName', 'Position']]

        for _, row in results.iterrows():
            driver = row['Abbreviation']
            team = row['TeamName']
            try:
                pos = int(row['Position'])
            except:
                continue
            pts = POINTS_PER_POSITION.get(pos, 0)
            driver_standings[driver] = driver_standings.get(driver, 0) + pts
            team_standings[team] = team_standings.get(team, 0) + pts

        if current_track == until_track:
            break

    return driver_standings, team_standings

# Main loop for all seasons
for year in [2022, 2023, 2024, 2025]:
    for current_track in tracks:
        quali = fastf1.get_session(year, current_track, 'Q')
        quali.load()
        quali_laps = quali.laps.pick_quicklaps()
        best_laps = [quali_laps.pick_driver(drv).pick_fastest() for drv in quali.drivers if not quali_laps.pick_driver(drv).empty]

        quali_df = pd.DataFrame(best_laps)[['Driver', 'Team', 'LapTime']]
        quali_df['LapTime'] = quali_df['LapTime'].dt.total_seconds()
        quali_df['QualiTime(s)'] = quali_df['LapTime']
        quali_df = quali_df.sort_values('QualiTime(s)').reset_index(drop=True)
        quali_df['GridPosition'] = quali_df.index + 1

        if year == predict_year and current_track == predict_track:
            prediction_data = quali_df.copy()
            prediction_data['Track'] = current_track
            continue

        race = fastf1.get_session(year, current_track, 'R')
        race.load()
        results = race.results[['Abbreviation', 'Position']].rename(columns={'Abbreviation': 'Driver', 'Position': 'RacePosition'})

        merged = pd.merge(quali_df, results, on='Driver')
        merged['Year'] = year
        merged['Track'] = current_track
        weather = race.weather_data
        merged['AirTemp'] = weather['AirTemp'].mean()
        merged['TrackTemp'] = weather['TrackTemp'].mean()
        merged['Rainfall'] = weather['Rainfall'].mean()
        merged['Weight'] = {2022: 0.5, 2023: 1.5, 2024: 3.0}.get(year, 1.0)

        # Add points inside loop
        dpoints, tpoints = calculate_standings(year, tracks, until_track=current_track)
        merged['DriverPoints'] = merged['Driver'].map(dpoints).fillna(0)
        merged['TeamPoints'] = merged['Team'].map(tpoints).fillna(0)

        all_data.append(merged)

# Combine data
training_df = pd.concat(all_data).reset_index(drop=True)

# Encode categorical features
encoders = {
    'Driver': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'Team': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'Track': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
}
training_df['DriverEncoded'] = encoders['Driver'].fit_transform(training_df[['Driver']])
training_df['TeamEncoded'] = encoders['Team'].fit_transform(training_df[['Team']])
training_df['Trackencoded'] = encoders['Track'].fit_transform(training_df[['Track']])

prediction_data['DriverEncoded'] = encoders['Driver'].transform(prediction_data[['Driver']])
prediction_data['TeamEncoded'] = encoders['Team'].transform(prediction_data[['Team']])
prediction_data['Trackencoded'] = encoders['Track'].transform(prediction_data[['Track']])

# Add 2025 driver/team points
dpoints, tpoints = calculate_standings(predict_year, tracks, until_track=predict_track)
prediction_data['DriverPoints'] = prediction_data['Driver'].map(dpoints).fillna(0)
prediction_data['TeamPoints'] = prediction_data['Team'].map(tpoints).fillna(0)
prediction_data['AirTemp'] = training_df['AirTemp'].mean()
prediction_data['TrackTemp'] = training_df['TrackTemp'].mean()
prediction_data['Rainfall'] = 0.70  # hypothetical

# Features & target
features = ['QualiTime(s)', 'GridPosition', 'DriverEncoded', 'TeamEncoded', 'AirTemp', 'TrackTemp', 'Rainfall', 'Trackencoded', 'DriverPoints', 'TeamPoints']
X = training_df[features]
y = training_df['RacePosition']
weights = training_df['Weight']
X_pred = prediction_data[features]

# Split and train
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, min_samples_split=2, random_state=42)
model.fit(X_train, y_train, sample_weight=w_train)

# Evaluation
y_pred = model.predict(X_test)
print("\n📊 Tuned Random Forest Evaluation:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# Predict 2025
prediction_data['PredictedFinish'] = model.predict(X_pred).round()
prediction_data = prediction_data.sort_values('PredictedFinish')
print("\n 2025 Predicted Race Results (Random Forest):")
print(prediction_data[['Driver', 'Team', 'GridPosition', 'PredictedFinish']])

importances = model.feature_importances_
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()


training_df['Top 10']=(training_df['RacePosition']<=10).astype(int)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,f1_score

features = ['QualiTime(s)', 'GridPosition', 'DriverEncoded', 'TeamEncoded', 'AirTemp', 'TrackTemp', 'Rainfall', 'Trackencoded', 'DriverPoints', 'TeamPoints']

X_cls=training_df[features]
y_cls=training_df['Top 10']

X_train_cls,X_test_cls,y_train_cls,y_test_cls,w_train_cls,w_test_cls=train_test_split(X_cls,y_cls,weights,test_size=0.2,random_state=42)

clf=RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42)
clf.fit(X_train_cls, y_train_cls, sample_weight=w_train_cls)
y_pred_cls = clf.predict(X_test_cls)

print("\n📊 Random Forest Classifier Evaluation:")
print(f"Accuracy: {accuracy_score(y_test_cls, y_pred_cls):.3f}")
print(f"F1 Score: {f1_score(y_test_cls, y_pred_cls):.3f}")
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_cls))
# Predict top 10 for 2025
prediction_data['PredictedTop10'] = clf.predict(prediction_data[features])
prediction_data = prediction_data.sort_values('PredictedTop10', ascending=False)
print("\n🏁 2025 Predicted Top 10 Finishers:")
print(prediction_data[['Driver', 'Team', 'GridPosition', 'PredictedFinish', 'PredictedTop10']])