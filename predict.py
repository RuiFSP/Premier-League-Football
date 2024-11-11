import joblib
import os
import pandas as pd
from io import StringIO
from scripts.teams_data import get_teams_data

from flask import Flask, request, jsonify

path_model = os.path.join("models", "football_model.pkl")

# Load the model with error handling
try:
    with open(path_model, 'rb') as f:
        loaded_model = joblib.load(f)
except FileNotFoundError:
    raise Exception(f"Model file not found at {path_model}")

app = Flask("predict")

@app.route("/predict", methods=["POST"])
def predict():

    input = request.get_json()
    
    # Validate input data
    required_keys = {'home_team', 'away_team', 'date'}
    if not all(key in input for key in required_keys):
        return jsonify({"error": "Invalid input format. Expected keys: 'home_team', 'away_team', 'date'."}), 400
    
    # Get the teams data
    data_json = get_teams_data(input['home_team'], input['away_team'], input['date']).to_json(orient='records')

    # Convert JSON string to dictionary
    data = pd.read_json(StringIO(data_json), orient='records')

    # Validate input data
    if not isinstance(data, pd.DataFrame):
        return jsonify({"error": "Invalid input format. Expected a JSON object."}), 400

    # Convert data to DataFrame
    data = pd.DataFrame(data)

    result_dict = {
        0: 'Away_Win',
        1: 'Draw',
        2: 'Home_Win'
    }

    try:
        raw_prediction = loaded_model.predict(data)[0]
        probabilities = loaded_model.predict_proba(data)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Show information in dictionary format with prediction and probability
    prediction_prob_df = {
        'Prob_Away_Win': round(float(probabilities[0]), 3),
        'Prob_Draw': round(float(probabilities[1]), 3),
        'Prob_Home_Win': round(float(probabilities[2]), 3),
        'Match_Result': result_dict[raw_prediction]
    }

    return jsonify(prediction_prob_df)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)