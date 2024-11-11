import joblib
import os
import pandas as pd

from flask import Flask
from flask import request
from flask import jsonify


path_model = os.path.join("models", "football_model.pkl")

with open(path_model, 'rb') as f:
    loaded_model = joblib.load(f)
    
app = Flask("predict")


@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.get_json()
    
    # Convert data to DataFrame
    data = pd.DataFrame(data, index=[0])
    
    result_dict = {
        0 : 'Away_Win',
        1 : 'Draw',
        2 : 'Home_Win'
    }
    
    raw_prediction = loaded_model.predict(data)[0]
    
    #print(f"This is what i got: {raw_prediction}")
    
    probabilies = loaded_model.predict_proba(data).sor

    
    # show information in dictionary format with prediction andprobability
    prediction_prob_df = {
        'Prob_Away_Win': round(float(probabilies [0][0]),3),
        'Prob_Draw': round(float(probabilies [0][1]),3),
        'Prob_Home_Win': round(float(probabilies [0][2]),3),
        'Team_to_Win': result_dict[raw_prediction]
    }
    
    return jsonify(prediction_prob_df)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)