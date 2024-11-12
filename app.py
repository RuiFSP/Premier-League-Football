import streamlit as st
import joblib
import os
import pandas as pd
from io import StringIO
from scripts.teams_data import get_teams_data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to the saved model and additional data
path_model = os.path.join(os.path.dirname(__file__), "models", "final_model_pipeline.pkl")
path_column_names = os.path.join(os.path.dirname(__file__), "models", "column_names.pkl")
path_dtypes = os.path.join(os.path.dirname(__file__), "models", "dtypes.pkl")

# Load the model and additional data
try:
    with open(path_model, 'rb') as f:
        loaded_model = joblib.load(f)
    with open(path_column_names, 'rb') as f:
        column_names = joblib.load(f)
    with open(path_dtypes, 'rb') as f:
        dtypes = joblib.load(f)
    logger.info("Model and additional data loaded successfully")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    st.error(f"File not found: {e}")

# Mapping for the prediction output
result_dict = {
    0: 'Away_Win',
    1: 'Draw',
    2: 'Home_Win'
}

# List of all team names in 2024/2025
list_of_teams = ['man united', 'ipswich', 'arsenal', 'everton', 'newcastle',
                 'nottm forest', 'west ham', 'brentford', 'chelsea', 'leicester',
                 'brighton', 'crystal palace', 'fulham', 'man city', 'southampton',
                 'tottenham', 'aston villa', 'bournemouth', 'wolves', 'liverpool']

# Streamlit UI
st.set_page_config(page_title="Premier League Match Predictor", layout="wide")
st.title("Premier League 2024/2025 Match Outcome Predictor")

st.sidebar.header("Match Details")
home_team = st.sidebar.selectbox("Home Team", list_of_teams)
away_team = st.sidebar.selectbox("Away Team", list_of_teams)
match_date = st.sidebar.date_input("Match Date")

st.sidebar.markdown("### Prediction")
if st.sidebar.button("Predict Outcome"):
    if home_team and away_team and match_date:
        try:
            # Get the teams data
            logger.info(f"Fetching data for {home_team} vs {away_team} on {match_date}")
            data_json = get_teams_data(home_team, away_team, str(match_date)).to_json(orient='records')
            data = pd.read_json(StringIO(data_json), orient='records')
            
            # Validate data format
            if not isinstance(data, pd.DataFrame):
                st.error("Invalid data format received. Expected a DataFrame.")
            else:
                # Making predictions
                raw_prediction = loaded_model.predict(data)[0]
                probabilities = loaded_model.predict_proba(data)[0]

                # Format prediction result for display
                prediction_result = {
                    'Match Result': result_dict[raw_prediction],
                    'Probability Home Win': round(float(probabilities[2]), 3),
                    'Probability Draw': round(float(probabilities[1]), 3),
                    'Probability Away Win': round(float(probabilities[0]), 3)
                }

                st.subheader("Prediction Result")
                st.write(prediction_result)

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please enter all the required fields.")