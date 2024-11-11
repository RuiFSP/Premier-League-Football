import requests
url = "http://127.0.0.1:9696/predict"
headers = {"Content-Type": "application/json"}

match_data = {
    "home_team": "arsenal",
    "away_team": "brentford",
    "day_of_week": 4,
    "month": 8,
    "day_of_week_sin": -0.433883739117558,
    "day_of_week_cos": -0.9009688679024191,
    "month_sin": -0.8660254037844384,
    "month_cos": -0.5000000000000004,
    "home_roll_3_avg_home_corners": 11.0,
    "away_roll_3_avg_home_corners": 10.0,
    "home_roll_3_avg_away_corners": 2.6666666666666665,
    "away_roll_3_avg_away_corners": 3.0,
    "home_roll_3_avg_home_yellow_cards": 1.6666666666666667,
    "away_roll_3_avg_home_yellow_cards": 2.6666666666666665,
    "home_roll_3_avg_away_yellow_cards": 3.0,
    "away_roll_3_avg_away_yellow_cards": 1.3333333333333333,
    "home_roll_3_avg_home_red_cards": 0.3333333333333333,
    "away_roll_3_avg_home_red_cards": 0.0,
    "home_roll_3_avg_away_red_cards": 0.0,
    "away_roll_3_avg_away_red_cards": 0.0,
    "home_roll_3_avg_ratio_h_a_shots": 3.775,
    "away_roll_3_avg_ratio_h_a_shots": 2.986111111111111,
    "home_roll_3_avg_ratio_h_a_fouls": 1.5529100529100528,
    "away_roll_3_avg_ratio_h_a_fouls": 2.340740740740741,
    "home_roll_3_avg_ratio_a_h_shots": 0.8049169859514688,
    "away_roll_3_avg_ratio_a_h_shots": 0.3510466988727858,
    "home_roll_3_avg_ratio_a_h_fouls": 0.6762626262626262,
    "away_roll_3_avg_ratio_a_h_fouls": 0.5028860028860029,
    "home_roll_5_avg_home_corners": 0.0,
    "away_roll_5_avg_home_corners": 0.0,
    "home_roll_5_avg_away_corners": 0.0,
    "away_roll_5_avg_away_corners": 0.0,
    "home_roll_5_avg_home_yellow_cards": 0.0,
    "away_roll_5_avg_home_yellow_cards": 0.0,
    "home_roll_5_avg_away_yellow_cards": 0.0,
    "away_roll_5_avg_away_yellow_cards": 0.0,
    "home_roll_5_avg_home_red_cards": 0.0,
    "away_roll_5_avg_home_red_cards": 0.0,
    "home_roll_5_avg_away_red_cards": 0.0,
    "away_roll_5_avg_away_red_cards": 0.0,
    "home_roll_5_avg_ratio_h_a_shots": 0.0,
    "away_roll_5_avg_ratio_h_a_shots": 0.0,
    "home_roll_5_avg_ratio_a_h_shots": 0.0,
    "away_roll_5_avg_ratio_a_h_shots": 0.0,
    "home_cumulative_points": 11,
    "away_cumulative_points": 0
}

response = requests.post(url, headers=headers, json=match_data)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)