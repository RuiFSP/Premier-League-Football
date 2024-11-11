def predict_outcome(home_team, away_team, day_of_week, month):
    
    # Filter rows for specific teams in home_stats and away_stats
    home_team_data = home_stats[home_stats['home_team'] == home_team]
    away_team_data = away_stats[away_stats['away_team'] == away_team]

    # Calculate common column values for the specific game day and month
    day_of_week_sin = np.sin(day_of_week * (2. * np.pi / 7))
    day_of_week_cos = np.cos(day_of_week * (2. * np.pi / 7))
    month_sin = np.sin(month * (2. * np.pi / 12))
    month_cos = np.cos(month * (2. * np.pi / 12))

    # Create a DataFrame for common columns as a single row
    common_data = pd.DataFrame({
        'day_of_week': [day_of_week],
        'month': [month],
        'day_of_week_sin': [day_of_week_sin],
        'day_of_week_cos': [day_of_week_cos],
        'month_sin': [month_sin],
        'month_cos': [month_cos]
    })

    # Reset index to enable a clean horizontal concatenation
    home_team_data = home_team_data.reset_index(drop=True)
    away_team_data = away_team_data.reset_index(drop=True)
    common_data = common_data.reset_index(drop=True)

    # Concatenate along the columns
    combined_data = pd.concat([home_team_data, away_team_data, common_data], axis=1)

    data_point = combined_data[['home_team',
            'away_team',
            'day_of_week',
            'month',
            'day_of_week_sin',
            'day_of_week_cos',
            'month_sin',
            'month_cos',
            'home_roll_3_avg_home_corners',
            'away_roll_3_avg_home_corners',
            'home_roll_3_avg_away_corners',
            'away_roll_3_avg_away_corners',
            'home_roll_3_avg_home_yellow_cards',
            'away_roll_3_avg_home_yellow_cards',
            'home_roll_3_avg_away_yellow_cards',
            'away_roll_3_avg_away_yellow_cards',
            'home_roll_3_avg_home_red_cards',
            'away_roll_3_avg_home_red_cards',
            'home_roll_3_avg_away_red_cards',
            'away_roll_3_avg_away_red_cards',
            'home_roll_3_avg_ratio_h_a_shots',
            'away_roll_3_avg_ratio_h_a_shots',
            'home_roll_3_avg_ratio_h_a_fouls',
            'away_roll_3_avg_ratio_h_a_fouls',
            'home_roll_3_avg_ratio_a_h_shots',
            'away_roll_3_avg_ratio_a_h_shots',
            'home_roll_3_avg_ratio_a_h_fouls',
            'away_roll_3_avg_ratio_a_h_fouls',
            'home_roll_5_avg_home_corners',
            'away_roll_5_avg_home_corners',
            'home_roll_5_avg_away_corners',
            'away_roll_5_avg_away_corners',
            'home_roll_5_avg_home_yellow_cards',
            'away_roll_5_avg_home_yellow_cards',
            'home_roll_5_avg_away_yellow_cards',
            'away_roll_5_avg_away_yellow_cards',
            'home_roll_5_avg_home_red_cards',
            'away_roll_5_avg_home_red_cards',
            'home_roll_5_avg_away_red_cards',
            'away_roll_5_avg_away_red_cards',
            'home_roll_5_avg_ratio_h_a_shots',
            'away_roll_5_avg_ratio_h_a_shots',
            'home_roll_5_avg_ratio_a_h_shots',
            'away_roll_5_avg_ratio_a_h_shots',
            'home_cumulative_points',
            'away_cumulative_points']]

    # Make a prediction using the loaded model
    prediction = result_dict[loaded_model.predict(data_point)[0]]

    # predict probability
    prob = loaded_model.predict_proba(data_point)

    # show information in dictionary format with prediction and probability
    prediction_prob_df = {
        'Home': round(float(prob[0][2]),3),
        'Draw': round(float(prob[0][1]),3),
        'Away': round(float(prob[0][0]),3),
        'Prediction': prediction
    }
    
    return prediction_prob_df