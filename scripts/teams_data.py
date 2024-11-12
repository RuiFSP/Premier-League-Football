import numpy as np
import pandas as pd
import os

def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'teams_stats_2024.csv'))

def extract_team_stats(df, team_column, features):
    return df.groupby(team_column).tail(1)[features].sort_values(by=team_column)

def filter_team_data(stats, team, team_column):
    return stats[stats[team_column] == team].reset_index(drop=True)

def calculate_date_features(date):
    date = pd.to_datetime(date)
    day_of_week = date.dayofweek
    month = date.month
    day_of_week_sin = np.sin(day_of_week * (2. * np.pi / 7))
    day_of_week_cos = np.cos(day_of_week * (2. * np.pi / 7))
    month_sin = np.sin(month * (2. * np.pi / 12))
    month_cos = np.cos(month * (2. * np.pi / 12))
    return pd.DataFrame({
        'day_of_week': [day_of_week],
        'month': [month],
        'day_of_week_sin': [day_of_week_sin],
        'day_of_week_cos': [day_of_week_cos],
        'month_sin': [month_sin],
        'month_cos': [month_cos]
    }).reset_index(drop=True)

def get_teams_data(home_team, away_team, date):
    df = load_data()
    
    features = ['home_team', 'away_team', 'day_of_week', 'month', 'day_of_week_sin',
       'day_of_week_cos', 'month_sin', 'month_cos',
       'home_roll_3_avg_home_corners', 'away_roll_3_avg_home_corners',
       'home_roll_3_avg_away_corners', 'away_roll_3_avg_away_corners',
       'home_roll_3_avg_home_yellow_cards',
       'away_roll_3_avg_home_yellow_cards',
       'home_roll_3_avg_away_yellow_cards',
       'away_roll_3_avg_away_yellow_cards', 'home_roll_3_avg_home_red_cards',
       'away_roll_3_avg_home_red_cards', 'home_roll_3_avg_away_red_cards',
       'away_roll_3_avg_away_red_cards', 'home_roll_3_avg_ratio_h_a_shots',
       'away_roll_3_avg_ratio_h_a_shots', 'home_roll_3_avg_ratio_h_a_fouls',
       'away_roll_3_avg_ratio_h_a_fouls', 'home_roll_3_avg_ratio_a_h_shots',
       'away_roll_3_avg_ratio_a_h_shots', 'home_roll_3_avg_ratio_a_h_fouls',
       'away_roll_3_avg_ratio_a_h_fouls', 'home_roll_5_avg_home_corners',
       'away_roll_5_avg_home_corners', 'home_roll_5_avg_away_corners',
       'away_roll_5_avg_away_corners', 'home_roll_5_avg_home_yellow_cards',
       'away_roll_5_avg_home_yellow_cards',
       'home_roll_5_avg_away_yellow_cards',
       'away_roll_5_avg_away_yellow_cards', 'home_roll_5_avg_home_red_cards',
       'away_roll_5_avg_home_red_cards', 'home_roll_5_avg_away_red_cards',
       'away_roll_5_avg_away_red_cards', 'home_roll_5_avg_ratio_h_a_shots',
       'away_roll_5_avg_ratio_h_a_shots', 'home_roll_5_avg_ratio_a_h_shots',
       'away_roll_5_avg_ratio_a_h_shots', 'home_cumulative_points',
       'away_cumulative_points']
    
    df_home_stats = extract_team_stats(df, 'home_team', features)
    df_away_stats = extract_team_stats(df, 'away_team', features)

    home_stats_columns = [
        'home_team', 'home_roll_3_avg_home_corners', 'home_roll_3_avg_away_corners', 
        'home_roll_3_avg_home_yellow_cards', 'home_roll_3_avg_away_yellow_cards', 
        'home_roll_3_avg_home_red_cards', 'home_roll_3_avg_away_red_cards', 
        'home_roll_3_avg_ratio_h_a_shots', 'home_roll_3_avg_ratio_h_a_fouls', 
        'home_roll_3_avg_ratio_a_h_shots', 'home_roll_3_avg_ratio_a_h_fouls', 
        'home_roll_5_avg_home_corners', 'home_roll_5_avg_away_corners', 
        'home_roll_5_avg_home_yellow_cards', 'home_roll_5_avg_away_yellow_cards', 
        'home_roll_5_avg_home_red_cards', 'home_roll_5_avg_away_red_cards', 
        'home_roll_5_avg_ratio_h_a_shots', 'home_roll_5_avg_ratio_a_h_shots',
        'home_cumulative_points'
    ]

    away_stats_columns = [
        'away_team', 'away_roll_3_avg_home_corners', 'away_roll_3_avg_away_corners', 
        'away_roll_3_avg_home_yellow_cards', 'away_roll_3_avg_away_yellow_cards', 
        'away_roll_3_avg_home_red_cards', 'away_roll_3_avg_away_red_cards', 
        'away_roll_3_avg_ratio_h_a_shots', 'away_roll_3_avg_ratio_h_a_fouls', 
        'away_roll_3_avg_ratio_a_h_shots', 'away_roll_3_avg_ratio_a_h_fouls', 
        'away_roll_5_avg_home_corners', 'away_roll_5_avg_away_corners', 
        'away_roll_5_avg_home_yellow_cards', 'away_roll_5_avg_away_yellow_cards', 
        'away_roll_5_avg_home_red_cards', 'away_roll_5_avg_away_red_cards', 
        'away_roll_5_avg_ratio_h_a_shots', 'away_roll_5_avg_ratio_a_h_shots',
        'away_cumulative_points'
    ]

    home_team_data = filter_team_data(df_home_stats[home_stats_columns], home_team, 'home_team')
    away_team_data = filter_team_data(df_away_stats[away_stats_columns], away_team, 'away_team')
    common_data = calculate_date_features(date)

    combined_data = pd.concat([home_team_data, away_team_data, common_data], axis=1)

    data_point = combined_data[['home_team', 'away_team', 'day_of_week', 'month', 'day_of_week_sin',
        'day_of_week_cos', 'month_sin', 'month_cos', 'home_roll_3_avg_home_corners',
        'away_roll_3_avg_home_corners', 'home_roll_3_avg_away_corners', 'away_roll_3_avg_away_corners',
        'home_roll_3_avg_home_yellow_cards', 'away_roll_3_avg_home_yellow_cards',
        'home_roll_3_avg_away_yellow_cards', 'away_roll_3_avg_away_yellow_cards',
        'home_roll_3_avg_home_red_cards', 'away_roll_3_avg_home_red_cards',
        'home_roll_3_avg_away_red_cards', 'away_roll_3_avg_away_red_cards',
        'home_roll_3_avg_ratio_h_a_shots', 'away_roll_3_avg_ratio_h_a_shots',
        'home_roll_3_avg_ratio_h_a_fouls', 'away_roll_3_avg_ratio_h_a_fouls',
        'home_roll_3_avg_ratio_a_h_shots', 'away_roll_3_avg_ratio_a_h_shots',
        'home_roll_3_avg_ratio_a_h_fouls', 'away_roll_3_avg_ratio_a_h_fouls',
        'home_roll_5_avg_home_corners', 'away_roll_5_avg_home_corners',
        'home_roll_5_avg_away_corners', 'away_roll_5_avg_away_corners',
        'home_roll_5_avg_home_yellow_cards', 'away_roll_5_avg_home_yellow_cards',
        'home_roll_5_avg_away_yellow_cards', 'away_roll_5_avg_away_yellow_cards',
        'home_roll_5_avg_home_red_cards', 'away_roll_5_avg_home_red_cards',
        'home_roll_5_avg_away_red_cards', 'away_roll_5_avg_away_red_cards',
        'home_roll_5_avg_ratio_h_a_shots', 'away_roll_5_avg_ratio_h_a_shots',
        'home_roll_5_avg_ratio_a_h_shots', 'away_roll_5_avg_ratio_a_h_shots',
        'home_cumulative_points', 'away_cumulative_points']]
    
    return data_point