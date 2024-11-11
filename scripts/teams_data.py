import numpy as np
import pandas as pd
import os

def get_teams_data(home_team, away_team, date):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'teams_stats_2024.csv'))
    
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
    
    
    # extract the last values for all teams
    df_home_stats = df.groupby('home_team').tail(1)[features].sort_values(by='home_team')

    # extract the last values for all teams
    df_away_stats = df.groupby('away_team').tail(1)[features].sort_values(by='away_team')

    # Identify columns related to home team stats and away team stats
    home_stats_columns = [
        'home_team',  # Keep 'home_team' for reference
        'home_roll_3_avg_home_corners', 'home_roll_3_avg_away_corners', 
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
        'away_team',  # Keep 'away_team' for reference
        'away_roll_3_avg_home_corners', 'away_roll_3_avg_away_corners', 
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

    # Select subsets for home_stats and away_stats
    home_stats = df_home_stats[home_stats_columns].copy()
    away_stats = df_away_stats[away_stats_columns].copy()
    
    print(home_stats)
    
    # Filter rows for specific teams in home_stats and away_stats
    home_team_data = home_stats[home_stats['home_team'] == home_team]
    away_team_data = away_stats[away_stats['away_team'] == away_team]
    
    print(home_team_data)

    # get day_of_week and month from the date
    date = pd.to_datetime(date)
    day_of_week = date.dayofweek
    month = date.month
    
    
    # Calculate common column values for the specific game day and month
    #day_of_week = 4  # Friday
    #month = 8        # August
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
    
    return data_point


# if __name__ == "__main__":
#     home_team = "arsenal"
#     away_team = "brentford"
#     date = "2024-08-16"
    
#     data_point = get_teams_data(home_team, away_team, date)
#     print(data_point)