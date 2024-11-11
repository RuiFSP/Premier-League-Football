
import pandas as pd
import os
import numpy as np

def fixing_columns_teams_referees(df):
    # Lowercase all columns and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
      
    # Define column renaming dictionary
    columns = {
        'hometeam': 'home_team',
        'awayteam': 'away_team',
        'fthg': 'home_total_goals',
        'ftag': 'away_total_goals',
        'ftr': 'full_time_result',
        'hthg': 'home_half_goals',
        'htag': 'away_half_goals',
        'htr': 'half_time_result',
        'hs': 'home_total_shots',
        'as': 'away_total_shots',
        'hst': 'home_shots_on_target',
        'ast': 'away_shots_on_target',
        'hf': 'home_fouls',
        'af': 'away_fouls',
        'hc': 'home_corners',
        'ac': 'away_corners',
        'hy': 'home_yellow_cards',
        'ay': 'away_yellow_cards',
        'hr': 'home_red_cards',
        'ar': 'away_red_cards',
        'b365h': 'market_home_odds',
        'b365d': 'market_draw_odds',
        'b365a': 'market_away_odds'
    }
    
    # Rename columns based on the dictionary
    df.rename(columns=columns, inplace=True)
    
    # Specific replacement to handle only the apostrophe in team names
    for col in ['home_team', 'away_team']:
        if col in df.columns:
            df[col] = df[col].str.lower().str.replace("'", "")  # Remove apostrophe specifically
            
    # Lowercase referee column if it exists
    if 'referee' in df.columns:
        df['referee'] = df['referee'].str.lower().replace(' ', '_')
    
    return df

def feature_engineering(df):
    # ---------------- Feature Engineering ----------------

    # ---------------- Goal Difference ----------------

    # Goal Difference
    df['goal_difference'] = df['home_total_goals'] - df['away_total_goals']

    # Aggregated Match Statistics
    df['total_shots'] = df['home_total_shots'] + df['away_total_shots']
    df['total_shots_on_target'] = df['home_shots_on_target'] + df['away_shots_on_target']
    df['total_fouls'] = df['home_fouls'] + df['away_fouls']
    df['total_corners'] = df['home_corners'] + df['away_corners']
    df['home_shot_accuracy'] = df['home_shots_on_target'] / df['home_total_shots'].replace(0, 1)
    df['away_shot_accuracy'] = df['away_shots_on_target'] / df['away_total_goals'].replace(0, 1)

    # Time-Based Features
    df['original_date'] = df['date']
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y', errors='coerce')
    df['date'] = df['date'].combine_first(pd.to_datetime(df['original_date'], format='%d/%m/%Y', errors='coerce'))
    df['date'] = df['date'].dt.strftime('%d/%m/%y')
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
    df.drop(columns=['original_date'], inplace=True)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 6.0)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 6.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # Team-Based Features
    df['ratio_h_a_shots'] = df['home_total_shots'] / df['away_total_shots'].replace(0, 1)
    df['ratio_h_a_fouls'] = df['home_fouls'] / df['away_fouls'].replace(0, 1)
    df['ratio_a_h_shots'] = df['away_total_shots'] / df['home_total_shots'].replace(0, 1)
    df['ratio_a_h_fouls'] = df['away_fouls'] / df['home_fouls'].replace(0, 1)

    # Betting Odds-Based Features
    df['implied_home_win_prob'] = 1 / df['market_home_odds']
    df['implied_draw_prob'] = 1 / df['market_draw_odds']
    df['implied_away_win_prob'] = 1 / df['market_away_odds']
    total_prob = df['implied_home_win_prob'] + df['implied_draw_prob'] + df['implied_away_win_prob']
    df['implied_home_win_prob'] /= total_prob
    df['implied_draw_prob'] /= total_prob
    df['implied_away_win_prob'] /= total_prob

    # Rolling Averages
    features = ['home_total_goals', 'away_total_goals', 'home_total_shots', 'away_total_shots', 
          'home_shots_on_target', 'away_shots_on_target', 'home_fouls', 'away_fouls',
          'home_corners', 'away_corners', 'home_yellow_cards', 'away_yellow_cards',
          'home_red_cards', 'away_red_cards', 'home_shot_accuracy', 'away_shot_accuracy',
          'ratio_h_a_shots', 'ratio_h_a_fouls', 'ratio_a_h_shots', 
          'ratio_a_h_fouls', 'goal_difference']
    new_columns = []
    for i in [3, 5]:
        for feature in features:
            home_rolling = (
                df.sort_values(['season', 'home_team', 'date'])
                  .groupby(['season', 'home_team'])[feature]
                  .apply(lambda x: x.shift(1).rolling(window=i).mean())
                  .reset_index(level=[0,1], drop=True)
                  .fillna(0)
            )
            away_rolling = (
                df.sort_values(['season', 'away_team', 'date'])
                  .groupby(['season', 'away_team'])[feature]
                  .apply(lambda x: x.shift(1).rolling(window=i).mean())
                  .reset_index(level=[0,1], drop=True)
                  .fillna(0)
            )
            new_columns.append(home_rolling.rename(f'home_roll_{i}_avg_{feature}'))
            new_columns.append(away_rolling.rename(f'away_roll_{i}_avg_{feature}'))
    df = pd.concat([df] + new_columns, axis=1)

    # Cumulative Points
    home_points = df['full_time_result'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    away_points = df['full_time_result'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
    df = pd.concat([df, home_points.rename('home_points'), away_points.rename('away_points')], axis=1)
    df['home_cumulative_points'] = df.groupby(['season', 'home_team'])['home_points'].transform('cumsum')
    df['away_cumulative_points'] = df.groupby(['season', 'away_team'])['away_points'].transform('cumsum')
    df.drop(columns=['home_points', 'away_points'], inplace=True)
    
    # removing uneccessary columns
    columns_to_drop = ['date', 'home_total_goals',
        'away_total_goals', 'home_half_goals',
        'away_half_goals', 'half_time_result', 'home_total_shots',
        'away_total_shots', 'home_shots_on_target', 'away_shots_on_target',
        'home_fouls', 'away_fouls', 'home_corners', 'away_corners',
        'home_yellow_cards', 'away_yellow_cards', 'home_red_cards',
        'away_red_cards','goal_difference','total_shots',
        'total_shots_on_target','total_fouls','total_corners','home_shot_accuracy',
        'away_shot_accuracy','ratio_h_a_shots','ratio_h_a_fouls',
        'ratio_a_h_shots','ratio_a_h_fouls','referee','market_home_odds',
        'market_draw_odds','market_away_odds']

    df_preparation = df.drop(columns=columns_to_drop)

    return df_preparation

def main():
    data_path = os.path.join('..', 'data', 'processed', 'all_concat_football_data.csv')
    df = pd.read_csv(data_path)
    df = fixing_columns_teams_referees(df)
    df = df.dropna().reset_index(drop=True)
    df = feature_engineering(df)
    
    # Save the prepared data
    df.to_csv(os.path.join('..', 'data', 'processed', 'prepared_football_data.csv'), index=False)
    
    # Save team stats for the current season - 2024/2025
    teams_stats_2024 = df[df['season'] == 2024]
    teams_stats_2024.to_csv(os.path.join('..', 'data', 'processed', 'teams_stats_2024.csv'), index=False)

if __name__ == "__main__":
    main()