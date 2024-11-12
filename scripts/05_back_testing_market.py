
import pandas as pd
import numpy as np
import joblib
import os

def load_model(model_path):
    return joblib.load(model_path)

def load_data(data_paths):
    return [pd.read_csv(path) for path in data_paths]

def make_predictions(model, X_test):
    return model.predict(X_test), model.predict_proba(X_test)

def prepare_team_names(X_test, y_test, y_pred, y_pred_proba, data_for_back_testing):
    team_names = X_test.loc[:, ['home_team', 'away_team']].copy()
    team_names['true_results'] = y_test
    team_names['predicted_results'] = y_pred
    team_names[['away_prob', 'draw_prob', 'home_prob']] = pd.DataFrame(y_pred_proba, index=team_names.index)
    team_names[['implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob']] = data_for_back_testing.loc[team_names.index, ['implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob']]
    team_names = team_names[['home_team', 'away_team', 'true_results', 'predicted_results', 'home_prob', 'draw_prob', 'away_prob', 'implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob']]
    return team_names

def encode_result(result):
    if result == 'H':
        return [1, 0, 0]
    elif result == 'D':
        return [0, 1, 0]
    elif result == 'A':
        return [0, 0, 1]

def decode_result(result):
    if result == 2:
        return 'H'
    elif result == 1:
        return 'D'
    elif result == 0:
        return 'A'

def calculate_brier_score(predictions, true_outcome):
    return np.mean((predictions - true_outcome) ** 2)

def calculate_brier_scores(team_names):
    team_names['true_predictions_brier'] = team_names['true_results'].apply(encode_result)
    team_names['brier_score_market'] = team_names.apply(lambda row: calculate_brier_score(np.array([row['home_prob'], row['draw_prob'], row['away_prob']]), row['true_predictions_brier']), axis=1)
    team_names['brier_score_model'] = team_names.apply(lambda row: calculate_brier_score(np.array([row['implied_home_win_prob'], row['implied_draw_prob'], row['implied_away_win_prob']]), row['true_predictions_brier']), axis=1)
    return team_names

def main():
    model_path = os.path.join(os.path.dirname(__file__),'..', 'models', 'final_model_pipeline.pkl')
    data_paths = [
        os.path.join(os.path.dirname(__file__),'..', 'data', 'processed', 'X_test.csv'),
        os.path.join(os.path.dirname(__file__),'..', 'data', 'processed', 'y_test.csv'),
        os.path.join(os.path.dirname(__file__),'..', 'data', 'processed', 'data_for_back_testing.csv')
    ]

    model = load_model(model_path)
    X_test, y_test, data_for_back_testing = load_data(data_paths)
    y_pred, y_pred_proba = make_predictions(model, X_test)
    
    team_names = prepare_team_names(X_test, y_test, y_pred, y_pred_proba, data_for_back_testing)
    team_names['predicted_results'] = team_names['predicted_results'].apply(decode_result)
    team_names = calculate_brier_scores(team_names)
    
    average_brier_score_market = team_names['brier_score_market'].mean()
    average_brier_score_model = team_names['brier_score_model'].mean()
    
    print(f"Average Brier Score Market: {average_brier_score_market}")
    print(f"Average Brier Score Model: {average_brier_score_model}")
    print(team_names[['home_team', 'away_team', 'true_results', 'predicted_results', 'brier_score_model', 'brier_score_market']])

if __name__ == "__main__":
    main()