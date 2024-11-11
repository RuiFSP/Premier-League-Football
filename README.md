# Midterm Project: Premier League Football Prediction

## Table of Contents

- [Midterm Project: Premier League Football Prediction](#midterm-project-premier-league-football-prediction)
  - [Table of Contents](#table-of-contents)
  - [Problem Description](#problem-description)
  - [Data](#data)
    - [Raw Data](#raw-data)
    - [Enriched Data](#enriched-data)
  - [Using the Project Locally](#using-the-project-locally)
    - [Running Docker](#running-docker)
    - [Testing the Model](#testing-the-model)
    - [Usage](#usage)
    - [Model Training](#model-training)
    - [Prediction](#prediction)
    - [Back Testing](#back-testing)
    - [Contributing](#contributing)

## Problem Description

As a long-time enthusiast of sports analytics and the mathematics of gambling, I have always enjoyed building models to predict football match outcomes. Although beating the market in the long run is challenging due to its dynamic nature, this project aims to explore which features are most important in predicting match results—whether it’s a home win, draw, or away win.

## Data

The raw data for this project is sourced from [Football Data](https://www.football-data.co.uk/data.php). The focus is exclusively on the Premier League, covering seasons from 2005/2006 to 2024/2025. The raw data files can be found [here](https://github.com/RuiFSP/mlzoomcamp2024-midterm-project/tree/main/data/raw_data).

### Raw Data

The raw data includes the following fields:

- Date: Match Date (dd/mm/yy)
- Time: Time of match kick-off
- HomeTeam: Home Team
- AwayTeam: Away Team
- FTHG: Full Time Home Team Goals
- FTAG: Full Time Away Team Goals
- FTR: Full Time Result (H=Home Win, D=Draw, A=Away Win)
- HTHG: Half Time Home Team Goals
- HTAG: Half Time Away Team Goals
- HTR: Half Time Result (H=Home Win, D=Draw, A=Away Win)
- Referee: Match Referee
- HS: Home Team Shots
- AS: Away Team Shots
- HST: Home Team Shots on Target
- AST: Away Team Shots on Target
- HHW: Home Team Hit Woodwork
- AHW: Away Team Hit Woodwork
- HC: Home Team Corners
- AC: Away Team Corners
- HF: Home Team Fouls Committed
- AF: Away Team Fouls Committed
- HFKC: Home Team Free Kicks Conceded
- AFKC: Away Team Free Kicks Conceded
- HO: Home Team Offsides
- AO: Away Team Offsides
- HY: Home Team Yellow Cards
- AY: Away Team Yellow Cards
- HR: Home Team Red Cards
- AR: Away Team Red Cards
- B365H: Bet365 home win odds
- B365D: Bet365 draw odds
- B365A: Bet365 away win odds
- Season: Data season

### Enriched Data

- Goal Differences
  - Calculated as the difference between home team goals and away team goals.
- Aggregated Match Statistics
  - Includes total shots, shots on target, fouls, and corners for both teams.
  - Shot accuracy for both home and away teams.
- Time-Based Features
  - Date features such as day of the week and month.
  - Sinusoidal transformations for cyclical features like day of the week and month.
- Team-Based Features
  - Ratios of shots and fouls between home and away teams.
- Betting Odds-Based Features
  - Implied probabilities of home win, draw, and away win based on betting odds.
- Rolling Averages for Features
  - Rolling averages for various match statistics over a specified number of previous games.
- Cumulative Points
  - Cumulative points for home and away teams throughout the season.

## Using the Project Locally

### Running Docker

Build the Docker image:

```bash
    docker build -t midterm-mlzoomcamp-project .
```

Run the Docker container:

```bash
    docker run -it --rm -p 9696:9696 midterm-mlzoomcamp-project
```

### Testing the Model

Open a new terminal and run the test script:

```python
    python tests/test_predict.py    
```

### Usage

To use the prediction service, send a POST request to the /predict endpoint with the following JSON payload:

```bash
curl -X POST http://127.0.0.1:9696/predict \
     -H "Content-Type: application/json" \
     -d '{
           "home_team": "arsenal",
           "away_team": "liverpool",
           "date": "2024-12-16"
         }'
```


### Model Training

The model training process is implemented in the 'train.py' script. It includes data loading, preprocessing, feature selection, and model training using XGBoost. The final model pipeline is saved to the 'models' directory.

### Prediction
The prediction process is implemented in the 'predict.py' script. It includes loading the trained model, validating input data, fetching team data, and making predictions.

### Back Testing
Back testing is implemented in the 'notebooks/05-back_testing.ipynb' notebook. It includes loading test data, making predictions, and evaluating the model's performance.

### Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License.

Feel free to customize this README file further to suit your project's specific needs.