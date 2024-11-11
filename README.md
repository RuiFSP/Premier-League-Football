# Midterm Project: Premier League Football Prediction

## Table of Contents
- [Midterm Project: Premier League Football Prediction](#midterm-project-premier-league-football-prediction)
  - [Table of Contents](#table-of-contents)
  - [Problem Description](#problem-description)
  - [Running Docker](#running-docker)
  - [Testing the Model](#testing-the-model)

## Problem Description

I've always been a big fan of sports analytics and the mathematics of gambling. One of my favorite hobbies in the past was building Excel models to predict outcomes in football matches and seeing if I could beat the market. Of course, it was impossible in the long run because markets are dynamic and adjust quickly to current news, like player injuries, coaching changes, etc. Still, it’s a fun experiment, and books like Moneyball, Soccermatics, and The Numbers Game have always sparked my curiosity in model-building. 

The goal of this project is to understand which features might be most important in building a model that can predict football match outcomes—whether it’s a home win, draw, or away win.

## Running Docker

Build the Docker image:
```bash
    docker build -t zoomcap-midterm-project .
```

Run the Docker container:
```bash
    docker run -it --rm -p 9696:9696 midterm-mlzoomcamp-project
```

## Testing the Model

Open a new terminal and run the test script:
```python
    python tests/test_predict.py    
```

You can also test the model with curl:
```bash
curl -X POST http://127.0.0.1:9696/predict \
     -H "Content-Type: application/json" \
     -d '{
           "home_team": "arsenal",
           "away_team": "liverpool",
           "date": "2024-12-16"
         }'
```
