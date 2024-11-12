# Midterm Project: Premier League Football Prediction

## Overview

This project aims to predict the outcomes of Premier League football matches using machine learning models. It explores various features to determine their importance in predicting match results—whether it’s a home win, draw, or away win.

## Table of Contents

- [Midterm Project: Premier League Football Prediction](#midterm-project-premier-league-football-prediction)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Problem Description](#problem-description)
  - [Data](#data)
  - [Notebooks](#notebooks)
    - [01\_data\_gathering](#01_data_gathering)
    - [02-data\_preparation](#02-data_preparation)
    - [03\_eda](#03_eda)
    - [04\_train\_model](#04_train_model)
    - [05\_back\_testing](#05_back_testing)
  - [Using the Project Locally](#using-the-project-locally)
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Installing Dependencies](#installing-dependencies)
      - [Navigate to Your Project Directory](#navigate-to-your-project-directory)
      - [Install the Project Dependencies](#install-the-project-dependencies)
      - [Activate the Virtual Environment](#activate-the-virtual-environment)
    - [Running Docker](#running-docker)
    - [Testing the Model](#testing-the-model)
    - [Usage](#usage)
    - [Running the Streamlit App (Bonus)](#running-the-streamlit-app-bonus)
    - [Contributing](#contributing)
  - [License](#license)

## Problem Description

Predicting the outcomes of football matches has always been a challenging yet fascinating task for sports analysts and enthusiasts. This project focuses on the Premier League, aiming to build robust machine learning models to forecast match results—whether it’s a home win, draw, or away win. By leveraging historical match data, team statistics, and betting odds, the project seeks to identify key features that influence match outcomes. Despite the inherent unpredictability and dynamic nature of sports betting markets, this project aspires to provide valuable insights and potentially profitable predictions.

## Data

The raw data for this project is sourced from [Football Data](https://www.football-data.co.uk/data.php). The focus is exclusively on the Premier League, covering seasons from 2005/2006 to 2024/2025. The raw data files can be found [here](https://github.com/RuiFSP/mlzoomcamp2024-midterm-project/tree/main/data/raw_data).

## Notebooks

### 01_data_gathering

![Data Gathering](images/data_gathering.png)

The '01-data_gathering.ipynb' notebook is responsible for gathering and processing football data from various seasons. Below are the key steps performed in the notebook:

1. Defining URLs and Paths: It defines the base URL for data sources and the local path for saving the data.
2. Data Scraping: Loops through the specified seasons to download CSV files containing football match data.
3. Data Checking: Verifies if all required columns are present in the downloaded data.
4. Data Concatenation: Reads the CSV files, selects specific columns, and concatenates them into a single DataFrame.
5. Saving Processed Data: The final concatenated DataFrame is saved as a CSV file for further analysis.

### 02-data_preparation

![Data Preparation](images/data_preparation.png)

1. Data Cleaning:
   - Fix column names to be lowercase and replace spaces with underscores.
   - Rename specific columns for better readability.
   - Handle missing values by removing rows with NaN values.
   - Check for duplicates and ensure data integrity.
2. Feature Engineering:
   - Create new features such as goal difference, total shots, shot accuracy, and time-based features.
   - Calculate rolling averages for various statistics over 3 and 5 game windows.
   - Compute cumulative points for home and away teams.
   - Normalize betting odds to implied probabilities.
3. Saving Processed Data:
   - Save the processed data for the current season (2024/2025) to a CSV file.
   - Save the final prepared dataset to a CSV file.

### 03_eda

![Exploratory Data Analysis](images/eda.png)

The '03-eda.ipynb' notebook is dedicated to Exploratory Data Analysis (EDA). It includes the following key steps:

1. Data Checking: Check data types, missing values, unique values, duplicates, and outliers.
2. Correlation Analysis: Identify highly correlated features using a correlation matrix.
3. Variance Inflation Factor (VIF): Calculate VIF to check for multicollinearity and remove features with high VIF values.
4. Cluster Maps: Plot clustered heatmaps to visualize feature correlations.
5. Target Distribution: Visualize the distribution of the target variable.
6. Saving Data: Save the cleaned and processed data for modeling and backtesting.

### 04_train_model

![Model Training](images/train_model.png)

This notebook covers the process of training a machine learning model for predicting football match outcomes. It includes data preprocessing, feature selection using Recursive Feature Elimination with Cross-Validation (RFECV), and model evaluation using RandomForest and XGBoost classifiers. The notebook also demonstrates hyperparameter tuning to reduce overfitting and finalizes the best model using a pipeline. The final model is saved for future predictions.

### 05_back_testing

![Back Testing](images/back_testing.png)

This notebook is dedicated to backtesting the trained model's performance on historical data. It includes loading the test data, making predictions, and evaluating the model's performance using metrics such as the Brier score. The notebook compares the model's predictions against market odds to assess its predictive power.

## Using the Project Locally

### Prerequisites

- Python 3.8 or higher
- Docker
- Pipenv

### Clone the Repository

Use `git clone` to copy the repository to your local machine and navigate into the project directory.

```bash
  git clone <repository-url>
  cd repository
```

Replace <repository-url> with the actual URL of the repository (for example, from GitHub, GitLab, etc.)

```bash
  git clone https://github.com/username/repository.git
  cd repository
```

### Installing Dependencies

#### Navigate to Your Project Directory

First, open a terminal and change to the directory where your `Pipfile` and `Pipfile.lock` are located.

```bash
  cd /path/to/your/project
```

#### Install the Project Dependencies

In the project directory, use `pipenv install` to create the virtual environment and install all dependencies specified in the `Pipfile.lock`.

```bash
  pipenv install
```

This command will:

- Create a virtual environment if one doesn’t already exist.
- Install the dependencies exactly as specified in the `Pipfile.lock`.

#### Activate the Virtual Environment

To activate the virtual environment, use:

```bash
  pipenv shell
```

Now you're in an isolated environment where the dependencies specified in the `Pipfile.lock` are installed.

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

### Running the Streamlit App (Bonus)

To run the Streamlit app locally, follow these steps:

1. Ensure you have all dependencies installed and the virtual environment activated as described in the [Installing Dependencies](#installing-dependencies) section.

2. Navigate to the project directory where `app.py` is located.

3. Run the Streamlit app using the following command:

```bash
    streamlit run app.py
```


### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

Feel free to customize this README file further to suit your project's specific needs.