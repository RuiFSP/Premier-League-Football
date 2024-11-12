# imports
import pandas as pd
import os
import logging
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath):
    """Loads data from a CSV file."""
    return pd.read_csv(filepath)

def create_preprocessor(numerical_features, categorical_features):
    """Returns a preprocessor for scaling numerical features and one-hot encoding categorical features."""
    numerical_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([ 
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

def tune_xgboost(X, y):
    """Tunes XGBoost using RandomizedSearchCV."""
    xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
    param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.05, 0.15),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'gamma': uniform(0, 0.2),
        'lambda': uniform(0, 1),
        'alpha': uniform(0, 1),
    }
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        random_state=42,
        scoring='accuracy'
    )
    random_search.fit(X, y)
    return random_search.best_estimator_, random_search.best_score_

def create_final_pipeline(X_train_selected, y_train, preprocessor):
    """Creates and fits the final pipeline with preprocessing, feature selection, and tuned XGBoost."""
    # Define RFECV with XGBoost as the estimator
    rfecv = RFECV(estimator=XGBClassifier(random_state=42), step=1,
                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                  scoring='accuracy', n_jobs=-1)
    
    # Create the pipeline with preprocessor, feature selection, and model
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', rfecv),
        ('model', XGBClassifier(random_state=42))
    ])
    
    # Fit the pipeline on the training data
    final_pipeline.fit(X_train_selected, y_train)
    
    # Access the best cross-validation score
    best_score = rfecv.cv_results_['mean_test_score'][rfecv.support_].max()
    logger.info(f"Best cross-validated accuracy with tuned XGBoost: {best_score:.4f}")
    
    return final_pipeline

# Main script
if __name__ == "__main__":
    # Load and prepare data
    data = load_data(os.path.join(os.path.dirname(__file__),'..','data', 'processed', 'data_for_model.csv'))
    
    # Encode target variable (convert strings to numeric labels)
    y = data['full_time_result']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Convert string labels to numeric
    
    # Drop the target variable from the feature set
    X = data.drop('full_time_result', axis=1)
    
    # Preprocessor and data split
    numerical_features = X.select_dtypes(exclude='object').columns
    categorical_features = X.select_dtypes(include='object').columns
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Create and evaluate the final model pipeline
    logger.info("Starting feature selection with RFECV...")
    final_pipeline = create_final_pipeline(X_train, y_train, preprocessor)
    
    # Evaluate on test set
    test_score = final_pipeline.score(X_test, y_test)  # Test set score
    logger.info(f"Test set accuracy with final pipeline: {test_score:.4f}")
    
    # Save the final pipeline
    joblib.dump(final_pipeline, os.path.join(os.path.dirname(__file__),'..','models', 'final_model_pipeline.pkl'))
    
    # Save column names for prediction
    column_names = X.columns
    joblib.dump(column_names, os.path.join(os.path.dirname(__file__),'..','models', 'column_names.pkl'))
    
    # Save dtypes for prediction
    dtypes = X.dtypes
    joblib.dump(dtypes, os.path.join(os.path.dirname(__file__),'..','models', 'dtypes.pkl'))
    
    logger.info("Final pipeline saved as 'final_model_pipeline.pkl'")
