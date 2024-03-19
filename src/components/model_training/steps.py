import pandas as pd
import logging
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV


logger = logging.getLogger("bike_prediction")


def prepare_data_for_training(
    df: pd.DataFrame, 
    columns: List[str], 
    test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    #creating dimensions for Modelling
    x_data = df.drop(columns, axis=1)
    y_data = df['cnt']
    
    # First, split the data into training (80%) and the remaining (20%)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)
    logger.info(f"shape of X_train: {X_train.shape}")
    logger.info(f"shape of y_train: {y_train.shape}")
    logger.info(f"shape of X_test: {X_test.shape}")
    logger.info(f"shape of y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test
    

def train_random_forest(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    n_estimators: List[int], 
    max_depth: List[int], 
    min_samples_split: List[int], 
    min_samples_leaf: List[int]
    ) -> RandomForestRegressor:

    # Create a random forest regressor
    rf = RandomForestRegressor()

    # Create the grid search object
    param_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Train a random forest model with the best parameters
    best_rf = RandomForestRegressor(**best_params)
    logger.info(f"The best parameters are: {best_params}")
    best_rf.fit(X_train, y_train)
    
    return best_rf

def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
    # Predict the target variable using the trained model
    y_pred = model.predict(X_test)

    # Calculate the RMSE
    rmse = root_mean_squared_error(y_test, y_pred)
    logging.info(f"The RMSE is: {round(rmse, 2)}")
    
    return rmse
