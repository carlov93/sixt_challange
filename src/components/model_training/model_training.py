import logging
import pandas as pd

from src.components.model_training.steps import prepare_data_for_training, train_random_forest, evaluate_model
from src.util.util import timed, pipeline_logging_config


logger = logging.getLogger("bike_prediction")


class ModelTrainingStep:
    def __init__(self, config):
        self.config = config.params
    
    @timed
    @pipeline_logging_config
    def run(self, df: pd.DataFrame) -> float:
        """ This function runs the model training step of the pipeline.

        Args:
            df (pd.DataFrame): Data ready for training the model

        Returns:
            float: RMSE score of the trained model
        """
        logger.info("Starting Model Training")
        X_train, X_test, y_train, y_test = prepare_data_for_training(df, self.config['predictor_col'], self.config['test_size'])
        
        model = train_random_forest(
            X_train, 
            y_train, 
            self.config['n_estimators'], 
            self.config['max_depth'], 
            self.config['min_samples_split'], 
            self.config['min_samples_leaf']
        )
        
        score = evaluate_model(model, X_test, y_test)
        
        return score
