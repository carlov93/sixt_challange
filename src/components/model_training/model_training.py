import logging

from src.components.model_training.steps import prepare_data_for_training, train_random_forest, evaluate_model
from src.util.util import timed, pipeline_logging_config


logger = logging.getLogger("bike_prediction")


class ModelTrainingStep:
    def __init__(self, config):
        self.config = config.params
    
    @timed
    @pipeline_logging_config
    def run(self, df):
        """_summary_

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
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
