import pandas as pd
import io
import logging

from src.components.preprocessing.preprocessing import PreprocessingStep
from src.components.feature_engineering.feature_engineering import FeatureEngineeringStep
from src.components.model_training.model_training import ModelTrainingStep
from config.config import PreprocessingConfig, FeatureEngineeringConfig, ModelTrainingConfig


logger = logging.getLogger("bike_prediction")
# create console handler, set level and add to logger
ch = logging.StreamHandler()
logger.addHandler(ch)
log_stringio_obj = io.StringIO()
# create file handler, set level and add to logger
s3_handler = logging.StreamHandler(log_stringio_obj)
logger.addHandler(s3_handler)


def run_pipeline(df: pd.DataFrame):
    # Load Configs
    preprocessing_config = PreprocessingConfig()
    feature_engineering_config = FeatureEngineeringConfig()
    model_training_config = ModelTrainingConfig()
    
    # Initial Pipeline Steps
    prepocessing_step = PreprocessingStep(preprocessing_config)
    feature_engineering_step = FeatureEngineeringStep(feature_engineering_config)
    model_training_step = ModelTrainingStep(model_training_config)
    
    # Run Pipeline
    df = prepocessing_step.run(df)
    df = feature_engineering_step.run(df)
    model_training_step.run(df)


if __name__ == "__main__":
    # read raw data
    df_all = pd.read_csv('./data/day.csv')

    # split dataset
    df_last30 = df_all.tail(30)
    df = df_all.iloc[:-30, :]
    run_pipeline(df)
