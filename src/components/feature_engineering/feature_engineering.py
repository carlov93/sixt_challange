import pandas as pd
import logging

from src.components.feature_engineering.steps import one_hot_encoding
from src.util.util import timed, pipeline_logging_config


logger = logging.getLogger("bike_prediction")


class FeatureEngineeringStep:
    def __init__(self, config):
        self.config = config.params
    
    @timed
    @pipeline_logging_config
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """ This function runs the feature engineering steps.

        Args:
            df (pd.DataFrame): Prepared DataFrame

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded categorical columns
        """
        logger.info("Starting Feature Engineering")
        df = one_hot_encoding(df, self.config['cat_columns'])
        logger.info(f"Size of the DataFrame after Feature Engineering: {df.shape}")
        return df
