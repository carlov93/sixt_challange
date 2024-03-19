import pandas as pd
import logging

from src.components.preprocessing.steps import drop_columns, drop_duplicates, type_casting, delete_outliers
from src.util.util import timed, pipeline_logging_config


logger = logging.getLogger("bike_prediction")


class PreprocessingStep:
    def __init__(self, config):
        self.config = config.params
    
    @timed
    @pipeline_logging_config
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """ This function runs the preprocessing step.
        Args:
            df (pd.DataFrame): Raw DataFrame containing the bike rentals and feature columns

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        
        logger.info("Starting Preprocessing")
        logger.info(f"Data before preprocessing: {df.shape[0]}")
        df = drop_columns(df, self.config['columns_to_drop'])
        df = drop_duplicates(df)
        logger.info(f"Data after dropping dublicates: {df.shape[0]}")
        df = type_casting(df)
        logger.info(f"Data before deleting outliers: {df.shape[0]}")
        df = delete_outliers(df, self.config['numerical_columns'])
        logger.info(f"Data after deleting outliers: {df.shape[0]}")
        logger.info(f"Size of the DataFrame after Preprocessing: {df.shape}")
        return df
