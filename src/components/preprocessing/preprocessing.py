import pandas as pd

from src.components.preprocessing.steps import drop_columns, drop_duplicates, type_casting, delete_outliers
from src.util.util import timed, pipeline_logging_config


class PreprocessingStep:
    def __init__(self, config):
        self.config = config.preprocessing_params
    
    @timed
    @pipeline_logging_config
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = drop_columns(df, self.config.columns_to_drop)
        df = drop_duplicates(df)
        df = type_casting(df)
        df = delete_outliers(df, self.config.numerical_columns)
        
        return df
