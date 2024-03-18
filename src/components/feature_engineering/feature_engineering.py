import pandas as pd

from src.components.feature_engineering.steps import one_hot_encoding
from src.util.util import timed, pipeline_logging_config


class FeatureEngineeringStep:
    def __init__(self, config):
        self.config = config.params
    
    @timed
    @pipeline_logging_config
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = one_hot_encoding(df, self.config['cat_columns'])
        return df
