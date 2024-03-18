from __future__ import annotations
from os import getenv
from dataclasses import dataclass, field, asdict
from typing import Dict
from abc import ABC

from config.types import *


class Config(ABC):
    def _get_config(self: Config) -> Config:
        return self

    def get_config_as_dict(self: Config) -> Dict[str, str]:
        return asdict(self._get_config())
    
    

@dataclass
class PreprocessingConfig(Config):
    name = "preprocessing"
    params: preprocessing_params = field(
        default_factory=lambda: {
        }
    )


@dataclass
class FeatureEngineeringConfig(Config):
    name = "feature_engineering"
    params: feature_engineering_params = field(
        default_factory=lambda: {
        }
    )


@dataclass
class ModelTrainingConfig(Config):
    name = "model_training"
    params: model_training_params = field(
        default_factory=lambda: {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    )
