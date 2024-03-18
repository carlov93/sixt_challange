from typing import TypedDict, NamedTuple, List


# PREPROCESSING STEP
preprocessing_params = TypedDict(
    "preprocessing_params", 
    {
        'columns_to_drop': List[str],
        'numerical_columns': List[str],
    })


# FEATURE ENGINEERING STEP
feature_engineering_params = TypedDict(
    "feature_engineering_params", 
    {
        'cat_columns': List[str],
    }
)


# MODEL TRAINING STEP
model_training_params = TypedDict(
    "model_training_params", 
    {
        'predictor_col': List[str],
        'test_size': float,
        'n_estimators': [List[int]],
        'max_depth': List[int],
        'min_samples_split': List[int],
        'min_samples_leaf': List[int]
    }
)
