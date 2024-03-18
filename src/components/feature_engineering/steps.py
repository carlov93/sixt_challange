from typing import List

import pandas as pd



def one_hot_encoding(df: pd.DataFrame, cat_columns: List) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = pd.get_dummies(df, columns=cat_columns, dtype=int)
    
    return df
