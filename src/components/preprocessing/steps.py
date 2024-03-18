from typing import List

import pandas as pd


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.drop(columns, axis=1)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def type_casting(df: pd.DataFrame) -> pd.DataFrame:
    # type casting binary columns to category
    df['holiday'] = df['holiday'].astype('category')
    df['workingday'] = df['workingday'].astype('category')   
    
    # type casting other columns
    df["weekday"] = df["weekday"].replace({0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'})
    df["weathersit"] = df["weathersit"].replace({1: 'Clear', 2: 'Misty', 3: 'Light_rainsnow', 4: 'Heavy_rainsnow'})
    df["mnth"] = df["mnth"].replace({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
    df['season'] = df['season'].replace({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
    return df


def delete_outliers(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    outliers = {}
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        df = df.drop(outliers[column].index)
         
    return df
