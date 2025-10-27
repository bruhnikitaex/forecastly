import pandas as pd

def ensure_datetime(df: pd.DataFrame, col: str = 'date') -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col])
    return df.sort_values(col)
