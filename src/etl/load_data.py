import pandas as pd
from pathlib import Path
from src.utils.config import PATHS
from src.utils.logger import logger

def load_sales(path: str | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else Path(PATHS['data']['raw'])
    logger.info(f'Loading raw data from {csv_path}')
    df = pd.read_csv(csv_path)
    return df

if __name__ == '__main__':
    df = load_sales()
    print(df.head())
