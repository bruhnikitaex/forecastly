from pathlib import Path
from src.utils.logger import logger
from src.etl.load_data import load_sales
from src.etl.clean_data import clean_sales
from src.etl.feature_builder import build_features
from src.etl.abc_xyz import abc_xyz
from src.utils.config import PATHS


def main(input_path: str | None = None):
    df = load_sales(input_path)
    df = clean_sales(df)
    feats = build_features(df)
    processed_dir = Path(PATHS['data']['processed'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_p = processed_dir / 'features.parquet'
    feats.to_parquet(out_p, index=False)
    abc = abc_xyz(df)
    abc_csv = processed_dir / 'abcxyz.csv'
    abc.to_csv(abc_csv, index=False)
    logger.info(f'Processed saved to {out_p}, ABCXYZ to {abc_csv}')
    return str(out_p), str(abc_csv)


if __name__ == '__main__':
    main()
