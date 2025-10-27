import argparse, joblib, pandas as pd, numpy as np
from pathlib import Path
from src.utils.config import PATHS
from src.utils.logger import logger

def main(horizon=14):
    models_dir = Path(PATHS['data']['models_dir'])
    out_csv = Path('data/processed/predictions.csv')
    preds = []
    for name in ['prophet_model.pkl','lgbm_model.pkl']:
        mp = models_dir / name
        if not mp.exists():
            logger.warning(f'Model not found: {mp}')
            continue
        if name.startswith('prophet'):
            m = joblib.load(mp)
            future = m.make_future_dataframe(periods=horizon, freq='D')
            fcst = m.predict(future).tail(horizon)
            preds.append(pd.DataFrame({
                'date': fcst['ds'],
                'model': 'prophet',
                'yhat': fcst['yhat'],
                'yhat_lower': fcst['yhat_lower'],
                'yhat_upper': fcst['yhat_upper']
            }))
        else:
            m = joblib.load(mp)
            import datetime as dt
            base = pd.Timestamp.today().normalize()
            rows = []
            for i in range(1, horizon+1):
                d = base + pd.Timedelta(days=i)
                rows.append({'dow': d.dayofweek, 'week': int(d.isocalendar().week), 'month': d.month,
                             'units_lag_1': 0.0, 'units_lag_7': 0.0})
            X = pd.DataFrame(rows)
            y = m.predict(X)
            preds.append(pd.DataFrame({'date':[r for r in pd.date_range(base+pd.Timedelta(days=1), periods=horizon)],
                                       'model':'lgbm','yhat':y,'yhat_lower':np.nan,'yhat_upper':np.nan}))
    if preds:
        out = pd.concat(preds, ignore_index=True)
        out.to_csv(out_csv, index=False)
        logger.info(f'Predictions saved to {out_csv}')
    else:
        logger.warning('No predictions produced.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=14)
    args = ap.parse_args()
    main(args.horizon)
