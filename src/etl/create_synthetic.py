# src/etl/create_synthetic.py
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path('data/raw/sales_synth.csv')

def generate(n_sku=30, n_store=3, start='2023-01-01', end='2025-06-30', seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq='D')
    sku_ids = [f'SKU{str(i).zfill(3)}' for i in range(1, n_sku+1)]
    stores = [f'S{str(i).zfill(2)}' for i in range(1, n_store+1)]
    rows = []
    for sku in sku_ids:
        base = rng.integers(15, 60)
        trend = rng.uniform(-0.0005, 0.001)
        price0 = rng.integers(150, 600)
        cat = f'C{rng.integers(1,6)}'
        for store in stores:
            store_k = rng.uniform(0.8, 1.2)
            for t, d in enumerate(dates):
                week_season = 1.0 + 0.15*np.sin(2*np.pi*(d.dayofweek)/7)
                year_season = 1.0 + 0.25*np.sin(2*np.pi*(d.timetuple().tm_yday)/365)
                promo = 1 if rng.random() < 0.05 else 0
                price = float(price0 * (0.9 if promo else 1.0) * rng.uniform(0.95, 1.05))
                price_elast = -0.35
                price_eff = (price/price0) ** price_elast
                trend_eff = 1.0 + trend*t
                noise = rng.normal(0, 2)
                units = max(0, int(base * week_season * year_season * trend_eff * price_eff * store_k + promo*8 + noise))
                rows.append((d.date(), sku, store, cat, units, round(price,2), promo))
    df = pd.DataFrame(rows, columns=['date','sku_id','store_id','category','units','price','promo_flag'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return OUT, len(df)

if __name__ == '__main__':
    path, n = generate()
    print(f'Generated {n} rows -> {path}')
