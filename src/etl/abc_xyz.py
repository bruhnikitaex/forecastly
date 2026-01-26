"""
Модуль ABC-XYZ анализа для классификации SKU.

ABC-анализ группирует товары по вкладу в оборот:
- A: 80% оборота (наиболее важные)
- B: 15% оборота (средней важности)
- C: 5% оборота (наименее важные)

XYZ-анализ группирует по стабильности спроса (коэффициент вариации):
- X: CV < 0.1 (стабильный спрос)
- Y: 0.1 <= CV < 0.25 (умеренные колебания)
- Z: CV >= 0.25 (нестабильный спрос)
"""

import pandas as pd
import numpy as np


def abc_xyz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет ABC-XYZ анализ по SKU.

    Args:
        df: DataFrame с колонками sku_id и units.

    Returns:
        DataFrame с колонками: sku_id, units_sum, ABC, cv, XYZ, ABCXYZ.
    """
    sums = df.groupby('sku_id', as_index=False)['units'].sum().rename(columns={'units':'units_sum'})
    totals = sums['units_sum'].sum()
    sums = sums.sort_values('units_sum', ascending=False)
    sums['cum_share'] = sums['units_sum'].cumsum()/totals

    def abc_mark(x):
        if x <= 0.8: return 'A'
        elif x <= 0.95: return 'B'
        return 'C'
    sums['ABC'] = sums['cum_share'].apply(abc_mark)

    cv = df.groupby('sku_id')['units'].agg(['mean','std']).reset_index()
    cv['cv'] = (cv['std']/(cv['mean'].replace(0,np.nan))).fillna(np.inf)
    def xyz_mark(v):
        if v < 0.1: return 'X'
        elif v < 0.25: return 'Y'
        return 'Z'
    cv['XYZ'] = cv['cv'].apply(xyz_mark)

    out = pd.merge(sums[['sku_id','units_sum','ABC']], cv[['sku_id','cv','XYZ']], on='sku_id', how='left')
    out['ABCXYZ'] = out['ABC'] + out['XYZ']
    return out.sort_values(['ABC','XYZ','units_sum'], ascending=[True,True,False])
