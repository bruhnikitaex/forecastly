import pandas as pd
import numpy as np

def abc_xyz(df: pd.DataFrame):
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
