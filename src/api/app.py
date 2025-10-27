from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
from pathlib import Path
from datetime import date, timedelta

app = FastAPI(title='Sales Forecasting API')

class PredictRequest(BaseModel):
    horizon: int = 14
    model: Optional[str] = None  # 'prophet' | 'lgbm' | None (оба)

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/predict')
def predict(req: PredictRequest):
    models_dir = Path('data/models')
    out = []
    names = ['prophet_model.pkl','lgbm_model.pkl'] if req.model is None else [f'{req.model}_model.pkl']
    for name in names:
        mp = models_dir / name
        if not mp.exists():
            out.append({'model': name.replace('_model.pkl',''), 'error': 'model_not_found'})
            continue
        if name.startswith('prophet'):
            m = joblib.load(mp)
            future = m.make_future_dataframe(periods=req.horizon, freq='D')
            fcst = m.predict(future).tail(req.horizon)
            out.append({'model':'prophet','dates':[str(x.date()) for x in fcst['ds']],
                        'yhat': [round(float(v),2) for v in fcst['yhat']]})
        else:
            m = joblib.load(mp)
            base = date.today()
            # Для простоты демо используем псевдопрогноз (без фичей)
            import random
            rnd = random.Random(42)
            y = [float(rnd.randint(20,60)) for _ in range(req.horizon)]
            out.append({'model':'lgbm','dates':[str((base+timedelta(days=i)).isoformat()) for i in range(1, req.horizon+1)],
                        'yhat': y})
    return {'predictions': out}
