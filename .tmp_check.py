import json, pathlib, urllib.request, urllib.error

out = {}
def fetch_text(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode('utf-8')
    except Exception as e:
        return 'ERROR:' + str(e)

def fetch_status(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.getcode()
    except Exception as e:
        return 'ERROR:' + str(e)

out['api'] = fetch_text('http://127.0.0.1:8000/health')
out['streamlit'] = fetch_status('http://127.0.0.1:8501')
out['xgb_exists'] = pathlib.Path('data/models/xgboost_model.pkl').exists()
out['pred_exists'] = pathlib.Path('data/processed/predictions.csv').exists()

print(json.dumps(out))
