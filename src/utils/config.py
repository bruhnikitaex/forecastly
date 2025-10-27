import yaml
from pathlib import Path

def load_config(p: str):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

ROOT = Path(__file__).resolve().parents[2]
PATHS = load_config(str(ROOT / 'configs' / 'paths.yaml'))
MODEL_CFG = load_config(str(ROOT / 'configs' / 'model.yaml'))
