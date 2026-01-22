import importlib, sys
if 'src.models.train_prophet' in sys.modules:
    del sys.modules['src.models.train_prophet']
from src.models.train_prophet import train
train()
