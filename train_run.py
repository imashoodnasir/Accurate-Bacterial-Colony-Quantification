
import yaml, json
from src.exp.train import run_train
cfg = yaml.safe_load(open("configs/sample.yaml"))
print("Config:", json.dumps(cfg, indent=2))
run_train(cfg)
