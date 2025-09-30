
import yaml
from src.exp.test import run_test
cfg = yaml.safe_load(open("configs/sample.yaml"))
dice, iou, mae = run_test(cfg)
print(f"Test Dice={dice:.4f}, IoU={iou:.4f}, MAE={mae:.2f}")
