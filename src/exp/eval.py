
import numpy as np
from skimage.measure import label

def dice_iou(pred: np.ndarray, gt: np.ndarray, eps=1e-6):
    predb = pred.astype(bool)
    gtb   = gt.astype(bool)
    inter = (predb & gtb).sum()
    d = 2*inter / (predb.sum() + gtb.sum() + eps)
    i = inter / ((predb | gtb).sum() + eps)
    return float(d), float(i)

def count_mae(pred: np.ndarray, gt: np.ndarray):
    np_pred = label(pred).max()
    np_gt   = label(gt).max()
    return abs(int(np_pred) - int(np_gt))
