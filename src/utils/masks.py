
import os, json, cv2
import numpy as np

def yolo_to_mask(txt_path: str, H: int, W: int, ellipse: bool = True) -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    if not os.path.exists(txt_path):
        return mask
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = parts[:5]
            cx, cy, bw, bh = float(cx)*W, float(cy)*H, float(bw)*W, float(bh)*H
            if ellipse:
                cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                            (max(1,int(round(bw/2))), max(1,int(round(bh/2)))),
                            0, 0, 360, 255, -1)
            else:
                x1 = int(round(cx - bw/2)); y1 = int(round(cy - bh/2))
                x2 = int(round(cx + bw/2)); y2 = int(round(cy + bh/2))
                cv2.rectangle(mask, (max(0,x1),max(0,y1)),
                              (min(W-1,x2),min(H-1,y2)), 255, -1)
    return mask

def json_to_mask(json_path: str, H: int, W: int, ellipse: bool = True) -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    if not os.path.exists(json_path):
        return mask
    meta = json.load(open(json_path))
    for obj in meta.get("labels", []):
        x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
        cx, cy = x + w//2, y + h//2
        if ellipse:
            cv2.ellipse(mask, (cx, cy), (w//2, h//2), 0, 0, 360, 255, -1)
        else:
            cv2.rectangle(mask, (x,y), (x+w,y+h), 255, -1)
    return mask

def save_overlay(base_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.3, channel: int = 1):
    color = np.zeros_like(base_bgr)
    color[:,:,channel] = (mask > 0).astype(np.uint8) * 255
    return cv2.addWeighted(base_bgr, 1.0, color, alpha, 0)
