
import os, glob, cv2, torch, numpy as np
from .dataset import Colonies
from ..models.hat_unet import HATUNet
from ..post.refine import refine
from ..exp.eval import dice_iou, count_mae
from torch.utils.data import DataLoader

def run_test(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Colonies(cfg["test_im"], cfg["test_mk"], cfg["size"], train=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = HATUNet(base=cfg["base"]).to(device)
    model.load_state_dict(torch.load(cfg["weights"], map_location=device))
    model.eval()

    os.makedirs("runs/preds", exist_ok=True)

    dices, ious, maes = [], [], []
    with torch.no_grad():
        for im, m, ip in dl:
            im = im.to(device)
            logits = model(im)
            prob = torch.sigmoid(logits)[0,0].cpu().numpy()
            seg  = refine(prob, th=0.5).astype(np.uint8)

            name = os.path.splitext(os.path.basename(ip[0]))[0]
            cv2.imwrite(f"runs/preds/{name}_prob.png", (prob*255).astype(np.uint8))
            cv2.imwrite(f"runs/preds/{name}_seg.png", seg*255)

            gt = (m[0,0].numpy() > 0.5).astype(np.uint8)
            d,i = dice_iou(seg, gt)
            dices.append(d); ious.append(i); maes.append(count_mae(seg, gt))

    return float(np.mean(dices)), float(np.mean(ious)), float(np.mean(maes))
