
import os, numpy as np, torch, torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import Colonies
from ..models.hat_unet import HATUNet
from ..losses.dice_bce import DiceBCE

def set_seeds(seed=1337):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def run_train(cfg):
    set_seeds(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(Colonies(cfg["train_im"], cfg["train_mk"], cfg["size"], True),
                    batch_size=cfg["bs"], shuffle=True, num_workers=2)
    vl = DataLoader(Colonies(cfg["val_im"], cfg["val_mk"], cfg["size"], False),
                    batch_size=cfg["bs"], shuffle=False, num_workers=2)

    model = HATUNet(base=cfg["base"]).to(device)
    opt   = optim.Adam(model.parameters(), lr=cfg["lr"])
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit  = DiceBCE(a=0.5)

    os.makedirs("weights", exist_ok=True)
    best = 0.0

    for ep in range(cfg["epochs"]):
        model.train()
        for im, m, _ in tr:
            im, m = im.to(device), m.to(device)
            opt.zero_grad()
            out = model(im)
            loss = crit(out, m)
            loss.backward()
            opt.step()

        model.eval(); dice_sum=0; n=0
        with torch.no_grad():
            for im, m, _ in vl:
                im, m = im.to(device), m.to(device)
                pr = torch.sigmoid(model(im)) > 0.5
                inter = (pr & (m>0.5)).sum(dim=(2,3))
                dice = (2*inter) / (pr.sum(dim=(2,3)) + (m>0.5).sum(dim=(2,3)) + 1e-6)
                dice_sum += dice.mean().item(); n += 1
        dice = dice_sum/max(1,n)
        sch.step(1 - dice)
        if dice > best:
            best = dice
            torch.save(model.state_dict(), "weights/best.pt")
        print(f"Epoch {ep+1}/{cfg['epochs']} - val Dice={dice:.4f} (best {best:.4f})")
