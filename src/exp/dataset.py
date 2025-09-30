
import os, glob, cv2, torch
import albumentations as A
from torch.utils.data import Dataset

class Colonies(Dataset):
    def __init__(self, img_dir, msk_dir, size=256, train=True):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        self.msk_dir = msk_dir
        if train:
            self.tf = A.Compose([
                A.LongestMaxSize(size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.ElasticTransform(alpha=20, sigma=5, alpha_affine=10, p=0.2),
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        import numpy as np, os
        ip = self.imgs[i]
        name = os.path.splitext(os.path.basename(ip))[0] + ".png"
        mp = os.path.join(self.msk_dir, name)
        im = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        m  = cv2.imread(mp, 0)
        if m is None:
            m = (0*im[:,:,0]).astype("uint8")
        a = self.tf(image=im, mask=m)
        im, m = a["image"], a["mask"]
        im = torch.from_numpy(im.transpose(2,0,1)).float()/255.0
        m  = torch.from_numpy((m>0).astype("float32")).unsqueeze(0)
        return im, m, ip
