
import torch, torch.nn as nn

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2,3))
    den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps
    loss = 1 - (num / den)
    return loss.mean()

class DiceBCE(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.a * self.bce(logits, targets) + (1 - self.a) * dice_loss(logits, targets)
