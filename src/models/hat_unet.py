
import torch, torch.nn as nn
from torchvision.ops import SqueezeExcitation

class ConvBNReLU(nn.Sequential):
    def __init__(self, c1, c2, k=3):
        super().__init__(
            nn.Conv2d(c1, c2, k, padding=k//2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

class AttnBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.se = SqueezeExcitation(c, max(1, c//4))
        self.spa = nn.Sequential(nn.Conv2d(c, 1, 1, bias=False), nn.Sigmoid())
    def forward(self, x):
        x = self.se(x)
        s = self.spa(x)
        return x * (1 + s)

class HATUNet(nn.Module):
    def __init__(self, base=32, out_ch=1):
        super().__init__()
        self.e1 = nn.Sequential(ConvBNReLU(3, base), ConvBNReLU(base, base), AttnBlock(base))
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(ConvBNReLU(base, 2*base), ConvBNReLU(2*base, 2*base), AttnBlock(2*base))
        self.p2 = nn.MaxPool2d(2)

        self.b  = nn.Sequential(ConvBNReLU(2*base, 4*base), AttnBlock(4*base))

        self.u2 = nn.ConvTranspose2d(4*base, 2*base, 2, 2)
        self.d2 = nn.Sequential(ConvBNReLU(4*base, 2*base), ConvBNReLU(2*base, 2*base))
        self.u1 = nn.ConvTranspose2d(2*base, base, 2, 2)
        self.d1 = nn.Sequential(ConvBNReLU(2*base, base), ConvBNReLU(base, base))
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        b  = self.b(self.p2(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)
