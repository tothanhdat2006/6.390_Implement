import torch
import torch.nn as nn
import torch.nn.functional as F

from unet.unet_parts import *

class UNET(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear) -> None:
        super(UNET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if self.bilinear else 1

        self.inp = DoubleConv(n_channels, 64)
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 1024 // factor)

        self.up1 = UpLayer(1024, 512 // factor, bilinear)
        self.up2 = UpLayer(512, 256 // factor, bilinear)
        self.up3 = UpLayer(256, 128 // factor, bilinear)
        self.up4 = UpLayer(128, 64 // factor, bilinear)
        self.out = OutConv(64, n_classes)
        

    def contracting_path(self, X):
        out1 = self.down1(X) # 280 x 280
        out2 = self.down2(X) # 136 x 136
        out3 = self.down3(X) # 64 x 64
        out4 = self.down4(X) # 28 x 28
        return out1, out2, out3, out4, X
    
    def expansive_path(self, X, c4, c3, c2, c1):
        X = self.up1(X, c4) # 52 x 52
        X = self.up2(X, c3) # 100 x 100
        X = self.up3(X, c2) # 196 x 196
        X = self.up4(X, c1) # 388 x 388
        return X
    
    def forward(self, X):
        X = self.inp(X) # 568 x 568
        c1, c2, c3, c4, X = self.contracting_path(X)
        X = self.expansive_path(X, c4, c3, c2, c1)
        logits = self.out(X)
        return logits
    
    def use_checkpointing(self):
        self.inp = torch.utils.checkpoint(self.inp)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.out = torch.utils.checkpoint(self.out)