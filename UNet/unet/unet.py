import torch
import torch.nn as nn
import torch.nn.functional as F

class UNET(nn.Module):
    def __init__(self) -> None:
        super(UNET, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 1)

    def contracting_path(self, X):
        # 572 x 572
        out1 = self.down1(X)
        X = self.maxpool(out1) # 568 x 568
        # 284 x 284
        out2 = self.down2(X)
        X = self.maxpool(out2) # 280 x 280
        # 140 x 140
        out3 = self.down3(X)
        X = self.maxpool(out3) # 136 x 136
        # 68 x 68
        out4 = self.down4(X)
        X = self.maxpool(out4) # 64 x 64
        return out1, out2, out3, out4, X
    
    def expansive_path(self, X, c4, c3, c2, c1):
        
        return
    
    def forward(self, X):
        c1, c2, c3, c4, X = self.contracting_path(X) # 64 x 64
        X = self.maxpool(X) # 32 x 32
        for _ in range(2):
            X = nn.Conv2d(512, 1024, kernel_size=3, stride=1)(X)
        # X: 28 x 28
        X = self.expansive_path(X, c4, c3, c2, c1)
        return X