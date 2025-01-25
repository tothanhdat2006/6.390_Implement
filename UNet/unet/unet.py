import torch
import torch.nn as nn
import torch.nn.functional as F

class UNET(nn.Module):
    def __init__(self) -> None:
        super(UNET, self).__init__()

    def contracting_path(self):
        return
    
    def expansive_path(self):
        return
    
    def forward(self, X):
        return X