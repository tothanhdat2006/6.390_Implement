import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.double_conv(X)

class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownLayer, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, 1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, X):
        return self.maxpool_conv(X)

class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super(UpLayer, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, X, C):
        X = self.up(X)
        diffY = C.size()[2] - X.size()[2] #H
        diffX = C.size()[3] - X.size()[3] #W
        X = nn.functional.pad(X, [diffX // 2, diffX - diffX / 2,
                                    diffY // 2, diffY - diffY / 2,])
        X = torch.cat([C, X], dim = 1)
        return self.conv(X)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
