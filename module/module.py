import torch
import torch.nn as nn
import torch.nn.functional as F



class ResBlock(nn.Module):
    def __init__(self, in_plane, plane, kernel_size = 3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_plane, plane, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return out

class BottleNeck(nn.Module):
    def __init__(self, in_plane, plane):
        super(BottleNeck, self).__init__()
        self.block1 = ResBlock(in_plane, int(plane/4), kernel_size=1, padding=0)
        self.block2 = ResBlock(int(plane/4), int(plane/4))
        self.block3 = ResBlock(int(plane/4), plane, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out

class DownBottleNeck(nn.Module):
    def __init__(self, in_plane, plane, stride=2):
        super(DownBottleNeck, self).__init__()
        self.block1 = ResBlock(in_plane, int(plane/4), kernel_size=1, stride=stride, padding=0)
        self.skipconv = nn.Conv2d(in_plane, plane, 1, stride, 0, bias=False)
        self.block2 = ResBlock(int(plane/4), int(plane/4))
        self.block3 = ResBlock(int(plane/4), plane, 1, padding=0)

    def forward(self, x):
        out = self.block1(x)
        identity = self.skipconv(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out

class ASPP(nn.Module):
    def __init__(self, in_plane, plane, dilation_rate=1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_plane, plane, 1, 1, padding=0, dilation=dilation_rate, bias=True),
            nn.BatchNorm2d(plane),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_plane, plane, 3, 1, padding=6*dilation_rate, dilation=6*dilation_rate, bias=True),
            nn.BatchNorm2d(plane),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_plane, plane, 3, 1, padding=12*dilation_rate, dilation=12*dilation_rate, bias=True),
            nn.BatchNorm2d(plane),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_plane, plane, 3, 1, padding=18*dilation_rate, dilation=18*dilation_rate, bias=True),
            nn.BatchNorm2d(plane),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_plane, plane, 1, 1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(plane*5, plane, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        n, c, h, w = x.size()
        layer1 = self.branch1(x)
        layer2 = self.branch2(x)
        layer3 = self.branch3(x)
        layer4 = self.branch4(x)
        gobal_feature = self.branch5(x)
        gobal_feature = F.interpolate(gobal_feature, (h, w), None, 'bilinear', True)
        feature_cat = torch.cat([layer1, layer2, layer3, layer4, gobal_feature], dim=1)
        result = self.cat_conv(feature_cat)
        return result
