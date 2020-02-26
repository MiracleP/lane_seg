import torch
import torch.nn as nn
from module.resnet import resnet50
from module.module import ASPP

class Deeplabv3plus(nn.Module):
    def __init__(self, cfg):
        super(Deeplabv3plus, self).__init__()
        self.backbone = resnet50(pretrained=True, os=cfg.OUTPUT_STRIDE)
        in_plane = 2048
        self.aspp = ASPP(in_plane, cfg.ASPP_OUTDIM)
        self.dropout = nn.Dropout(0.5)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_aspp = nn.UpsamplingBilinear2d(scale_factor=cfg.OUTPUT_STRIDE//4)
        in_dim = 256
        self.shortconv = nn.Sequential(
            nn.Conv2d(in_dim, cfg.SHORTCUT_DIM, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cfg.SHORTCUT_DIM),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(cfg.ASPP_OUTDIM+cfg.SHORTCUT_DIM, cfg.ASPP_OUTDIM, kernel_size=3, stride=1,
                      padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(cfg.ASPP_OUTDIM),
            nn.CELU(alpha=0.075, inplace=False),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.ASPP_OUTDIM, cfg.ASPP_OUTDIM, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cfg.ASPP_OUTDIM),
            nn.CELU(alpha=0.075, inplace=False),
            nn.Dropout(0.1)
        )
        self.clt_conv = nn.Conv2d(cfg.ASPP_OUTDIM, cfg.NUM_CLASS, 1, 1, 0, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layer = self.backbone(x)
        asppfeature = self.aspp(layer[-1])
        asppfeature = self.dropout(asppfeature)
        asppfeature = self.upsample_aspp(asppfeature)

        shorcut = self.shortconv(layer[0])
        result = torch.cat([asppfeature, shorcut], 1)
        result = self.cat_conv(result)
        result = self.clt_conv(result)
        result = self.upsample_sub(result)
        return result