import torch
import torch.nn as nn
import torchvision
'''
尝试按照课上优秀作业的思路来搭建Unet模型，能优化代码结构
决定采用resnext-wsl作为Unet的backbone
resnet chan --> [in_plane, 64, 256, 512, 1024, 2048]
'''

class UnetFactory(nn.Module):
    '''
    U型网络，encode, decode结构，中间将低维度信息作shortcut与高维度特征融合
    '''
    def __init__(self, encode_blocks, decode_blocks):
        super(UnetFactory, self).__init__()
        self.encode = UnetEncode(encode_blocks)
        self.decode = UnetDecode(decode_blocks)

    def forward(self, x):
        res = self.encode(x)
        out, skip = res[:1], res[1:]
        out = out[0]
        out = self.decode(out, skip)
        return out

def unetconv(in_plane, plane, padding=1):
    '''
    用于Unet的conv，在encode和decode中的conv
    尝试将bn放在relu后面。
    padding为1，这样输入和输出的size就不会改变
    ！或者尝试其他激活函数如GeLU
    '''
    return nn.Sequential(
        nn.Conv2d(in_plane, plane, kernel_size=3, padding=padding),
        nn.ReLU(),
        nn.BatchNorm2d(plane),
        nn.Conv2d(plane, plane, kernel_size=3, padding=padding),
        nn.ReLU(),
        nn.BatchNorm2d(plane)
    )

class UnetEncode(nn.Module):
    '''
    Unet encode中，每个block为一次下采样，最后一个block不进行下采样。total=5*block
    因为decode中需要skip concat，所以将encode中每个block输出保存，最后一个block不做concat
    '''
    def __init__(self, blocks):
        super(UnetEncode, self).__init__()
        self.layers = nn.ModuleList(blocks)

    def forward(self, x):
        skip = []
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            skip.append(x)
        res = [self.layers[-1](x)]
        res += skip
        return res

class UnetDecode(nn.Module):
    '''
    decode中， 以block为分解，每一个block做一次上采样，最后一个block不用，上采样后跟着Unetconv
    使用binarylinear，upsample，理论上不会有size不一致，但是为保证加上centercrop
    ！！因为skip顺序和decode中skip顺序相反，记得要reverse
    '''
    def __init__(self, blocks):
        super(UnetDecode, self).__init__()
        self.layers = nn.ModuleList(blocks)

    def _center_crop(self, layer, target):
        _, _, h1, w1 = layer.size()
        _, _, h2, w2 = target.size()
        ht, wt = min(h1, h2), min(w1, w2)
        nh1 = (h1 - ht) //2 if h1>ht else 0
        nw1 = (w1 - wt) //2 if w1>wt else 0
        nh2 = (h2 - ht) //2 if h2>ht else 0
        nw2 = (w2 - wt) //2 if w2>wt else 0
        if abs(h1-h2) == 1 :
            nh1 = (h1 - ht)+1 // 2 if h1 > ht else 0
            nh2 = (h2 - ht)+1 // 2 if h2 > ht else 0
        if abs(w1-w2) == 1:
            nw1 = (w1 - wt)+1 // 2 if w1 > wt else 0
            nw2 = (w2 - wt)+1 // 2 if w2 > wt else 0
        return layer[:, :, nh1:(nh1+h1), nw1:(nw1+w1)], target[:, :, nh2:(nh2+h2), nw2:(nw2+w2)]

    def forward(self, x, skip, reverse=True):
        assert len(self.layers)-1 == len(skip), 'block concat 数与skip不一致'
        if reverse:
            skip = skip[::-1]
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            skips, x = self._center_crop(skip[i-1], x)
            x = torch.cat([skips, x],dim=1)
            x = self.layers[i](x)
        return x


def restnextunet(cfg, resnet_model='resnext-wsl', pretrain=True):
    if resnet_model == 'resnext-wsl':
        encoder = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        encode_output_plane = [cfg.UNET_IN, 64, 256, 512, 1024, 2048]
    elif resnet_model == 'resnext50_32x4d':
        encoder = torchvision.models.resnext50_32x4d(pretrain)
        encode_output_plane = [cfg.UNET_IN, 64, 256, 512, 1024, 2048]
    else:
        raise ValueError('Unexpected resnet model')
    #encode block，使用resnext
    encode_block = [
        nn.Sequential(),
        nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu
        ),
        nn.Sequential(
            encoder.maxpool,
            encoder.layer1
        ),
        encoder.layer2,
        encoder.layer3,
        encoder.layer4
    ]

    for layer in encode_block:
        for param in layer.parameters():
            param.requires_grad = False

    decode_block = []
    in_ch = encode_output_plane[-1]
    out_ch = in_ch // 2
    decode_block.append(nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                      nn.Conv2d(in_ch, out_ch, 1)))
    for i in range(1, len(encode_block)-1):
        in_ch = encode_output_plane[-i-1] + out_ch
        decode_block.append(nn.Sequential(unetconv(in_ch, out_ch),
                                          nn.UpsamplingBilinear2d(scale_factor=2),
                                          nn.Conv2d(out_ch, out_ch//2, 1)))
        out_ch = out_ch // 2
    in_ch = encode_output_plane[0] + out_ch
    decode_block.append(nn.Sequential(unetconv(in_ch, out_ch),
                                      nn.Conv2d(out_ch, cfg.NUM_CLASS, kernel_size=1, bias=False)))
    return UnetFactory(encode_block, decode_block)

#class ResNextUnet(nn.Module):
    #def __init__(self):
