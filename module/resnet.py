import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}

def conv3x3(in_plane, plane, stride=1, atrous=1):
    return nn.Conv2d(in_plane, plane, kernel_size=3, padding=atrous, stride=stride,
                     dilation=atrous, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_plane, plane, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_plane, plane, stride=stride, atrous=atrous)
        self.bn1 = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(plane, plane)
        self.bn2 = nn.BatchNorm2d()
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_plane, plane, stride=1, atrous=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, plane, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(plane, plane, stride, atrous)
        self.bn2 = nn.BatchNorm2d(plane)
        self.conv3 = nn.Conv2d(plane, self.expansion*plane, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*plane)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        result = self.relu(out)
        return result

class ResNet(nn.Module):
    def __init__(self, block, layer, atrous=None, os=16):
        super(ResNet, self).__init__()
        stride_list = None
        if os == 8:
            stride_list = [2, 1, 1]
        elif os == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('ResNet.py : output stride=%d is not suported' % os)

        self.in_plane = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layer[0])
        self.layer2 = self._make_layer(block, 128, layer[1], stride_list[0])
        self.layer3 = self._make_layer(block, 256, layer[2], stride_list[1], atrous= 16 // os)
        self.layer4 = self._make_layer(block, 512, layer[3], stride_list[2],
                                       atrous=[item * 16 // os for item in atrous])
        self.layer5 = self._make_layer(block, 512, layer[3], stride=1,
                                       atrous=[item * 16 // os for item in atrous])
        self.layer6 = self._make_layer(block, 512, layer[3], stride=1,
                                       atrous=[item * 16 // os for item in atrous])
        #初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, plane, block_num, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1] * block_num
        elif isinstance(atrous, int):
            atrous_list = [atrous] * block_num
            atrous = atrous_list
        # downsample
        if stride != 1 or self.in_plane != plane*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_plane, plane*block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(plane*block.expansion)
            )
        layer = []
        layer.append(block(self.in_plane, plane, stride=stride, atrous=atrous[0], downsample=downsample))
        self.in_plane = plane*block.expansion
        for i in range(1, block_num):
            layer.append(block(self.in_plane, plane, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layer)

    def forward(self, x):
        layer_list = []
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        layer_list.append(x)
        x = self.layer2(x)
        layer_list.append(x)
        x = self.layer3(x)
        layer_list.append(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        layer_list.append(x)
        return layer_list

def resnet50(pretrained=True, os=16, **kwargs):
    model = ResNet(BottleNeck, [3, 4, 6, 3], [1, 2, 1], os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        old_dict = {k:v for k,v in old_dict.items() if k in model_dict}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model

def resnet101(pretrained=True, os=16, **kwargs):
    model = ResNet(BottleNeck, [3, 4, 23, 3], [1, 2, 1], os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        old_dict = {k:v for k,v in old_dict.items() if k in model_dict}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model
