from module.unet import restnextunet
from torchsummary import summary
from config import Config
import torchvision

cfg = Config()
net = restnextunet(cfg, resnet_model='resnext50_32x4d',pretrain=False)
#net = torchvision.models.resnext50_32x4d()
summary(net.cuda(), (3, 512, 256))
#print(net)