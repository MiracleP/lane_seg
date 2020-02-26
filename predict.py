from PIL import Image
import numpy as np
from config import Config
from module.deeplabv3plus import Deeplabv3plus
from utils.data_preprocess import resize_color_data, crop_resize_data
from utils.label_preprocess import decode_label_color
import torch
from torchvision import transforms
import os

device_id = 0


def load_model(cfg):
    '''
    :param cfg: config file : param save path, model config,etc..
    :return: 返回训练好的模型
    '''
    model_path = os.path.join(cfg.MODEL_SAVE_PATH+'/predictParm.pth.tar')
    net = Deeplabv3plus(cfg)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda(device=device_id)
        map_location = 'cuda:%d' % device_id
    model_parm = torch.load(model_path, map_location=map_location)['state_dict']
    model_parm = {k.replace('module.', ''):v for k, v in model_parm.items()}
    net.load_state_dict(model_parm)
    return net

def img_transform(img):
    '''
    input:pil file
    :param img: 接收图片输入，需要做预处理，resize，totensor等，还可以做torch的normalization
    :return: 预处理好的tensor
    '''
    img = crop_resize_data(img)
    #shape(h, w, c)
    img = np.array(img)
    img = transforms.ToTensor()(img)
    img = img.view(1, 3, 384, -1)
    #img = np.transpose(img, (2, 0, 1))
    #img = img[np.newaxis, ...].astype(np.float32)
    #img = torch.from_numpy(img)
    if torch.cuda.is_available():
        img = img.cuda(device=device_id)
    return img

def color_mask(pred):
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    pred = torch.squeeze(pred)
    pred = pred.detach().cpu().numpy()
    pred = decode_label_color(pred)
    pred = np.transpose(pred, (1, 2, 0))
    return pred

def main():
    lane_seg_config = Config()
    net = load_model(lane_seg_config)
    if not os.path.exists(lane_seg_config.TEST_PATH):
        os.makedirs(lane_seg_config.TEST_PATH)
    img_dir = r'E:/data/ColorImage/road04/Record002/Camera 6/'
    for name in os.listdir(img_dir):
        img_name = os.path.join(img_dir, name)
        img = Image.open(img_name)
        input_tensor = img_transform(img)
        out = net(input_tensor)
        pred_mask = color_mask(out)
        pred_mask = Image.fromarray(pred_mask, mode='RGB')
        pred_mask.save(lane_seg_config.TEST_PATH+'/mask_{}'.format(name))

if __name__ == '__main__':
    main()

