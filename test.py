import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from utils.loss import MySoftmaxCrossEntropyLoss, DiceLoss, one_hot_encoder, compute_loss
from utils.data_preprocess import LaneDataset, DeformAug, ImageAug, ScaleAug, Totensor
from Train import load_model, get_confusion_matrix, get_logging
from module.module import ResBlock
from module.deeplabv3plus import Deeplabv3plus
from torchsummary import summary
from config import Config
from tensorboardX import SummaryWriter
from module.unet import restnextunet
import logging
'''
just for testing and debug
def One_Hot_Encoder(inputs, num_class):
    ''
    独热编码
    :param inputs:shape[N, 1, *]
    :param num_class: number of class
    :return: shape[N, number_class, *]
    ''
    shape = np.array(inputs.shape)
    shape[1] = num_class
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, inputs.cpu(), 1)
    return result

A = torch.randn((2, 1, 1010))
#print(A)
B = One_Hot_Encoder(A, 4)
print(B.shape)'''
'''
-----------------------------------------------------------------------------------------------
def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    label = label.view(-1, 384, 1024)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def val(net, dataloader, epoch, config):
    net.eval()
    confusion_matrix = np.zeros((config.NUM_CLASS, config.NUM_CLASS))
    dataprocess = tqdm(dataloader)
    for batch in dataprocess:
        image, mask = batch['image'].cuda(), batch['mask'].cuda()
        size = mask.size()
        out = net(image)
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASS)(out, mask)
        confusion_matrix += get_confusion_matrix(
            mask,
            out,
            size,
            config.NUM_CLASS,
            ignore=-1
        )
        dataprocess.set_description_str("Epoch {}:".format(epoch))
        dataprocess.set_postfix_str('mask_loss is : {:.4f}'.format(mask_loss.item()))
    pos = confusion_matrix.sum(0)
    res = confusion_matrix.sum(1)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    print(IoU_array)
    mean_IoU = IoU_array.mean()
    print(mean_IoU)


test_data = LaneDataset('val.csv', transform=Totensor())
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
lane_cfg = Config()
net = Deeplabv3plus(lane_cfg).cuda()
module_path = os.path.join(lane_cfg.MODEL_SAVE_PATH, 'predictParm.pth.tar')
net = load_model(net, module_path)
for i in range(2):
    val(net, test_loader, i, lane_cfg)
---------------------------------------------------------------
'''
device_list = [0]
nets = {'DeepLab': Deeplabv3plus, 'UNet': restnextunet}



def test_epoch(net, dataloader, logger, config):
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataloader)
    confusion_matrix = np.zeros((config.NUM_CLASS, config.NUM_CLASS))
    logger.info("Testing  : ")
    with torch.no_grad():
        for batch_item in dataprocess:
            image, mask = batch_item['image'], batch_item['mask']
            if torch.cuda.is_available():
                image, mask = image.cuda(), mask.cuda()
            out = net(image)
            weights = torch.tensor([0.75, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25]).cuda()
            mask_loss =compute_loss(out, mask, weights=weights, device_id=0, num_class=config.NUM_CLASS)
            total_mask_loss += mask_loss.detach().item()
            confusion_matrix += get_confusion_matrix(
                mask,
                out,
                mask.size(),
                config.NUM_CLASS
            )
            dataprocess.set_description_str('Test')
            dataprocess.set_postfix_str('mask loss is {:.4f}'.format(mask_loss.item()))
        logger.info("\taverage loss is {:.4f}".format(total_mask_loss/len(dataloader)))
        pos = confusion_matrix.sum(0)
        res = confusion_matrix.sum(1)
        tp = np.diag(confusion_matrix)
        IoU_array = (tp/ np.maximum(1.0, pos + res - tp))
        for i in range(8):
            print('{} IoU is : {}'.format(i, IoU_array[i]))
            logger.info('\t{} Iou is : {}'.format(i, IoU_array[i]))
        miou = IoU_array[1:].mean()
        logger.info('Test miou is : {:.4f}'.format(miou))
        print('Test: miou is {}'.format(miou))

def main():
    lane_seg_config = Config()
    test_logger = get_logging(lane_seg_config.LOG_PATH, 'test_log.txt')
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_dataset = LaneDataset('test.csv', transform=transforms.Compose([Totensor()]))
    test_data = DataLoader(test_dataset, batch_size=lane_seg_config.TRAIN_BATCH_SIZE,
                               shuffle=True, drop_last=True, **kwargs)
    net = nets[lane_seg_config.MODEL_NAME](lane_seg_config).cuda(device=device_list[0])
    models_path = os.path.join(lane_seg_config.MODEL_SAVE_PATH + '/predictParm.pth.tar')
    net = load_model(net, models_path)
    test_epoch(net, test_data, test_logger, lane_seg_config)

if __name__ == '__main__':
    main()