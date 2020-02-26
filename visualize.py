import torch
from utils.data_preprocess import LaneDataset, cutout, DeformAug, ScaleAug, ImageAug, Totensor
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Config
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from utils.label_preprocess import decode_labels

lane_config = Config()
kwarge = {'num_workers':0, 'pin_memory':True}
Lane_data = LaneDataset('train.csv', sizes=lane_config.SIZE, transform=transforms.Compose([ImageAug(),
                                                                                           cutout(20, 0.4),
                                                                                           ScaleAug(),
                                                                                           DeformAug(),
                                                                                           Totensor()]))
lane_dataloader = DataLoader(Lane_data, batch_size=16, shuffle=True, **kwarge)
lane_process = tqdm(lane_dataloader)
for batch_items in lane_process:
    image, mask = batch_items['image'], batch_items['mask']
    mask = mask.view(16, 384, -1)
    image = image.numpy()
    mask = mask.numpy()
    img = decode_labels(mask[0])
    img = Image.fromarray(img)
    plt.figure()
    plt.subplot(211)
    plt.imshow(np.transpose(image[0], (1, 2, 0)))
    plt.subplot(212)
    plt.imshow(mask[0])
    plt.show()
    img.show()