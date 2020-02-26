from utils.label_preprocess import encode_labels, decode_label_color
from PIL import Image
import numpy as np
from utils.data_preprocess import ImageAug, DeformAug, ScaleAug
from torchvision import transforms
from utils.metric import compute_iou
'''
just for testing the function work
'''

dir = r'E:/data/Label/Label_road02/Label/Record001/Camera 5/170927_063813228_Camera_5_bin.png'
color_dir = r'E:/data/ColorImage/road02/Record001/Camera 5/170927_063813228_Camera_5.jpg'
img = Image.open(dir)
color_img = Image.open(color_dir)
gray_img = Image.open(dir)
gray_numpy = np.array(gray_img)
color_numpy = np.array(color_img)
gray_numpy = encode_labels(gray_numpy)
nof = gray_numpy == 5
print(np.sum(nof))
process = transforms.Compose(
    [ImageAug(), DeformAug(), ScaleAug()]
)
result = {'TP': {i:0 for i in range(8)}, 'TA': {i:0 for i in range(8)}}
result = compute_iou(gray_numpy, gray_numpy, result)
for i in range(8):
    print('{} iou is : {}'.format(i, result['TP'][i]/(result['TA'][i]+1)))