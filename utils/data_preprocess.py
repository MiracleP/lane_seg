import os 
from PIL import Image
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
from utils.label_preprocess import encode_labels, decode_labels, decode_label_color

sometimes = lambda aug: iaa.Sometimes(0.5, aug)


#crop and resize the image
def crop_resize_data(image, label=None, img_resize=(1024, 384), offset=690):
	roi_image = image.crop((0, offset, 3384, 1710))
	if label is not None:
		roi_label = label.crop((0, offset, 3384, 1710))
		train_label = roi_label.resize(img_resize, Image.BILINEAR)
		train_image = roi_image.resize(img_resize, Image.BILINEAR)
		return train_image, train_label
	else:
		train_image = roi_image.resize(img_resize, Image.BILINEAR)
		return train_image

# dataset for dataloader
class LaneDataset(Dataset):
	'''
	cvs_file: train.csv, test.csv, val.csv
	transform: data augmentation
	csv: root/data_list/xxx.csv
	'''
	def __init__(self, csv_file, sizes, transform=False):
		super(LaneDataset, self).__init__()
		self.data = pd.read_csv(os.path.join(os.getcwd(), 'data_list', csv_file), header=None,
								names=['image', 'label'])
		self.images = self.data['image'].values[1:]
		self.labels = self.data['label'].values[1:]
		self.sizes = sizes
		self.transform = transform

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self, idx):
		ori_image = Image.open(self.images[idx])
		ori_mask = Image.open(self.labels[idx])
		train_image, train_mask = crop_resize_data(ori_image, ori_mask, img_resize=self.sizes)
		train_image, train_mask = np.array(train_image), np.array(train_mask)
		#Encode
		train_mask = encode_labels(train_mask)
		#train_image = Image.fromarray(train_image.astype('uint8')).convert('RGB')
		#train_mask = Image.fromarray(train_mask.astype('uint8')).convert('RGB')
		sample = [train_image.copy(), train_mask.copy()]
		if self.transform is not None:
			sample = self.transform(sample)

		return sample  #type=np.array,shape=(h, w, c)

#data augmentation,with iaa
class ImageAug(object):
	def __call__(self, sample):
		'''
		no need to process mask
		'''
		image, mask = sample
		if np.random.uniform(0,1) >0.5:
			aug = iaa.Sequential([iaa.OneOf([
				iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),
				iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)), 
				iaa.GaussianBlur(sigma=(0, 1.0))])])
			image = aug.augment_image(image)

		return image, mask

class DeformAug(object):
	def __call__(self, sample):
		'''
		image,label 
		'''
		image, mask = sample
		aug = iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.1))])
		aug_to = aug.to_deterministic()
		image = aug_to.augment_image(image)
		mask = aug_to.augment_image(mask)

		return image, mask

#scale Augment
#channal is [h, w, c]
class ScaleAug(object):
	def __call__(self, sample):
		image, mask = sample
		scale = np.random.uniform(0.7, 1.5)
		h, w, _ = image.shape
		aug_image = Image.fromarray(image, 'RGB')
		aug_mask = Image.fromarray(mask)
		aug_image = aug_image.resize((int( w*scale), int( h*scale)))
		aug_mask = aug_mask.resize((int( w*scale), int( h*scale)))
		aug_image, aug_mask = np.array(aug_image), np.array(aug_mask)
		#if scale<1 then,pad
		if scale < 1.0:
			new_h, new_w, _ = aug_image.shape
			pre_h = int((h - new_h) / 2)
			pre_w = int((w - new_w) / 2)
			# maybe h or w is odd,so pad like this
			pad_list = [[pre_h, h-new_h-pre_h], [pre_w, w-pre_w-new_w], [0, 0]]#the last list is channal
			aug_image = np.pad(aug_image, pad_list, mode='constant')
			aug_mask = np.pad(aug_mask, pad_list[:2], mode='constant')
		#scale>1.0, then crop
		elif scale > 1.0:
			new_h, new_w, _ = aug_image.shape
			pre_h = int((new_h - h) / 2)
			pre_w = int((new_w - w) / 2)
			post_h = h + pre_h
			post_w = w + pre_w
			aug_image = aug_image[pre_h:post_h, pre_w:post_w, :]
			aug_mask = aug_mask[pre_h:post_h, pre_w:post_w]
			
		return aug_image, aug_mask

#random cutout part of image, 
class cutout(object):
	'''
	mask_size: cutout mask size
	p: 概率
	'''
	def __init__(self, mask_size, p):
		self.mask_size = mask_size
		self.p = p

	def __call__(self, sample):
		image, mask = sample
		half_mask_size = self.mask_size // 2
		offset = 1 if not self.mask_size % 2 == 0 else 0
		h, w = image.shape[:2] #[h, w, c]
		cminx, cmaxx = half_mask_size, w + offset - half_mask_size
		cminy, cmaxy = half_mask_size, h + offset - half_mask_size
		cx = np.random.randint(cminx, cmaxx)
		cy = np.random.randint(cminy, cmaxy)
		xmin, ymin = cx - half_mask_size, cy - half_mask_size
		xmax, ymax =  cx + self.mask_size, cy + self.mask_size
		xmin, xmax, ymin, ymax = max(0, xmin),  min(w, xmax), max(0, ymin), min(h, ymax)
		if random.uniform(0, 1) < self.p:
			image[ymin:ymax, xmin:xmax] = (0, 0, 0)
			mask[ymin:ymax, xmin:xmax] = 0
		return image, mask

def resize_data(prediction=None, submission_size=(3384, 1710), offset=690):
	pred = decode_labels(prediction)
	#pred = Image.fromarray(pred.astype('uint8'))
	pred = np.resize(pred, (submission_size[1], submission_size[0] - offset))#, Image.BILINEAR)
	#box = (submission_size[1]-offset, submission_size[1], 0, submission_size[0])
	submission_mask = np.zeros((submission_size[1], submission_size[0]), dtype='uint8')
	submission_mask[offset:, :] = pred
	#type is np.array
	return submission_mask

def resize_color_data(prediction=None, submission_size=(3382, 1710), offset=690):
	'''
	input PILImage File
	output PILImage File
	:param prediction:
	:param submission_size:
	:param offset:
	:return:
	'''
	#pred = decode_label_color(prediction)
	#pred = Image.fromarray(pred.astype('uint8'))
	#decode label shape , (c, h, w)
	##change channal(0, 1 ,2) --> (1, 2, 0)
	target = Image.new('RGB', submission_size)
	pred = prediction.resize((submission_size[0], submission_size[1]-offset), Image.BILINEAR)
	target.paste(pred, (0, offset, submission_size[0], submission_size[1]))
	return target
	#pred = np.transpose(pred, (1, 2, 0))
	#pred = np.resize(pred, (submission_size[1], submission_size[0] - offset))#, Image.BILINEAR)
	#box = (submission_size[1]-offset, submission_size[1], 0, submission_size[0])
	#submission_mask = np.zeros((submission_size[1], submission_size[0], 3), dtype='uint8')#numpy shape [h, w, c]
	#submission_mask[offset:, :, :] = pred
	#type is np.array
	#return submission_mask

class Totensor(object):
	def __call__(self, sample):
		image, mask = sample
		# h, w, c --> c, h, w
		return {'image': transforms.ToTensor()(image.copy()),
				'mask': transforms.ToTensor()(mask.copy())}
