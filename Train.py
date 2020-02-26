from tqdm import tqdm
import os
import torch
import numpy as np
import logging
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data_preprocess import LaneDataset, ImageAug, DeformAug, ScaleAug
from utils.data_preprocess import cutout, resize_data, resize_color_data, Totensor
from utils.metric import compute_iou, get_confusion_matrix
from module.deeplabv3plus import Deeplabv3plus
from config import Config
from utils.loss import MySoftmaxCrossEntropyLoss, one_hot_encoder, DiceLoss
from tensorboardX import SummaryWriter
import shutil
from module.unet import restnextunet

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device_list = [0]
nets = {'DeepLab': Deeplabv3plus, 'UNet': restnextunet}

def train_epoch(net, epoch, dataloader, optimizer, writer, logger, config):
	#iter = 0
	net.train()
	confusion_matrix = np.zeros((config.NUM_CLASS, config.NUM_CLASS))
	total_mask_loss = 0.0
	dataprocess = tqdm(dataloader)
	logging.info("Train Epoch {}:".format(epoch))
	for batch_item in dataprocess:
		image, mask = batch_item['image'], batch_item['mask']
		if torch.cuda.is_available():
			image, mask = image.cuda(), mask.cuda()
		optimizer.zero_grad()
		out = net(image)
		mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASS)(out, mask)
		confusion_matrix += get_confusion_matrix(
			mask,
			out,
			mask.size(),
			config.NUM_CLASS,
		)
		#if iter % 10 == 0:
			#w.add_scalar('Epoch{}:Loss'.format(epoch), mask_loss.item(), iter)
		#iter += 1
		total_mask_loss += mask_loss.item()
		mask_loss.backward()
		optimizer.step()
		dataprocess.set_description('epoch:{}'.format(epoch))
		dataprocess.set_postfix_str('mask loss:{:.4f}'.format(mask_loss.item()))
	logger.info("\taverage loss is : {:.3f}".format(total_mask_loss/len(dataloader)))
	#confusion matrix
	pos = confusion_matrix.sum(0)
	res = confusion_matrix.sum(1)
	tp = np.diag(confusion_matrix)
	IoU_array = (tp/ np.maximum(1.0, pos + res - tp))
	for i in range(8):
		print('{} iou is : {:.4f}'.format(i, IoU_array[i]))
		logger.info("\t {} iou is : {:.4f}".format(i, IoU_array[i]))
	miou = IoU_array[1:].mean()
	print('EPOCH mIoU is : {}'.format(miou))
	logger.info("Train mIoU is : {:.4f}".format(miou))
	with writer as w:
		w.add_scalar('EPOCH Loss', total_mask_loss/len(dataloader), epoch)
		w.add_scalar('Train miou', miou, epoch)

def test_epoch(net, epoch, dataloader, writer, logger, config):
	net.eval()
	total_mask_loss = 0.0
	dataprocess = tqdm(dataloader)
	confusion_matrix = np.zeros((config.NUM_CLASS, config.NUM_CLASS))
	logger.info("Val EPOCH {}: ".format(epoch))
	with torch.no_grad():
		for batch_item in dataprocess:
			image, mask = batch_item['image'], batch_item['mask']
			if torch.cuda.is_available():
				image, mask = image.cuda(), mask.cuda()
			out = net(image)
			mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASS)(out, mask)
			total_mask_loss += mask_loss.detach().item()
			confusion_matrix += get_confusion_matrix(
				mask,
				out,
				mask.size(),
				config.NUM_CLASS
			)
			dataprocess.set_description_str('epoch{}:'.format(epoch))
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
		logger.info('Val miou is : {:.4f}'.format(miou))
		with writer as w:
			w.add_scalar('EPOCH Loss', total_mask_loss / len(dataloader), epoch)
			w.add_scalar('EPOCH mIoU', miou, epoch)
		print('epoch{}: miou is {}'.format(epoch, miou))

def adjust_lr(optimizer, epoch):
	if epoch == 5:
		lr = 6e-4
	elif epoch == 15:
		lr = 2e-4
	elif epoch == 20:
		lr = 1e-4
	else:
		return
	for param_group in optimizer.param_gropus:
		param_group['lr'] = lr

def load_model(nets, model_path):
	device_id = device_list[0]
	map_location = 'cuda:%d' % device_id
	model_param = torch.load(model_path, map_location=map_location)['state_dict']
	model_param = {k.replace('model.', ''):v for k,v in model_param.items()}
	nets.load_state_dict(model_param)
	return nets

def get_logging(dir, name='train_log.txt'):
	word_dir = dir
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s", datefmt="%a %b %d %H:%M:%S %Y")
	fHandler = logging.FileHandler(word_dir+'/'+name, mode='w')
	fHandler.setLevel(level=logging.DEBUG)
	fHandler.setFormatter(fmt=formatter)
	logger.addHandler(fHandler)
	return logger

def main():
	lane_seg_config = Config()
	if not os.path.exists(lane_seg_config.MODEL_SAVE_PATH):
		os.makedirs(lane_seg_config.MODEL_SAVE_PATH)
	if os.path.exists(lane_seg_config.LOG_PATH):
		shutil.rmtree(lane_seg_config.LOG_PATH)
	os.makedirs(lane_seg_config.LOG_PATH, exist_ok=True)
	train_logger = get_logging(lane_seg_config.LOG_PATH, 'train_log.txt')
	val_logger = get_logging(lane_seg_config.LOG_PATH, 'val_log.txt')
	kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
	training_dataset = LaneDataset('train.csv', sizes=lane_seg_config.SIZE, transform=transforms.Compose([ImageAug(), DeformAug(),
																			ScaleAug(), Totensor()]))
	training_data = DataLoader(training_dataset, batch_size=lane_seg_config.TRAIN_BATCH_SIZE,
							   shuffle=True, drop_last=True, **kwargs)
	# drop last , 当数据集剩下的数据不足一个batchsize的时候，剩下的会被丢弃，进行下一个epoch训练
	val_dataset = LaneDataset('val.csv', sizes=lane_seg_config.SIZE, transform=Totensor())
	val_data = DataLoader(val_dataset, batch_size=lane_seg_config.VAL_BATCH_SIZE, shuffle=False,
						  drop_last=False, **kwargs)
	net = nets[lane_seg_config.MODEL_NAME](lane_seg_config)
	#net = Deeplabv3plus(lane_seg_config)
	#net = restnextunet(lane_seg_config)
	if torch.cuda.is_available():
		net = net.cuda(device=device_list[0])
	if lane_seg_config.PRETRAIN:
		models_path = os.path.join(lane_seg_config.MODEL_SAVE_PATH + '/predictParm.pth.tar')
		net = load_model(net, models_path)
	trainwriter = SummaryWriter(logdir='runs/train')
	testwriter = SummaryWriter(logdir='runs/test')
	optimizer = torch.optim.Adam(net.parameters(), lr=lane_seg_config.BASE_LR, weight_decay=lane_seg_config.WEIGHT_DECAY)
	for epoch in range(lane_seg_config.EPOCH):
		#adjust_lr(optimizer, epoch)
		train_epoch(net, epoch, training_data, optimizer, trainwriter, train_logger, lane_seg_config)
		test_epoch(net, epoch, val_data, testwriter, val_logger, lane_seg_config)
		if epoch % 5 == 0:
			torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_seg_config.MODEL_SAVE_PATH, 'SegLaneNet{}.pth.tar'.format(epoch)))
	torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_seg_config.MODEL_SAVE_PATH, 'FinalNet.pth.tar'))

# 包装，进度条
if __name__ == '__main__':
	main()