import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_loss(pred, label, weights, device_id, num_class=8):
    one_hot_label = one_hot_encoder(label, num_class).cuda(device=device_id)
    ce = MySoftmaxCrossEntropyLoss(nbclasses=num_class, weight=weights)(pred, label)
    dice_loss = DiceLoss(weight=weights)(pred, one_hot_label)
    loss = ce + dice_loss
    return loss


class MySoftmaxCrossEntropyLoss(nn.Module):
    '''
    CrossEntropyLoss
    channal: [n,C]--> in baidu segmentic competition ,class is 8
    :arg
    nbclasses: number of class
    '''
    def __init__(self, nbclasses, weight=None):
        super(MySoftmaxCrossEntropyLoss, self).__init__()
        self.nbclasses = nbclasses
        self.weight = weight

    def forward(self, inputs, target):
        if inputs.dim() >2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)#[N, C, H, W] --> [N, C, H*W]
            inputs = inputs.transpose(1, 2)#[N, C, H*W] --> [N, H*W, C]
            inputs = inputs.contiguous().view(-1, self.nbclasses)#[N, H*W, C] --> [N*H*W, C]
            #!!!!!!!!!  inputs.contiguous()的作用
            target = target.view(-1).long()
        return nn.CrossEntropyLoss(weight=self.weight, reduction='mean')(inputs, target)

class FocalLoss(nn.Module):
    '''

    '''
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim() >2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)#[N, C, H, W] --> [N, C, H*W]
            inputs = inputs.transpose(1, 2)#[N, C, H*W] --> [N, H*W, C]
            inputs = inputs.contiguous().view(-1, self.nbclasses)#[N, H*W, C] --> [N*H*W, C]
            target = target.view(-1, 1)

        #理解实现原理
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma *logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def one_hot_encoder(inputs, num_class):
    '''
    独热编码
    :param inputs:shape[N, 1, *]
    :param num_class: number of class
    :return: shape[N, number_class, *]
    '''
    shape = np.array(inputs.shape)
    shape[1] = num_class
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, inputs.cpu().long(), 1)
    return result

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, inputs, target):
        assert inputs.shape[0] == target.shape[0], 'inputs and target batch size dont match'
        inputs = inputs.contiguous().view(inputs.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = 2*torch.sum(torch.mul(inputs, target), dim=1) + self.smooth
        den = torch.sum(inputs.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        '''

        :param weights: a array of weights
        :param ignore_index:
        :param kwargs:  kwargs of binarydiceloss smooth, p, reduction
        '''
        super(DiceLoss, self).__init__()
        self.weights = weight
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def forward(self, inputs, target):
        assert inputs.shape == target.shape, 'inputs shape dont match target shape'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        inputs = F.softmax(inputs, dim=1)

        for i in range(inputs.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(inputs[:, i], target[:, i])

                if self.weights is not None:
                    self.weights = self.weights.view(-1)
                    assert self.weights.shape[0] == target.shape[1], \
                    'Except weight shape {}, bug got shape {} '.format(target.shape[1], self.weights.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
        return total_loss / target.shape[1]