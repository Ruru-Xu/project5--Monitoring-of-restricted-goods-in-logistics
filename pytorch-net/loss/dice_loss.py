"""dice loss"""

import numpy as np
import torch
import torch.nn as nn


class DiceLoss(nn.Module):

	def __init__(self):
		super(DiceLoss, self).__init__()
		self.smooth = 1.0

	def forward(self, predict, target):
		N = target.size(0)
		predict = predict.reshape(N, -1)
		target = target.reshape(N, -1)

		intersection = predict * target
		loss = (2 * intersection + self.smooth) / (predict.sum(1) + target.sum(1) + self.smooth)
		loss = 1 - loss.sum() / N
		return loss


class MultiDiceLoss(nn.Module):
	"""dice loss for multi classes"""

	def __init__(self, weights=None):
		super(MultiDiceLoss, self).__init__()
		self.weights = weights
		self.smooth = 1.0

	def forward(self, predict, target):
		"""
		Args:
			predict:  (n, c, h, w)  class probabilities at each prediction (between 0 and 1)
			target: (n, h, w)   ground truth labels (between 0 and C - 1)
		"""
		n, c, h, w = predict.size()
		one_hot_target = torch.zeros_like(predict)
		one_hot_target.scatter_(1, target.unsqueeze(1), 1) # label to one hot

		predict = predict.view(n, c, -1)
		one_hot_target = one_hot_target.contiguous().view(n, c, -1)

		intersection = torch.sum(predict * one_hot_target, 2) # (n, c)
		union = torch.sum(predict, 2) + torch.sum(one_hot_target, 2)

		loss = 1.0 - torch.sum((2.0 * intersection + self.smooth) / \
								(union + self.smooth), 0) / n

		if self.weights is not None:
			loss = self.weights * loss

		return loss.mean()


def multi_dice_coef(predict, target):
	"""
	dice coefficient
	"""
	smooth = 1.0
	n, c, h, w = predict.size()
	one_hot_target = torch.zeros_like(predict)
	one_hot_target.scatter_(1, target.unsqueeze(1), 1) # label to one hot

	predict_de = predict.detach().view(n, c, -1)
	one_hot_target = one_hot_target.contiguous().view(n, c, -1)

	intersection = torch.sum(predict_de * one_hot_target, 2) # (n, c)
	union = torch.sum(predict_de, 2) + torch.sum(one_hot_target, 2)

	dice = (2.0 * intersection + smooth) / (union + smooth)

	return torch.mean(dice)










