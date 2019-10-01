"""focal loss"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):

	def __init__(self, weight=None, alpha=0.25, gamma=2):
		super(FocalLoss, self).__init__()
		self.weights = weight
		self.alpha = alpha
		self.gamma = gamma


	def forward(self, predict, target):
		"""
		Args:
			predict:  (n, c, h, w)  class probabilities at each prediction (between 0 and 1)
			target: (n, h, w)   ground truth labels (between 0 and C - 1)
		"""
		n, c, h, w = predict.size()
		predict = torch.log(predict)
		criterion = nn.NLLLoss(self.weights) 

		logpt = -criterion(predict, target)
		pt = torch.exp(logpt)
		loss = - self.alpha * ((1 - pt)**self.gamma) * logpt

		return loss / n




