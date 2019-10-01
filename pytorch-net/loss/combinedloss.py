"""combine several losses"""

import torch
import torch.nn as nn
from loss.focal_loss import FocalLoss
from loss.dice_loss import MultiDiceLoss

class CombinedLoss(nn.Module):
	"""docstring for CombinedLoss"""

	def __init__(self, alpha=10, weight=None):
		super(CombinedLoss, self).__init__()
		self.alpha = alpha
		self.dice_loss = MultiDiceLoss(weight)
		self.focal_loss = FocalLoss(weight)

	def forward(self, predict, target):
		loss = self.alpha * self.focal_loss(predict, target) - torch.log(self.dice_loss(predict, target))
		return loss.mean()

