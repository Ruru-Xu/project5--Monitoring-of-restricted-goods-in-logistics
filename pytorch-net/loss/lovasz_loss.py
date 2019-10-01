"""
lovasz softmax loss
Reference:
	https://github.com/bermanmaxim/LovaszSoftmax
"""

import torch
import torch.nn as nn


class MultiLovaszLoss(nn.Module):
	"""lovasz loss for multi classes"""

	def __init__(self):
		super(MultiLovaszLoss, self).__init__()

	def forward(self, predict, target):
		"""
		Args:
			predict:  (n, c, h, w)  class probabilities at each prediction (between 0 and 1)
			target: (n, h, w)   ground truth labels (between 0 and C - 1)
		"""
		predict, target = self._flatten_predict(predict, target)
		loss = self._lovasz_softmax_flat(predict, target)
		return loss

	def _lovasz_softmax_flat(self, predict, target):
		"""
		Args:
			predict: (n * h * w, c)
			target:  (n * h * w)
		"""
		c = predict.size(1)
		losses = []
		for l in range(c):
			fg = (target == l).float()
			if fg.sum() == 0:
				continue
			errors = (fg - predict[:, l]).abs()
			errors_sorted, perm = torch.sort(errors, 0, descending=True)
			fg_sorted = fg[perm]
			losses.append(torch.dot(errors_sorted, self._lovasz_grad(fg_sorted)))
		return self._mean(losses)

	def _flatten_predict(self, predict, target):
		"""flattens predictions"""
		n, c, h, w = predict.size()
		predict = predict.permute(0, 2, 3, 1).reshape(-1, c)
		target = target.reshape(-1)
		return predict, target

	def _lovasz_grad(self, gt_sorted):
		"""Computes gradient of the Lovasz extension w.r.t sorted errors"""
		p = len(gt_sorted)
		gts = gt_sorted.sum()
		intersection = gts - gt_sorted.float().cumsum(0)
		union = gts + (1 - gt_sorted).float().cumsum(0)
		jaccard = 1. - intersection / union
		if p > 1:
			jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
		return jaccard

	def _mean(self, losses):
		losses = iter(losses)
		try:
			n = 1
			acc = next(losses)
		except StopIteration:
			return 0
		for n, v in enumerate(losses, 2):
			acc += v
		if n == 1:
			return acc
		return acc / n





		