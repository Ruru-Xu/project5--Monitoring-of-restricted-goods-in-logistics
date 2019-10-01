"""
semantic segmentation metrics
Reference:
	https://en.wikipedia.org/wiki/Confusion_matrix
	https://github.com/jfzhang95/pytorch-deeplab-xception
"""

import numpy as np


class Evaluator():

	def __init__(self, num_class):
		self.num_class = num_class
		self.confusion_matrix = np.zeros((self.num_class,)*2)

	def pixel_accuracy(self):
		ACC = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
		return ACC

	def pixel_accuracy_class(self):
		ACC = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
		ACC = np.nanmean(ACC)
		return ACC

	def mean_intersection_over_union(self):
		MIoU = np.diag(self.confusion_matrix) / (
					np.sum(self.confusion_matrix, axis=1) +
					np.sum(self.confusion_matrix, axis=0) -
					np.diag(self.confusion_matrix))
		MIoU = np.nanmean(MIoU)
		return MIoU

	def frequency_weighted_intersection_over_union(self):
		freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
		iu = np.diag(self.confusion_matrix) / (
					np.sum(self.confusion_matrix, axis=1) +
					np.sum(self.confusion_matrix, axis=0) -
					np.diag(self.confusion_matrix))
		FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
		return FWIoU

	def _generate_matrix(self, target, predict):
		mask = (target >= 0) & (target < self.num_class)
		label = self.num_class * target[mask].astype('int') + predict[mask] # how to understand
		count = np.bincount(label, minlength=self.num_class**2)
		confusion_matrix = count.reshape(self.num_class, self.num_class)
		return confusion_matrix


	def add_batch(self, target, predict):
		assert target.shape == predict.shape
		self.confusion_matrix += self._generate_matrix(target, predict)

	def reset(self):
		self.confusion_matrix = np.zeros((self.num_class,)*2)
