"""dense dilated net"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


"""
# torch.nn.GroupNorm
https://pytorch.org/docs/stable/nn.html?highlight=group%20norm#torch.nn.GroupNorm
"""

class DilatedConv(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(DilatedConv, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=5, dilation=5)
		self.bn3 = nn.BatchNorm2d(out_channels)
		self.conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=7, dilation=7)
		self.bn4 = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		return x

class Concatenate(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(Concatenate, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, 1, padding=0)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		"""
		Args:
			x(list):
		"""
		x = torch.cat(x, dim=1)
		x = self.bn(self.conv(x))
		return x

class DDNet(nn.Module):

	def __init__(self, in_channels, num_classes):
		"""
		Args:
			in_channels(int): 
			num_classes(int): number of classes
		"""
		super(DDNet, self).__init__()
		self.layer1 = DilatedConv(in_channels, 64)
		self.layer2 = DilatedConv(64, 128)
		self.cat2 = Concatenate(64+128, 128)
		self.layer3 = DilatedConv(128, 128)
		self.cat3 = Concatenate(64+128+128, 128)
		self.layer4 = DilatedConv(128, 128)
		self.cat4 = Concatenate(64+128+128+128, 128)
		# self.cat4 = Concatenate(64+128+128+128, num_classes)
		self.layer5 = DilatedConv(128, 128)
		self.cat5 = Concatenate(64+128+128+128+128, num_classes)


	def forward(self, x):
		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.layer3(self.cat2([x1, x2]))
		x4 = self.layer4(self.cat3([x1, x2, x3]))
		x5 = self.layer5(self.cat4([x1, x2, x3, x4]))
		return self.cat5([x1, x2, x3, x4, x5])

if __name__ == '__main__':

	ddnet = DDNet(3, 21)
	# print(ddnet)
	out = ddnet(torch.randn(2, 3, 512, 512))
	out_grad = torch.randn(out.size())
	out.backward(out_grad)
	print(summary(ddnet, input_size=(3, 512, 512)))







