import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
	return	nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 
							stride=stride, padding=padding, groups=groups, bias=False),
						nn.BatchNorm2d(out_channels))


class FPA(nn.Module):
	"""FPA: Feature Pyramid Attention"""

	def __init__(self, in_channels, out_channels, groups=1):
		"""
		Args:
			in_channels(int): 	the input channels of the layer
			out_channels(int):	the output channels of the layer
			groups(int): 		group convolution
		"""
		super(FPA, self).__init__()
		self.global_pool = nn.Sequential(
								nn.AdaptiveAvgPool2d(1),
								nn.Conv2d(in_channels, out_channels, 1, bias=False),
								nn.BatchNorm2d(out_channels)
							)
		self.conv1 = conv(in_channels, out_channels, 1, stride=1, groups=groups)
		self.conv7_1 = conv(in_channels, in_channels, 7, stride=2, padding=3, groups=groups)
		self.conv7_2 = conv(in_channels, out_channels, 7, stride=1, padding=3, groups=groups)
		self.conv5_1 = conv(in_channels, in_channels, 5, stride=2, padding=2, groups=groups)
		self.conv5_2 = conv(in_channels, out_channels, 5, stride=1, padding=2, groups=groups)
		self.conv3_1 = conv(in_channels, in_channels, 3, stride=2, padding=1, groups=groups)
		self.conv3_2 = conv(in_channels, out_channels, 3, stride=1, padding=1, groups=groups)

	def forward(self, x):
		x_gp = F.interpolate(self.global_pool(x), scale_factor=32, mode='bilinear', align_corners=True)
		x0 = F.relu(self.conv1(x))
		x1 = F.relu(self.conv7_1(x))
		x2 = F.relu(self.conv5_1(x1))
		x3 = F.relu(self.conv3_1(x2))
		x1 = F.relu(self.conv7_2(x1))
		x2 = F.relu(self.conv5_2(x2))
		x3 = F.relu(self.conv3_2(x3))
		x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
		x2 = F.interpolate(x3 + x2, scale_factor=2, mode='bilinear', align_corners=True)
		x1 = F.interpolate(x2 + x1, scale_factor=2, mode='bilinear', align_corners=True)

		x = x0 * x1 + x_gp

		return x


class GAU(nn.Module):
	"""GAU: Global Attention Upsample"""

	def __init__(self, low_in_channels, high_in_channels, groups=1):
		"""
		Args:
			in_channels(int): 	the input channels of the layer
			out_channels(int):	the output channels of the layer
			groups(int): 		group convolution
		"""
		super(GAU, self).__init__()
		self.low_conv = conv(low_in_channels, low_in_channels, 3, padding=1, groups=groups)
		self.high_conv = nn.Sequential(
							nn.AdaptiveAvgPool2d(1),
							nn.Conv2d(high_in_channels, low_in_channels, 1, bias=False, groups=groups),
							nn.BatchNorm2d(low_in_channels)
						)
		self.conv = conv(high_in_channels, low_in_channels, 1)
	
	def forward(self, low_x, high_x):
		low_x = F.relu(self.low_conv(low_x))
		x = F.relu(self.high_conv(high_x))
		x = x * low_x

		high_x = F.relu(self.conv(high_x))
		high_x = F.interpolate(high_x, scale_factor=2, mode='bilinear', align_corners=True)

		x = x + high_x
		return x


class PAN(nn.Module):
	"""docstring for PAN"""

	def __init__(self, in_channels, out_channels, layers=34, groups=1, pretrained=True):
		"""Input: (3, 512, 512)"""

		super(PAN, self).__init__()
		if layers == 34:
			self.resnet = torchvision.models.resnet34(pretrained)
			self.final_channels = 512
		elif layers == 50:
			self.resnet = torchvision.models.resnet50(pretrained)
			self.final_channels = 2048
		else:
			self.resnet = torchvision.models.resnet101(pretrained)
			self.final_channels = 2048

		self.down1 = nn.Sequential(
						nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
						self.resnet.bn1,
						self.resnet.relu
					)
		self.down2 = self.resnet.layer1
		self.down3 = self.resnet.layer2
		self.down4 = self.resnet.layer3
		self.down5 = self.resnet.layer4 

		self.fpa = FPA(self.final_channels, self.final_channels, groups=groups) 

		self.up4 = GAU(self.final_channels // 2, self.final_channels, groups=groups) 
		self.up3 = GAU(self.final_channels // 4, self.final_channels // 2, groups=groups) 
		self.up2 = GAU(self.final_channels // 8, self.final_channels // 4, groups=groups)  

		self.out = conv(self.final_channels // 8, out_channels, 1)

	def forward(self, x):
		x1 = self.down1(x)
		x2 = self.down2(x1)
		x3 = self.down3(x2)
		x4 = self.down4(x3)
		x5 = self.down5(x4)

		x5 = self.fpa(x5)
		x4 = self.up4(x4, x5)
		x3 = self.up3(x3, x4)
		x2 = self.up2(x2, x3)

		x1 = F.relu(self.out(x2))
		x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
		x1 = F.softmax(x1, dim=1)

		return x1


def PAN34(in_channels, out_channels, pretrained=True):
	pan = PAN(in_channels, out_channels, 34, 2, pretrained)
	return pan

def PAN50(in_channels, out_channels, pretrained=True):
	pan = PAN(in_channels, out_channels, 50, 8, pretrained)
	return pan

def PAN101(in_channels, out_channels, pretrained=True):
	pan = PAN(in_channels, out_channels, 101, 8, pretrained)
	return pan