"""
ResUnet
ResUnet + dilated conv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
	return nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, \
					  padding=padding, dilation=dilation, bias=False),
			nn.BatchNorm2d(out_channels)
		)

class DoubleConv(nn.Module):

	def __init__(self, in_channels, out_channels):
		"""
		Args:
			in_channels(int): 	the input channels of the layer
			out_channels(int):	the output channels of the layer
		"""
		super(DoubleConv, self).__init__()
		self.conv1 = conv(in_channels, out_channels, kernel_size=3, padding=1)
		self.conv2 = conv(out_channels, out_channels, kernel_size=3, padding=1)


	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		return x


class DilatedConv(nn.Module):

	def __init__(self, in_channels, out_channels):
		"""
		Args:
			in_channels(int): 	the input channels of the layer
			out_channels(int):	the output channels of the layer
		"""
		super(DilatedConv, self).__init__()
		self.conv1 = conv(in_channels, out_channels, 3, padding=1, dilation=1)
		self.conv2 = conv(out_channels, out_channels, 3, padding=2, dilation=2)
		self.conv3 = conv(out_channels, out_channels, 3, padding=4, dilation=4)
		self.conv4 = conv(out_channels, out_channels, 3, padding=8, dilation=8)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		return x


class UnetUp(nn.Module):

	def __init__(self, in_channels, out_channels, is_trconv=True):
		"""
		Args:
			in_channels(int): 	the input channels of the layer
			out_channels(int):	the output channels of the layer
		"""
		super(UnetUp, self).__init__()
		if is_trconv:
			self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, bias=False)
			# self.up_padding = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, output_padding=1, bias=False)
		else:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear')
		self.bn = nn.BatchNorm2d(out_channels)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, up_x, lat_x):
		up_x = F.relu(self.bn(self.up(up_x)))

		up_x = torch.cat([up_x, lat_x], dim=1)
		up_x = self.conv(up_x)

		return up_x


class UnetCenter(nn.Module):

	def __init__(self, in_channels, out_channels, is_dilated=False):
		"""
		Args:
			in_channels(int): 	the input channels of the layer
			out_channels(int):	the output channels of the layer
			is_dilated(bool): 	dilated conv if True else conv
		"""
		super(UnetCenter, self).__init__()
		if is_dilated:
			self.conv = DilatedConv(in_channels, out_channels)
		else:
			self.conv = DoubleConv(in_channels, out_channels)
		self.pool = nn.MaxPool2d(2, 2)

	def forward(self, x):
		x = self.conv(x)
		x = self.pool(x)
		return x


class ResUnet(nn.Module):
	"""
	Resnet + Unet
	"""
	def __init__(self, in_channels=1, num_classes=2, layers=34, pretrained=True,
				is_dilated=False, is_trconv=True, is_gn=False):
		"""
		Input: (3, 512, 512)
		Args:
			in_channels(int): 	the number of channel of image
			num_classes(int):	number of classes,including foreground and background
			layers(int):		the number of layers
			is_dilated(bool): 	dilated conv if True else conv
			is_trconv(bool): 	ConvTranspose2d if True else Upsample
			is_gn(bool);		group normalization if True else batch normalization
		"""
		super(ResUnet, self).__init__()
		# self.resnet = torchvision.models.resnet34(True)
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
						nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
						self.resnet.bn1,
						self.resnet.relu
					)
		self.down2 = self.resnet.layer1 # (64, 512, 512)
		self.down3 = self.resnet.layer2 # (128, 256, 256)
		self.down4 = self.resnet.layer3 # (256, 128, 128)
		self.down5 = self.resnet.layer4 # (512, 64, 64)

		self.center = UnetCenter(self.final_channels, self.final_channels * 2, is_dilated)

		self.up5 = UnetUp(self.final_channels * 2, self.final_channels)
		self.up4 = UnetUp(self.final_channels, self.final_channels // 2)
		self.up3 = UnetUp(self.final_channels // 2, self.final_channels // 4)
		self.up2 = UnetUp(self.final_channels // 4, self.final_channels // 8)

		self.out = nn.Conv2d(self.final_channels // 8, num_classes, 1)


	def forward(self, x):
		x = self.down1(x)
		down2 = self.down2(x)
		down3 = self.down3(down2)
		down4 = self.down4(down3)
		down5 = self.down5(down4)

		x = self.center(down5)

		x = self.up5(x, down5)
		x = self.up4(x, down4)
		x = self.up3(x, down3)
		x = self.up2(x, down2)

		x = F.softmax(self.out(x), dim=1)

		return x


def Res34Unet(in_channels, out_channels, pretrained=True):
	net = ResUnet(in_channels=in_channels, num_classes=out_channels, layers=34, 
				pretrained=pretrained, is_dilated=False, is_trconv=True, is_gn=False)
	return net

def Res50Unet(in_channels, out_channels, pretrained=True):
	net = ResUnet(in_channels=in_channels, num_classes=out_channels, layers=50, 
				pretrained=pretrained, is_dilated=False, is_trconv=True, is_gn=False)
	return net

def Res101Unet(in_channels, out_channels, pretrained=True):
	net = ResUnet(in_channels=in_channels, num_classes=out_channels, layers=101, 
				pretrained=pretrained, is_dilated=False, is_trconv=True, is_gn=False)
	return net



