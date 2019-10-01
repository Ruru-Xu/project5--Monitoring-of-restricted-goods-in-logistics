"""CDNet"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #self.gn1 = GroupNorm(out_channels, out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class DilatedConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DilatedConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        return x

class Concatenate(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Concatenate, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

class Out(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.log_softmax(x, dim=1)
        return x


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias

class CDNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels(int):
            num_classes(int):
        """
        super(CDNet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.dilated_conv1 = DilatedConv(in_channels, 64)
        self.cat1 = Concatenate(128, 64)

        self.conv2 = DoubleConv(64, 128)
        self.dilated_conv2 = DilatedConv(64, 128)
        self.cat2 = Concatenate(256, 128)

        self.conv3 = DoubleConv(128, 256)
        self.dilated_conv3 = DilatedConv(128, 256)
        self.cat3 = Concatenate(512, 64)

        self.out = Out(256, num_classes)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.dilated_conv1(x)
        cat1 = self.cat1(x1, x2)

        x3 = self.conv2(cat1)
        x4 = self.dilated_conv2(cat1)
        cat2 = self.cat2(x3, x4)

        x5 = self.conv3(cat2)
        x6 = self.dilated_conv3(cat2)
        cat3 = self.cat3(x5, x6)

        x = torch.cat([cat1, cat2, cat3], dim=1)
        x = self.out(x)

        return x

# cdnet = CDNet(3, 21)
# out = cdnet(torch.randn(2, 3, 512, 512))
# out_grad = torch.randn(out.size())
# out.backward(out_grad)
#summary(CDNet(3, 21).cuda(), input_size=(3, 512, 512))
# summary(CDNet(3, 21), input_size=(3, 512, 512))






