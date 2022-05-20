import torch.nn as nn
import torch
from torch import autograd


class ConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class FusionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FusionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.conv3 = nn.Conv2d(out_ch, in_ch, 3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x_up, x_down):
        x_up = self.conv1(x_up)
        x_down = self.up(x_down)
        sum = x_up + x_down
        output = self.conv3(sum)
        return sum, output


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class FAUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FAUnet, self).__init__()
        self.Convolution1 = ConvolutionBlock(in_ch, 64)
        self.maxpooling1 = nn.MaxPool2d(2)
        self.Convolution2 = ConvolutionBlock(64, 128)
        self.maxpooling2 = nn.MaxPool2d(2)
        self.Convolution3 = ConvolutionBlock(128, 256)
        self.maxpooling3 = nn.MaxPool2d(2)
        self.Convolution4 = ConvolutionBlock(256, 512)
        self.maxpooling4 = nn.MaxPool2d(2)
        self.Convolution5 = ConvolutionBlock(512, 1024)

        self.Attention5 = cbam_block(1024)

        self.fusion6 = FusionBlock(512, 1024)
        self.maxpooling6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.Convolution6 = ConvolutionBlock(1024, 512)
        self.fusion7 = FusionBlock(256, 1024)
        self.maxpooling7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.Convolution7 = ConvolutionBlock(512, 256)
        self.fusion8 = FusionBlock(128, 1024)
        self.maxpooling8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.Convolution8 = ConvolutionBlock(256, 128)
        self.fusion9 = FusionBlock(64, 1024)
        self.maxpooling9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.Convolution9 = ConvolutionBlock(128, 64)
        self.Convolution10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1 = self.Convolution1(x)
        pool1 = self.maxpooling1(conv1)
        conv2 = self.Convolution2(pool1)
        pool2 = self.maxpooling2(conv2)
        conv3 = self.Convolution3(pool2)
        pool3 = self.maxpooling3(conv3)
        conv4 = self.Convolution4(pool3)
        pool4 = self.maxpooling4(conv4)
        conv5 = self.Convolution5(pool4)
        conv5_att = self.Attention5(conv5)

        sum4, conv4 = self.fusion6(conv4, conv5)
        up_6 = self.maxpooling6(conv5_att)
        merge6 = torch.cat([up_6, conv4], dim=1)
        conv6 = self.Convolution6(merge6)

        sum3, conv3 = self.fusion7(conv3, sum4)
        up_7 = self.maxpooling7(conv6)
        merge7 = torch.cat([up_7, conv3], dim=1)
        conv7 = self.Convolution7(merge7)

        sum2, conv2 = self.fusion8(conv2, sum3)
        up_8 = self.maxpooling8(conv7)
        merge8 = torch.cat([up_8, conv2], dim=1)
        conv8 = self.Convolution8(merge8)

        sum1, conv1 = self.fusion9(conv1, sum2)
        up_9 = self.maxpooling9(conv8)
        merge9 = torch.cat([up_9, conv1], dim=1)
        conv9 = self.Convolution9(merge9)
        conv10 = self.Convolution10(conv9)
        out = nn.Softmax2d()(conv10)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 3, 512, 512)
    net = FAUnet(3, 4)
    print(net(x).shape)
