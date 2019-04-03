import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet


class MyNet(nn.Module):
    """
        This net builds upon ResNet-50. Last fully-connected layer, which was part of the
        original ResNet-50 architecture, is replaced with fast up-sampling blocks, yielding
        an output of roughly half the input resolution
    """
    def __init__(self, block, n_upsampling_blocks=4):
        super(MyNet, self).__init__()

        # This model builds upon ResNet-50, last (fully-connected) layer is replaced
        self.resnet50 = resnet.resnet50(last_layer=False, pretrained=True)

        self.in_channels = 2048
        out_channels = int(self.in_channels / 2)

        # Converting output of ResNet-50 from Nx2048x10x8 to Nx1024x10x8 and applying batch normalization
        self.entrance = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(out_channels)
        )

        # Creating fast up-sampling blocks
        self.in_channels = out_channels
        self.upsamling_layers = self._make_upsample(block, n_upsampling_blocks)

        # Last convolutional layer
        self.conv3x3 = nn.Conv2d(self.in_channels, 1, kernel_size=(3, 3), stride=1, padding=(1, 1))

    def _make_upsample(self, block, n_upsampling_blocks):
        layers = []

        out_channels = int(self.in_channels / 2)
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels

        for _ in range(1, n_upsampling_blocks):
            out_channels = int(self.in_channels / 2)
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet50(x)

        x = self.entrance(x)
        x = self.upsamling_layers(x)
        x = self.conv3x3(x)
        x = F.relu(x)

        return x


class FastUpConvolution(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FastUpConvolution, self).__init__()

        # zero padding - adds 2 to W and 2 to H
        self.convA = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.convB = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2), stride=1)
        self.convC = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3), stride=1)
        self.convD = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), stride=1)

    def forward(self, input_data, apply_relu=False):
        # Dimension labels - HxW
        # Input dimensions - NxCxHxW

        # Convolution with A part of filter
        outputA = self.convA(input_data)

        # Convolution with B part of filter
        input_dataB = F.pad(input_data, (1, 0, 1, 1))  # zero padding - adds 1 to W and 2 to H
        outputB = self.convB(input_dataB)

        # Convolution with C part of filter
        input_dataC = F.pad(input_data, (1, 1, 1, 0))  # zero padding - adds 2 to W and 1 to H
        outputC = self.convC(input_dataC)

        # Convolution with D part of filter
        input_dataD = F.pad(input_data, (1, 0, 1, 0))  # zero padding - adds 1 to W and 1 to H
        outputD = self.convD(input_dataD)

        # Interleaving ABCD feature maps
        outputAB = interleave([outputA, outputB], dim=2)  # dim=2 -> H
        outputCD = interleave([outputC, outputD], dim=2)
        outputABCD = interleave([outputAB, outputCD], dim=3)  # dim=3 -> W

        if apply_relu:
            outputABCD = F.relu(outputABCD)

        return outputABCD


class FastUpProjection(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FastUpProjection, self).__init__()

        # Adds padding to the input data on the left and right to keep same dimensions
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.fast_up_conv_up = FastUpConvolution(in_channels, out_channels)

        self.fast_up_conv_down = FastUpConvolution(in_channels, out_channels)

    def forward(self, input_data):
        up_branch = self.fast_up_conv_up(input_data, apply_relu=True)
        up_branch = self.conv3x3(up_branch)

        down_branch = self.fast_up_conv_down(input_data, apply_relu=False)

        result = up_branch + down_branch
        return F.relu(result)


def interleave(feature_maps, dim):
    reshape = [*feature_maps[0].shape]
    reshape[dim] = len(feature_maps) * reshape[dim]
    result = torch.reshape(torch.stack(feature_maps, dim=dim + 1), reshape)
    return result


def create_mynet():
    return MyNet(FastUpProjection)