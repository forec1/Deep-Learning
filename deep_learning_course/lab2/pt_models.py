import torch
from torch import nn
import math


class ConvolutionalModelMNIST(nn.Module):

    def __init__(self, input_shape, in_channels, conv1_width, conv2_width, fc1_width,
                 class_count):
        super(ConvolutionalModelMNIST, self).__init__()

        self.visualizers = []

        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5,
                               stride=1, padding=2, bias=True)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        fc1_in_featuers = ((input_shape[0] // 4)*(input_shape[1] // 4)) * conv2_width
        self.fc1 = nn.Linear(fc1_in_featuers, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc_logits(x)

    def attach(self, visualizer):
        self.visualizers += [visualizer]

    def visualize(self, epoch):
        for visualizer in self.visualizers:
            visualizer.visualize(epoch)


class ConvolutionalModel(nn.Module):

    def __init__(self, input_shape, in_channels, conv_widths, fc_widths, class_count):
        super(ConvolutionalModel, self).__init__()
        self.visualizers = []

        self.conv_layers = nn.Sequential()
        # Constructing convolutioanl layers
        for i in range(len(conv_widths)):
            if i == 0:
                self.conv_layers.add_module('conv%d' % (i), nn.Conv2d(in_channels, conv_widths[i], 5, bias=True, padding=2))
            else:
                self.conv_layers.add_module('conv%d' % (i), nn.Conv2d(conv_widths[i-1], conv_widths[i], 5, bias=True, padding=2))
            self.conv_layers.add_module('relu%d' % (i), nn.ReLU())
            self.conv_layers.add_module('pool%d' % (i), nn.MaxPool2d(3, 2))

        # Calculating number of input features for first fully connected layer
        num_pools = len(conv_widths)
        ww = input_shape[0]
        hh = input_shape[1]
        for _ in range(num_pools):
            ww = math.floor((ww - 3)/2 + 1)
            hh = math.floor((hh - 3)/2 + 1)

        fc1_in_featuers = int(ww * hh * conv_widths[-1])

        self.fc_layers = nn.Sequential()

        # Constructing fully connected layers
        for i in range(len(fc_widths)):
            if i == 0:
                self.fc_layers.add_module('fc%d' % (i), nn.Linear(fc1_in_featuers, fc_widths[i], bias=True))
            else:
                self.fc_layers.add_module('fc%d' % (i), nn.Linear(fc_widths[i-1], fc_widths[i], bias=True))
            self.fc_layers.add_module('relu%d' % (len(conv_widths) + i), nn.ReLU())

        self.fc_layers.add_module('fc_logits', nn.Linear(fc_widths[-1], class_count, bias=True))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        return self.fc_layers(x)

    def attach(self, visualizer):
        self.visualizers += [visualizer]

    def visualize(self, epoch):
        for visualizer in self.visualizers:
            visualizer.visualize(epoch)
