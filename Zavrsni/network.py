import torch
import torch.nn as nn
import torch.nn.functional as F


def fast_up_convolution(input_data, out_channels, ReLU=False):

    # Dimension labels - HxW
    # Input dimensions - NxCxHxW

    in_channels = input_data[1]

    # A Convolution - 3x3
    convA = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1))
    outputA = convA(input_data)

    # B Convolution - 3x2
    convB = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2), stride=1)
    input_dataB = F.pad(input_data, (1, 0, 1, 1))
    outputB = convB(input_dataB)

    # C Convolution - 2x3
    convC = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3), stride=1)
    input_dataC = F.pad(input_data, (1, 1, 1, 0))
    outputC = convC(input_dataC)

    # D Convolution - 2x2
    convD = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), stride=1)
    input_dataD = F.pad(input_data, (1, 0, 1, 0))
    outputD = convD(input_dataD)

    # Interleaving ABCD feature maps
    outputAB = interleave([outputA, outputB], dim=2)
    outputCD = interleave([outputC, outputD], dim=2)
    outputABCD = interleave([outputAB, outputCD], dim=3, transpose=True)

    if ReLU:
        outputABCD = F.relu(outputABCD)
    
    return outputABCD


def interleave(tensors, dim, transpose=False):
    shape = list(tensors[0].shape)[1:]
    reshape = [-1] + shape
    reshape[dim] *= len(tensors)
    result = torch.reshape(torch.stack(tensors, dim=dim + 1), reshape)
    if transpose:
        result = result.transpose(2, 3)
    return result
