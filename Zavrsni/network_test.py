import torch
import network

at = torch.zeros(1, 1, 4, 4)
bt = torch.ones(1, 1, 4, 4)
ct = at + 2
dt = bt + 2

outputAB = network.interleave([at, bt], dim=2)
print('outputAB:')
print(outputAB)

outputCD = network.interleave([ct, dt], dim=2)
print('outputCD:')
print(outputCD)

outputABCD = network.interleave([outputAB, outputCD], dim=3)
print('outputABCD:')
print(outputABCD)

test_tensor = torch.ones((1, 256, 40, 32))  # NxCxHxW
test_result = network.fast_up_convolution(test_tensor, 128)
print('test result: ', test_result.shape)
