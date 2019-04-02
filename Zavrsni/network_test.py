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

outputABCD = network.interleave([outputAB, outputCD], dim=3, transpose=True)
print('outputABCD:')
print(outputABCD)