import torch
import mynn

at = torch.zeros(1, 1, 4, 4)
bt = torch.ones(1, 1, 4, 4)
ct = at + 2
dt = bt + 2

outputAB = mynn.interleave([at, bt], dim=2)
print('outputAB:')
print(outputAB)

outputCD = mynn.interleave([ct, dt], dim=2)
print('outputCD:')
print(outputCD)

outputABCD = mynn.interleave([outputAB, outputCD], dim=3)
print('outputABCD:')
print(outputABCD)

test_tensor = torch.ones((1, 256, 40, 32))  # NxCxHxW
fast_up_conv = mynn.FastUpConvolution(256, 128)
test_fpu_result = fast_up_conv(test_tensor)
print('test result: ', test_fpu_result.shape)
fast_up_proj = mynn.FastUpProjection(256, 128)
test_fpp_result = fast_up_proj(test_tensor)
print('test fpp result', test_fpp_result.shape)

test_tensor = torch.ones((2, 3, 304, 228))
net = mynn.create_mynet()
test_result = net(test_tensor)
print(test_result.shape)
