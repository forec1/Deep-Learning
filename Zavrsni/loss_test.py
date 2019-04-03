import loss
import torch

berHu = loss.ReverseHuberLoss()
a = torch.tensor([[[1., 4., 10.]]], requires_grad=True)
b = torch.tensor([[[2., 6., 4.]]], requires_grad=True)
l = berHu(a, b)
print(l)
