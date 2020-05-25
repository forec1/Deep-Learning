import src.features.build_features as features
import matplotlib.pyplot as plt
import src.models.resnet as resnet
import torch
import torch.utils.data as data
import src.data.transforms as t
import numpy as np
import math
import torch.nn.functional as F


def cut_img(img, shape):
    h_cut = 320 - shape[1]
    w_cut = 320 - shape[0]
    h_cut_top = math.ceil(h_cut / 2)
    h_cut_btm = h_cut - h_cut_top
    w_cut_left = math.ceil(w_cut / 2)
    w_cut_right = w_cut - w_cut_left
    img = img[:, h_cut_top:, w_cut_left:]
    img = img[:, :img.shape[1] - h_cut_btm, :img.shape[2] - w_cut_right]
    return img


dataset = features.SBD('./data/processed', transform=t.ToTensor(), mode='val')
dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
sample = dataset[4]

model = resnet.resnet50_2d_semseg(num_classes=9)
model.load_state_dict(torch.load('models/resnet50_semseg2.pt', map_location=torch.device('cpu')))
model.eval()

img = sample['image'].cpu()
print(img.shape)
shape = sample['shape'][1], sample['shape'][0]
output = model(torch.unsqueeze(img, dim=0))
output = F.interpolate(output, size=shape, mode='bilinear', align_corners=False)
pred = torch.argmax(output, dim=1)

pred = np.squeeze(pred.detach().numpy(), axis=0)
print(pred.shape)

label = np.squeeze(sample['label'].cpu().numpy())
print(label.shape)

img = np.squeeze(img.numpy()).transpose((1, 2, 0))
print(img.shape)

plt.subplot(1, 3, 1)
plt.imshow(pred)
plt.subplot(1, 3, 2)
plt.imshow(label)
plt.subplot(1, 3, 3)
plt.imshow(img)
plt.show()
