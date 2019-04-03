import dataset.nyudepth as nyu
import torch.utils.data as data
import matplotlib.pyplot as plt

dataset = nyu.NYUDepthV2Dataset('./data/nyu_depth_v2_labeled.mat')
dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True)

en = enumerate(dataloader)

i, sample = next(en)
img = sample['image'][0].numpy().transpose((2, 1, 0))
depth = sample['depth'][0].numpy().transpose()

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(depth, cmap='jet')

img = sample['image'][1].numpy().transpose((2, 1, 0))
depth = sample['depth'][1].numpy().transpose()

plt.subplot(2, 2, 3)
plt.imshow(img)
plt.subplot(2, 2, 4)
plt.imshow(depth, cmap='jet')
plt.show()