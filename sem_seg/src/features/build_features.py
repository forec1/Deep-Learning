import torch.utils.data as data
import os
import numpy as np
import matplotlib.pyplot as plt

VALID_MODE_VALUES = {'train', 'test', 'val'}


class SBD(data.Dataset):
    """
    Stanford Background Dataset for semantic segmentation
    """

    def __init__(self, data_dir, transform=None, mode='train'):
        if mode not in VALID_MODE_VALUES:
            raise ValueError('SBD __init__: argument \'mode\' must be one of %r.' % VALID_MODE_VALUES)

        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.images_names = sorted(os.listdir(data_dir + '/' + mode + '/images'))
        self.labels_names = sorted(os.listdir(data_dir + '/' + mode + '/labels'))

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir + '/' + self.mode, 'images/' + self.images_names[idx])
        label_path = os.path.join(self.data_dir + '/' + self.mode, 'labels/' + self.labels_names[idx])

        image = plt.imread(image_path)
        label = np.load(label_path)
        shape = 0

        if self.mode != 'train':
            shape_path = os.path.join(self.data_dir + '/' + self.mode, 'shapes/' + self.labels_names[idx])
            shape = np.load(shape_path)

        sample = {'image': image, 'label': label, 'shape': shape}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images_names)
