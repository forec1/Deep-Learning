import torch.utils.data as data
import os
from skimage import io
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class StanfordBackgroundDataset(data.Dataset):
    """
    Stanford background data set.
    """

    def __init__(self, root_dir, dataset_info_file, transform=None, mode='train'):
        """
        Args:
             root_dir (string): Directory with images and labels directories.
             dataset_info_file (string): Name of the dataset info file.
             transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_info_file = dataset_info_file
        self.dataset_info = self._read_dataset_info()
        self.images_names = self.__read_lines()
        self.length = self.__get_len()
        # self.length = 100
        self.max_h, self.max_w = 320, 320
#        self.samples = self.__readitems()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        images_info_split = self.images_names[idx].split()
        images_dir = os.path.join(self.root_dir, 'padded_images/' + self.mode)
        img_name = os.path.join(images_dir, images_info_split[0] + '.jpg')
        img = io.imread(img_name)
        label_region = self.load_label(idx)
        shape = 0
        if self.mode != 'train':
            shape = int(images_info_split[2]), int(images_info_split[1])
        sample = {'image': img, 'region': label_region, 'shape': shape}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def _read_dataset_info(self):
        file_name = os.path.join(self.root_dir, self.dataset_info_file)
        with open(file_name) as f:
            info = [line.strip() for line in f]
        return info

    def __get_len(self):
        DIR = self.root_dir + '/padded_images/' + self.mode
        length = int(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) / 2)
        return length
    '''
    def __readitems(self):
        samples = []
        for idx in range(self.length):
            images_info_split = self.images_info[idx].split()
            images_dir = os.path.join(self.root_dir, 'padded_images/' + self.mode)
            img_name = os.path.join(images_dir, images_info_split[0] + '.jpg')
            img = io.imread(img_name)
            label_region = self.load_label(idx)

            sample = {'image': img, 'region': label_region, 'shape': 0}

            if self.transform:
                sample = self.transform(sample)

            samples.append(sample)
            if idx % 100 == 0:
                print('Loaded {} of {}'.format(idx, self.length))
        return samples
    '''
    def load_label(self, idx):
        img_labels_dir = os.path.join(self.root_dir, 'padded_images/' + self.mode)
        file_name = os.path.join(img_labels_dir, self.images_names[idx].split()[0] + '.npy')
        label_region = np.load(file_name)
        return label_region

    def __read_lines(self):
        with open(os.path.join(self.root_dir, self.mode + '_imgs')) as f:
            images = [line.strip() for line in f]
        return images

    def get_max_dim(self):
        return 320, 320
