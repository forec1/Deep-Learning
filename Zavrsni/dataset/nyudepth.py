import torch.utils.data as data
import h5py


class NYUDepthV2Dataset(data.Dataset):

    def __init__(self, filepath, transform=None):
        self.transform = transform
        self.file = h5py.File(filepath)
        self.length = self.file['images'].shape[0]

    def __getitem__(self, index):
        # Dimension labels - NxCxWxH

        image = self.file['images'][index]
        depth = self.file['depths'][index]

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.length
