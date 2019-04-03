import torch


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Transpose(object):
    """
        Converts the shape of the image form NxCxWxH to NxCxHxW and
        the shape of the depth map from NxWxH to NxHxW
    """
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = torch.transpose(image, 1, 2)
        depth = torch.transpose(depth, 0, 1)
        return {'image': image, 'depth': depth}


class ToTensor(object):

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = torch.tensor(image, dtype=torch.float32)
        depth = torch.tensor(depth, dtype=torch.float32)
        return {'image': image, 'depth': depth}