import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random
import math


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToNumpy(object):

    def __call__(self, sample):
        image, labels, shape = sample['image'], sample['region'], sample['shape']
        image = np.array(image)
        labels = np.array(labels)
        return {'image': image, 'region': labels, 'shape': shape}


class Pad(object):

    def __init__(self, max_h, max_w):
        self.max_h = max_h
        self.max_w = max_w

    def __call__(self, sample, img_value=0, lbl_value=10):
        image, labels, shape = sample['image'], sample['region'], sample['shape']
        height = len(image)
        width = len(image[0])
        pad_h = self.max_h - height
        pad_h_top = math.ceil(pad_h / 2)
        pad_h_bottom = pad_h - pad_h_top
        pad_w = self.max_w - width
        pad_w_left = math.ceil(pad_w / 2)
        pad_w_right = pad_w - pad_w_left

        pad_shape2 = ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right))
        pad_shape3 = ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0))

        image = np.pad(image, pad_shape3, 'constant', constant_values=img_value)
        labels = np.pad(labels, pad_shape2, 'constant', constant_values=lbl_value)
        return {'image': image, 'region': labels, 'shape': shape}


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
       Args:
           brightness (float or tuple of float (min, max)): How much to jitter brightness.
               brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
               or the given [min, max]. Should be non negative numbers.
           contrast (float or tuple of float (min, max)): How much to jitter contrast.
               contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
               or the given [min, max]. Should be non negative numbers.
           saturation (float or tuple of float (min, max)): How much to jitter saturation.
               saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
               or the given [min, max]. Should be non negative numbers.
           hue (float or tuple of float (min, max)): How much to jitter hue.
               hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
               Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
       """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        image, labels, shape = sample['image'], sample['region'], sample['shape']
        image = self.color_jitter(image)
        return {'image': image, 'region': labels, 'shape': shape}


class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, labels, shape = sample['image'], sample['region'], sample['shape']

        if random.random() < self.p:
            image = F.vflip(image)
            labels = F.vflip(labels)
        return {'image': image, 'region': labels, 'shape': shape}


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, labels, shape = sample['image'], sample['region'], sample['shape']

        if random.random() < self.p:
            image = F.hflip(image)
            labels = F.hflip(labels)
        return {'image': image, 'region': labels, 'shape': shape}


class ToPILImage(object):

    def __call__(self, sample):
        image, labels, shape = sample['image'], sample['region'], sample['shape']
        image = F.to_pil_image(image, None)
        labels = F.to_pil_image(labels, None)
        return {'image': image, 'region': labels, 'shape': shape}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['region']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        labels = labels[top: top + new_h, left: left + new_w]

        shape = self.output_size
        return {'image': image, 'region': labels, 'shape': shape}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, region, shape = sample['image'], sample['region'], sample['shape']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((2, 0, 1))
        img = img / 255;
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = torch.from_numpy(img).to(device).float()  # mozda treba skalirat na [0.0, 1.0]
        region = torch.from_numpy(region).to(device).long()
        return {'image': img, 'region': region, 'shape': shape}
