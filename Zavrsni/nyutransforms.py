import torch
import random
import torchvision.transforms.functional as F
import numpy as np
import torchvision
import PIL
import numbers


class CenterCrop():

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = np.array(F.center_crop(F.to_pil_image(image), (228, 304)))
        return {'image': image, 'depth': depth}


class Downsample():

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image[::2, ::2]  # down-sample to 1/2 resolution
        return {'image': image, 'depth': depth}


class Transpose(object):

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = np.transpose(image, (2, 1, 0))
        depth = np.transpose(depth, (1, 0))
        return {'image': image, 'depth': depth}


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):

    def __call__(self, sample):

        image, depth = sample['image'], sample['depth']
        image = F.to_tensor(image)
        depth = F.to_tensor(depth)
        return {'image': image, 'depth': depth}


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if random.random() < self.p:
            image = F.hflip(image)
            depth = F.hflip(depth)

        return {'image': image, 'depth': depth}


class ToPILImage(object):

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = F.to_pil_image(image)
        depth = F.to_pil_image(depth)
        return {'image': image, 'depth': depth}


class ToNumpy(object):

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = np.array(image)
        depth = np.array(depth)
        return {'image': image, 'depth': depth}


class Scale(object):

    def __init__(self, smin, smax):
        self.smin = smin
        self.smax = smax

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        s = self.smin + (random.random() * (self.smax - self.smin))
        image = F.resize(image, (int(image.size[1]*s), int(image.size[0]*s)), PIL.Image.BILINEAR)
        depth = F.resize(depth, (int(depth.size[1]*s), int(depth.size[0]*s)), PIL.Image.BILINEAR)
        depth = PIL.Image.eval(depth, lambda x: x * (1/s))
        return {'image': image, 'depth': depth}


class RandomRotation(object):

    def __init__(self, degrees):
        assert isinstance(degrees, tuple) or isinstance(degrees, int) or \
                isinstance(degrees, float)

        if isinstance(degrees, tuple):
            self.angle_min = degrees[0]
            self.angle_max = degrees[1]
        else:
            self.angle_min = -degrees
            self.angle_max = degrees

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        angle = self.angle_min + (random.random() * (self.angle_max - self.angle_min))
        image = F.rotate(image, angle)
        depth = F.rotate(depth, angle)
        return {'image': image, 'depth': depth}


class ColorTransformation(object):

    def __init__(self, c):
        assert isinstance(c, tuple) or isinstance(c, int) or \
               isinstance(c, float)

        if isinstance(c, tuple):
            self.c_min = c[0]
            self.c_max = c[1]
        else:
            self.c_min = -c
            self.c_max = c

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        for i in range(3):
            c = self.c_min + (random.random() * (self.c_max - self.c_min))
            image[i] = torch.where(image[i]*c > 1.0, torch.tensor(1.0), image[i]*c)
        return {'image': image, 'depth': depth}


class RandomCrop(object):

    def __init__(self, img_size, depth_size):
        self.img_size = img_size
        self.depth_size = depth_size

    @staticmethod
    def get_params(img, output_size):

        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):

        image, depth = sample['image'], sample['depth']
        dw, dh = depth.size
        iw, ih = image.size
        i, j, h, w = self.get_params(image, self.img_size)
        ratio = dh / (ih+24), dw / (iw+24)
        di = int(i * ratio[0])
        dj = int(j * ratio[1])

        image = F.crop(image, i, j, h, w)
        depth = F.crop(depth, di, dj, self.depth_size[0], self.depth_size[1])
        return {'image': image, 'depth': depth}
