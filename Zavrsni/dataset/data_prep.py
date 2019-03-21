import numpy as np
import math
import os
from skimage import io
from random import shuffle
from dataset import transforms as mytransforms


def __read_lines(root_dir, dataset_info_file):
    with open(os.path.join(root_dir, dataset_info_file)) as f:
        images = [line.strip() for line in f]
    return images


def __save_padded(root_dir, images_info, data_type):
    assert data_type == 'train' or data_type == 'val' or data_type == 'test'
    length = len(images_info)
    for idx in range(length):
        images_dir = os.path.join(root_dir, 'images')
        img_name = os.path.join(images_dir, images_info[idx].split()[0] + '.jpg')
        img = io.imread(img_name)
        img = __pad(img, value=0)
        padded_images_dir = os.path.join(root_dir, 'padded_images/' + data_type)
        padded_image_name = os.path.join(padded_images_dir, images_info[idx].split()[0] + '.jpg')
        io.imsave(padded_image_name, img)
        if idx % 100 == 0:
            print('Saved images: {} of {}'.format(idx, length))


def __save_labels(root_dir, images_info, data_type):
    assert data_type == 'train' or data_type == 'val' or data_type == 'test'
    length = len(images_info)
    for idx in range(length):
        label_region = __load_label_txt(root_dir, images_info, idx)
        label_region += 1  # unknown class is -1 and now is 0 so it can be calculated
        label_region = __pad(label_region, value=10)
        img_labels_dir = os.path.join(root_dir, 'padded_images/' + data_type)
        lbl_file_name = os.path.join(img_labels_dir, images_info[idx].split()[0])
        np.save(lbl_file_name, label_region)
        if idx % 100 == 0:
            print('Saved labels: {} od {}'.format(idx, length))


def __load_label_txt(root_dir, images_info, idx):
    labels_dir = os.path.join(root_dir, 'labels')
    file_name = os.path.join(labels_dir, images_info[idx].split()[0] + '.regions.txt')
    label_region = np.genfromtxt(file_name, dtype='uint8')
    return label_region


def __pad(image, value=0):
    max_h, max_w = 320, 320
    height = len(image)
    width = len(image[0])
    dim = len(image.shape)
    pad_h = max_h - height
    pad_h_top = math.ceil(pad_h / 2)
    pad_h_bottom = pad_h - pad_h_top
    pad_w = max_w - width
    pad_w_left = math.ceil(pad_w / 2)
    pad_w_right = pad_w - pad_w_left

    if dim == 2:
        pad_shape = ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right))
    elif dim == 3:
        pad_shape = ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0))

    image = np.pad(image, pad_shape, 'constant', constant_values=value)
    return image


def data_augmentation(root_dir, images_info):
    assert os.path.isdir(os.path.join(root_dir, 'padded_images/train'))
    data_dir = os.path.join(root_dir, 'padded_images/train')
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'labels')
    train_imgs_info_f = open(root_dir + '/train_imgs', 'w+')
    cnt = 0
    for i in range(9):
        for image_info in images_info:
            image_info_split = image_info.split()
            if i == 0:
                train_imgs_info_f.write(image_info_split[0] + '\n')

            image_name = image_info_split[0] + '.jpg'
            image = io.imread(os.path.join(images_dir, image_name))

            labels_name = os.path.join(labels_dir, image_info_split[0] + '.regions.txt')
            labels = np.genfromtxt(labels_name, dtype='uint8')
            labels += 1

            shape = image.shape[:2]

            _sample = __transform({'image': image, 'region': labels, 'shape': shape}, image_info)
            __save(_sample, data_dir, image_info_split[0], str(i))
            train_imgs_info_f.write(image_info_split[0] + str(i) + '\n')
            if cnt % 500 == 0:
                print('Saved {}'.format(cnt))
            cnt += 1
    print('Total images saved: {}'.format(cnt))
    train_imgs_info_f.close()


def __save(sample, save_dir, img_name, img_id):
    image, labels = sample['image'], sample['region']
    img_path = os.path.join(save_dir, img_name + img_id + '.jpg')
    lbl_path = os.path.join(save_dir, img_name + img_id)
    io.imsave(img_path, image)
    np.save(lbl_path, labels)


def __transform(sample, image_info):
    image_info_split = image_info.split()
    h, w = int(image_info_split[2]), int(image_info_split[1])

    transform = mytransforms.Compose([mytransforms.RandomCrop((int(h*0.8), int(w*0.8))),
                                      mytransforms.ToPILImage(),
                                      mytransforms.RandomHorizontalFlip(p=0.5),
                                      mytransforms.RandomVerticalFlip(p=0.5),
                                      mytransforms.ColorJitter(brightness=0.2),
                                      mytransforms.ToNumpy(),
                                      mytransforms.Pad(320, 320)])
    return transform(sample)


def separate(root_dir, train_imgs, val_imgs, test_imgs):
    assert not os.path.isdir(os.path.join(root_dir, 'padded_images/train'))
    assert not os.path.isdir(os.path.join(root_dir, 'padded_images/val'))
    assert not os.path.isdir(os.path.join(root_dir, 'padded_images/test'))
    os.makedirs(os.path.join(root_dir, 'padded_images/train'))
    os.makedirs(os.path.join(root_dir, 'padded_images/val'))
    os.makedirs(os.path.join(root_dir, 'padded_images/test'))
    __save_padded(root_dir, train_imgs, 'train')
    __save_labels(root_dir, train_imgs, 'train')
    __save_padded(root_dir, val_imgs, 'val')
    __save_labels(root_dir, val_imgs, 'val')
    __save_padded(root_dir, test_imgs, 'test')
    __save_labels(root_dir, test_imgs, 'test')


def data_separation(root_dir, dataset_info_file, percentage):
    images_info = __read_lines(root_dir, dataset_info_file)
    shuffle(images_info)
    n_images = len(images_info)
    train_p, val_p, test_p = percentage
    n_train = math.floor(n_images * train_p)
    n_val = math.floor(n_images * val_p)
    train_imgs = images_info[:n_train]
    val_imgs = images_info[n_train: n_val + n_train]
    test_imgs = images_info[n_train + n_val:]
    separate(root_dir, train_imgs, val_imgs, test_imgs)
    __geninfofile(os.path.join(root_dir, 'val_imgs'), val_imgs)
    __geninfofile(os.path.join(root_dir, 'test_imgs'), test_imgs)
    return train_imgs, val_imgs, test_imgs


def __geninfofile(save_dir, img_list):
    f = open(save_dir, 'w+')
    for img in img_list:
        f.write(img + '\n')
    f.close()


if __name__ == '__main__':
    train, val, test = data_separation('./data', 'horizons.txt', (0.64, 0.16, 0.20))
    data_augmentation('./data', train)
