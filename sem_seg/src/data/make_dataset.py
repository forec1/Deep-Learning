# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
import os
from skimage import io
import numpy as np
import src.data.transforms as T


def read_info_file(input_filepath):
    filepath = os.path.join(input_filepath, 'horizons.txt')
    with open(filepath) as f:
        images_info = [line.strip() for line in f]
    return images_info


def get_shape(images_info, image_name):
    image_info = next(image_info for image_info in images_info if image_name in image_info)
    return np.fromstring(image_info[8:16].strip(), dtype=int, sep=' ')


def data_augmentation(transforms, iters, images, labels, input_filepath, output_filepath, set_type, save_shape=False):
    input_images_filepath = os.path.join(input_filepath, 'images')
    input_labels_filepath = os.path.join(input_filepath, 'labels')

    output_images_filepath = os.path.join(output_filepath, set_type + '/images')
    output_labels_filepath = os.path.join(output_filepath, set_type + '/labels')

    if not os.path.exists(output_images_filepath):
        os.makedirs(output_images_filepath)
    if not os.path.exists(output_labels_filepath):
        os.makedirs(output_labels_filepath)

    if save_shape:
        images_info = read_info_file(input_filepath)
        output_shape_filepath = os.path.join(output_filepath, set_type + '/shapes')
        if not os.path.exists(output_shape_filepath):
            os.makedirs(output_shape_filepath)

    for i in range(iters):
        for image_name, label_name in zip(images, labels):
            image_path = os.path.join(input_images_filepath, image_name)
            label_path = os.path.join(input_labels_filepath, label_name)

            image = io.imread(image_path)
            label = np.genfromtxt(label_path).astype(np.uint8) + 1

            sample = {'image': image, 'label': label, 'shape': None}
            if transforms is not None:
                sample = transforms(sample)

            suffix_lbl = '.npy' if i == 0 else str(i) + '.npy'
            suffix_img = '.jpg' if i == 0 else str(i) + '.jpg'

            label_name = label_name[:7] + suffix_lbl
            image_name = image_name[:-4] + suffix_img

            if save_shape and i == 0:
                shape = get_shape(images_info, image_name[:-4])
                np.save(os.path.join(output_shape_filepath, label_name), shape)

            io.imsave(os.path.join(output_images_filepath, image_name), sample['image'])
            np.save(os.path.join(output_labels_filepath, label_name), sample['label'].astype(np.uint8))


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_images_filepath = os.path.join(input_filepath, 'images')
    input_labels_filepath = os.path.join(input_filepath, 'labels')

    # Loading raw data
    images = sorted(os.listdir(input_images_filepath))
    labels = sorted([label for label in os.listdir(input_labels_filepath) if 'region' in label])

    # Separating them into train, test and validation sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=7)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.16, random_state=7)

    # Saving test and validation images (not doing data augmentation -> iters=1, transform=None)
    data_augmentation(None, 1, test_images, test_labels, input_filepath,
                      output_filepath, 'test', save_shape=True)
    data_augmentation(None, 1, val_images, val_labels, input_filepath,
                      output_filepath, 'val', save_shape=True)

    # Data augmentation on train set
    pipe = T.Compose([T.RandomCrop(0.8),
                      T.ToPILImage(),
                      T.RandomHorizontalFlip(p=0.5),
                      T.RandomVerticalFlip(p=0.5),
                      T.ColorJitter(brightness=0.2),
                      T.ToNumpy(),
                      T.Pad(320, 320)])

    data_augmentation(pipe, 9, train_images, train_labels, input_filepath, output_filepath, 'train')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
