import numpy as np
from skimage import io
import os
import click
import logging
import src.data.transforms as transforms

#########################################################################
# Skripta koja mi je posluzila kao tutorial za naredbu make i Makefiles #
#########################################################################

MAX_HEIGHT = 320
MAX_WIDTH = 320

pad_transform = transforms.Pad(MAX_HEIGHT, MAX_WIDTH)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def pad(input_filepath, output_filepath):

    logger = logging.getLogger(__name__)
    logger.info('padding raw images')

    input_images_filepath = os.path.join(input_filepath, 'images')
    input_labels_filepath = os.path.join(input_filepath, 'labels')
    output_images_filepath = os.path.join(output_filepath, 'images')
    output_labels_filepath = os.path.join(output_filepath, 'labels')

    if not os.path.exists(output_images_filepath):
        os.makedirs(output_images_filepath)
    if not os.path.exists(output_labels_filepath):
        os.makedirs(output_labels_filepath)

    images = os.listdir(input_images_filepath)
    labels = [label for label in os.listdir(input_labels_filepath) if 'region' in label]

    for image_name, label_name in zip(images, labels):
        image_path = os.path.join(input_images_filepath, image_name)
        label_path = os.path.join(input_labels_filepath, label_name)

        image = io.imread(image_path)
        label = np.genfromtxt(label_path)
        sample = pad_transform({'image': image, 'label': label, 'shape': None})

        label_name = label_name[:-3] + 'npy'
        io.imsave(os.path.join(output_images_filepath, image_name), sample['image'])
        np.save(os.path.join(output_labels_filepath, label_name), sample['label'].astype(np.uint8))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    pad()
