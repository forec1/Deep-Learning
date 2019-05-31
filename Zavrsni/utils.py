from datetime import date
import os, re
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np


def test_plot(data, group):
    path = 'train/' + group
    for i in range(0, 50, 3):
        plt.subplot(1, 2, 1)
        plt.imshow(data[path+'/images'][i])
        plt.subplot(1, 2, 2)
        plt.imshow(data[path + '/depths'][i], cmap='jet')
        plt.show()


def get_timestamp_from_filename(filename):
    parts = filename.split('-')
    millis = float(parts[1])
    unixEpoch = date.toordinal(date(1969, 12, 31)) + 365
    time = millis / 86400 + unixEpoch
    return time, millis


def get_synced_frames(scenes_dir):

    rgbImages = []
    frameList = {'D': [], 'R': []}

    files = [name for name in os.listdir(scenes_dir) if not re.match(r'^a-*', name)]
    files = sorted(files)

    numDepth = 0
    numRgb = 0

    for i in range(len(files)):
        if re.match(r'^d-*', files[i]):
            numDepth += 1
            frameList['D'].append(files[i])
        elif re.match(r'^r-*', files[i]):
            numRgb += 1
            rgbImages.append(files[i])

    print('Found %d depth, %d rgb images.' % (numDepth, numRgb))

    jj = 0

    for ii in range(numDepth):
        print('Matching depth image %d/%d' % (ii+1, numDepth))

        timePartsDepth = frameList['D'][ii].split('-')
        timePartsRgb = rgbImages[jj].split('-')

        tDepth = Decimal(timePartsDepth[1])
        tRgb = Decimal(timePartsRgb[1])

        tDiff = abs(tDepth-tRgb)

        while jj < numRgb-1:
            timePartsRgb = rgbImages[jj+1].split('-')
            tRgb = Decimal(timePartsRgb[1])

            tmpDiff = abs(tDepth-tRgb)
            if tmpDiff > tDiff:
                break;
            tDiff = tmpDiff

            jj += 1

        print('Matched depth %d to rgb %d.' % (ii+1, jj+1))

        frameList['R'].append(rgbImages[jj])

    print()
    return frameList


def test_plot_sample(sample):
    plt.subplot(1, 2, 1)
    plt.imshow(sample['image'])
    plt.subplot(1, 2, 2)
    plt.imshow(sample['depth'], cmap='jet')
    plt.show()


def to_numpy(sample):
    sample['image'] = sample['image'].numpy().transpose((1,2,0))
    sample['depth'] = sample['depth'].numpy()
    sample['depth'] = np.reshape(sample['depth'], (480, 640))

















