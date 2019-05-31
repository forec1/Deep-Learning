import os
from skimage import io
import matplotlib.pyplot as plt
import utils
import h5py
import operator


def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


sceneDirs = sorted(os.listdir('data/scene_dirs'))

file = open('data/train_scenes.txt')
train_scenes = file.readlines()
train_scenes = [name[:-1] for name in train_scenes]
file.close()

file = open('data/test_scenes.txt')
test_scenes = file.readlines()
test_scenes = [name[:-1] for name in test_scenes]
file.close()


for sceneDir in sceneDirs:

    t = ''
    if sceneDir in train_scenes:
        t = 'train'
    elif sceneDir in test_scenes:
        continue
    else:
        continue

    fpath = os.path.join('data/scene_dirs', sceneDir)
    result = utils.get_synced_frames(fpath)

    res = {'D': [], 'R': []}

    for key, value in result.items():
        for fname in value:
            res[key].append(os.path.join(fpath, fname))

    dist = int((len(res['D']) / 50))

    print(len(res['D']), len(res['R']))

    grpname = sceneDir

    dset_path = os.path.join(t, grpname)

    data = h5py.File('data/nyudP')
    data[t].create_group(grpname)
    print('Created group ' + grpname)
    data[dset_path].create_dataset('images', (50, 228, 304, 3), dtype='uint8')
    data[dset_path].create_dataset('depths', (50, 480, 640))

    cnt = 0
    for i in range(len(res['R'])):
        if cnt == 50: break
        if i % dist == 0:
            fname = res['R'][i]
            img = io.imread(fname)
            img = img[::2, ::2]                 # down-sample to 1/2 resolution
            img = cropND(img, (228, 304, 3))    # center crop
            imgdset_path = os.path.join(dset_path, 'images')
            data[imgdset_path][cnt] = img
            cnt += 1

    cnt = 0
    scale = 2**16 - 1
    for i in range(len(res['D'])):
        if cnt == 50: break
        if i % dist == 0:
            fname = res['D'][i]
            depth = io.imread(fname)
            depthdset_path = os.path.join(dset_path, 'depths')
            data[depthdset_path][cnt] = (depth / scale) * 10
            cnt += 1


    plt.subplot(1, 2, 1)
    plt.imshow(data[imgdset_path][49])
    plt.subplot(1, 2, 2)
    plt.imshow(data[depthdset_path][49], cmap='jet')
    plt.show()


