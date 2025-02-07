import os
import copy
import json
import operator
import numpy as np
from PIL import Image
from os.path import join
from itertools import chain
from scipy.io import loadmat
from collections import defaultdict

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision import datasets

from src.datasets.root_paths import DATA_ROOTS


class VGGFlower(data.Dataset):
    NUM_CLASSES = 102
    FILTER_SIZE = 32
    MULTI_LABEL = False
    NUM_CHANNELS = 3

    def __init__(self, root=DATA_ROOTS['meta_vgg_flower'], train=True, image_transforms=None):
        super().__init__()
        self.dataset = BaseVGGFlower(
            root=root, 
            train=train,
            image_transforms=image_transforms,
        )

    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        _, img_data, label = self.dataset.__getitem__(index)
        _, img2_data, _ = self.dataset.__getitem__(index)
        _, neg_data, _ = self.dataset.__getitem__(neg_index)
        # build this wrapper such that we can return index
        data = [index, img_data.float(), img2_data.float(), 
                neg_data.float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)


class BaseVGGFlower(data.Dataset):

    def __init__(self, root=DATA_ROOTS['meta_vgg_flower'], train=True, image_transforms=None):
        super().__init__()
        self.root = root
        self.train = train
        self.image_transforms = image_transforms
        self.dataset = datasets.Flowers102(
            root,
            split='val',
            download=True,
        )

    def load_images(self):
        rs = np.random.RandomState(42)
        imagelabels_path = os.path.join(self.root, 'imagelabels.mat')
        with open(imagelabels_path, 'rb') as f:
            labels = loadmat(f)['labels'][0]

        all_filepaths = defaultdict(list)
        for i, label in enumerate(labels):
            all_filepaths[label].append(
                os.path.join(self.root, 'jpg', 'image_{:05d}.jpg'.format(i+1)))
        # train test split
        split_filepaths, split_labels = [], []
        for label, paths in all_filepaths.items():
            num = len(paths)
            paths = np.array(paths)
            indexer = np.arange(num)
            rs.shuffle(indexer)
            paths = paths[indexer].tolist()

            if self.train:
                paths = paths[:int(0.8 * num)]
            else:
                paths  = paths[int(0.8 * num):]

            labels = [label] * len(paths)
            split_filepaths.extend(paths)
            split_labels.extend(labels)
        
        return split_filepaths, split_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = int(self.labels[index]) - 1
        image = Image.open(path).convert(mode='RGB')

        if self.image_transforms:
            image = self.image_transforms(image)

        return index, image, label
