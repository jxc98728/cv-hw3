import os
import sys
import pickle
import struct

import cv2
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple

def load_mnist_images(path, num=60000):
    
    with open(path, 'rb') as f:
        buf = f.read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, buf, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(num_rows * num_cols) + 'B'

    if num <= num_images:
        num_images = num

    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        im = struct.unpack_from(fmt_image, buf, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return np.asarray(images).astype(np.uint8)

def load_mnist_labels(path, num=60000):
    with open(path, 'rb') as f:
        fb_data = f.read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_labels = struct.unpack_from(fmt_header, fb_data, offset)
    offset += struct.calcsize(fmt_header)
    labels = []

    if num <= num_labels:
        num_labels = num

    fmt_label = '>B'
    
    for i in range(num_labels):
        labels.append(struct.unpack_from(fmt_label, fb_data, offset)[0])
        offset += struct.calcsize(fmt_label)
    return np.asarray(labels).astype(np.uint8)


class MyMNIST(Dataset):

    training_file = 'train.pt'
    test_file = 'test.pt'

    def __init__(self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        
        self.train = train
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(root, data_file))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MyCIFAR10(Dataset):

    train_list_1 = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list_1 = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    train_list_2 = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list_2 = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        self.root = root
        self.train = train

        if self.train:
            downloaded_list_1 = self.train_list_1
            downloaded_list_2 = self.train_list_2
        else:
            downloaded_list_1 = self.test_list_1
            downloaded_list_2 = self.test_list_2

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list_1:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        for file_name, checksum in downloaded_list_2:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                labels = np.asarray(entry['coarse_labels'])
                images = entry['data']
                people_indices = np.where(labels == 14)[0]
                for i in people_indices:
                    self.data.append(images[i])
                    self.targets.append(10)
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.transform = transform
        self.target_transform = target_transform
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def preprocess():
    root = 'data/mymnist/'

    train_images_path = 'data/mnist/MNIST/raw/train-images.idx3-ubyte'
    train_labels_path = 'data/mnist/MNIST/raw/train-labels.idx1-ubyte'
 
    test_images_path = 'data/mnist/MNIST/raw/t10k-images.idx3-ubyte'
    test_labels_path = 'data/mnist/MNIST/raw/t10k-labels.idx1-ubyte'

    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)

    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    indices_zero = np.where(train_labels == 0)[0]
    indices_one = np.where(train_labels == 1)[0]
    for i in indices_zero:
        j = np.random.choice(indices_one)
        train_images[i] = np.maximum(np.rot90(train_images[i]), train_images[j])
    i = np.random.choice(indices_zero)
    cv2.imwrite('./images/train_zero.jpg', train_images[i])
    
    
    indices_zero = np.where(test_labels == 0)[0]
    indices_one = np.where(test_labels == 1)[0]
    for i in indices_zero:
        j = np.random.choice(indices_one)
        test_images[i] = np.maximum(np.rot90(test_images[i]), test_images[j])
    i = np.random.choice(indices_zero)
    cv2.imwrite('./images/test_zero.jpg', test_images[i])

    torch.save((train_images, train_labels), os.path.join(root, 'train.pt'))
    torch.save((test_images, test_labels), os.path.join(root, 'test.pt'))
    return None

# if __name__ == "__main__":
#     preprocess()