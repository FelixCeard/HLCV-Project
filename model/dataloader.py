"""
Custom data loader
"""
import glob

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from skimage.transform import rescale, resize, downscale_local_mean

import os
import re
import torch


class PathException(Exception):
    def __init__(self, string):
        super(PathException, self).__init__(string)


class CustomImageDataLoader(Dataset):

    def check_dataset_folder(self):
        if os.path.isdir(self.path_dataset) == False:
            raise PathException(f"The given path is not a valid: {self.path_dataset}")

    def __init__(self, path_dataset: str, transform=None):
        self.path_dataset = path_dataset

        self.check_dataset_folder()

        self.transform = transform

        # get images
        self.image_paths = []
        self.image_paths.extend(glob.glob(os.path.join(path_dataset, '*.png')))
        self.image_paths.extend(glob.glob(os.path.join(path_dataset, '*.jpg')))
        self.image_paths.extend(glob.glob(os.path.join(path_dataset, '*.jpeg')))

        self.size = len(self.image_paths)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        image = io.imread(img_name)
        # print(torch.zeros_like(image))
        image = torch.tensor(image, dtype=torch.float32)

        sample = {'image': image, 'image_path': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SketchImageDataLoader(Dataset):

    def check_dataset_folder(self):
        if os.path.isdir(self.path_dataset) == False:
            raise PathException(f"The given path is not a valid: {self.path_dataset}")
        if os.path.isdir(os.path.join(self.path_dataset, 'images')) == False:
            raise PathException(f"Expected to have the images saved in a folder called 'images'")
        if os.path.isdir(os.path.join(self.path_dataset, 'sketches')) == False:
            raise PathException(f"Expected to have the sketches saved in a folder called 'sketches'")

    def __init__(self, path_dataset: str, transform=None):
        self.path_dataset = path_dataset

        self.check_dataset_folder()

        self.transform = transform

        # get images
        self.image_paths = []
        self.image_paths.extend(glob.glob(os.path.join(path_dataset, 'images', '*.png')))
        self.image_paths.extend(glob.glob(os.path.join(path_dataset, 'images', '*.jpg')))
        self.image_paths.extend(glob.glob(os.path.join(path_dataset, 'images', '*.jpeg')))

        self.sketch_path = self.get_sketch_paths()

        assert len(self.image_paths) == len(self.sketch_path)

        self.size = len(self.image_paths)

    def get_sketch_paths(self):
        """
        Verifies that each image has an associated sketch
        :return:
        """
        sketch_paths = []
        invalid_images = []
        sketch_path = []

        for image_path in self.image_paths:
            # file name

            file_name = re.split(r'(\\)|(/)', image_path)[-1]
            # remove the extension
            file_name = '.'.join((file_name.split('.'))[:-1])

            sketch_path.extend(glob.glob(os.path.join(self.path_dataset, 'sketches', file_name + '.png')))
            sketch_path.extend(glob.glob(os.path.join(self.path_dataset, 'sketches', file_name + '.jpg')))
            sketch_path.extend(glob.glob(os.path.join(self.path_dataset, 'sketches', file_name + '.jpeg')))

            if len(sketch_path) > 0:
                sketch_paths.append(sketch_path[0])
            else:
                invalid_images.append(image_path)

        # self.remove_invalid_images(invalid_images)
        return sketch_path

    def remove_invalid_images(self, invalid_images):
        for path in invalid_images:
            self.image_paths.remove(path)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        image = io.imread(img_name)
        image_resized = resize(image, (224, 224), anti_aliasing=True)
        image = torch.tensor(image)

        sketch_name = self.sketch_path[idx]
        sketch = io.imread(sketch_name)
        sketch = torch.tensor(sketch)

        # print(image_resized.shape)

        sample = {'image': image, 'sketch': sketch, 'image_path': img_name, 'sketch_path': sketch_name,
                  'resized': image_resized}

        if self.transform:
            sample = self.transform(sample)

        return sample
