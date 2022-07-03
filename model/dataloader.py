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


class ImageSketchDataLoader(Dataset):

    def check_dataset_folder(self):
        if not os.path.isdir(self.path_images):
            raise PathException(f"The given path is not a valid: {self.path_images}")
        if not os.path.isdir(self.path_sketches):
            raise PathException(f"The given path is not a valid: {self.path_sketches}")

    def __init__(self, path_images: str, path_sketches: str):
        self.path_images = path_images
        self.path_sketches = path_sketches

        self.check_dataset_folder()

        # get images
        self.image_paths = []
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.png')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpg')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpeg')))

        # get sketches
        self.sketch_paths = []
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.png')))
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.jpg')))
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.jpeg')))

        self.image_paths.sort()
        self.sketch_paths.sort()

        # check whether we find a sketch for each image
        assert len(self.image_paths) == len(self.sketch_paths)

        self.size = len(self.image_paths)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        image = io.imread(img_name, as_gray=False)
        image = torch.tensor(image, dtype=torch.float32)

        sketch_name = self.sketch_paths[idx]
        sketch = io.imread(sketch_name, as_gray=False)
        sketch = torch.tensor(sketch, dtype=torch.float32)

        if len(image.shape) == 2:
            # convert grayscale to rgb
            image = torch.stack([image, image, image], 2)

        if len(sketch.shape) == 2:
            # convert grayscale to rgb
            sketch = torch.stack([sketch, sketch, sketch], 2)

        # if self.transform:
        image, sketch = self.transform(image, sketch)

        sample = {'image': image, 'image_path': img_name, 'sketch': sketch, 'sketch_path': sketch_name}

        return sample

    def transform(self, image: torch.tensor, sketch: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """
        Applies the same transformations to the original images and sketches.
        You will have to overwrite this function for your specific need.
        :param image:
        :param sketch:
        :return: image, sketch
        """

        return image, sketch


class ImageDataLoader(Dataset):

    def check_dataset_folder(self):
        if os.path.isdir(self.path_dataset) == False:
            raise PathException(f"The given path is not a valid: {self.path_dataset}")
        # if os.path.isdir(os.path.join(self.path_dataset, 'images')) == False:
        #     raise PathException(f"Expected to have the images saved in a folder called 'images'")
        # if os.path.isdir(os.path.join(self.path_dataset, 'sketches')) == False:
        #     raise PathException(f"Expected to have the sketches saved in a folder called 'sketches'")

    def __init__(self, path_images: str, transform=None):
        self.path_dataset = path_images

        self.check_dataset_folder()

        self.transform = transform

        # get images
        self.image_paths = []
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.png')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpg')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpeg')))

        # todo: remove, just for testing
        self.image_paths = self.image_paths[:500]  # simulate 500 images

        self.size = len(self.image_paths)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        image = io.imread(img_name)
        image = torch.tensor(image, dtype=torch.float32)  # float32 for full precision

        image = image / 255.0  # convert to [0, 1]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'image_path': img_name}

        return sample
