#!/usr/bin/env python
"""
Custom data loader
"""
import glob
import os
import random

import PIL
import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL.Image import Image
from skimage import io
from torch.utils.data import Dataset
import torchvision.transforms as T


class PathException(Exception):
    def __init__(self, string):
        super(PathException, self).__init__(string)


class ImageSketchDataLoader(Dataset):

    def check_dataset_folder(self):
        if not os.path.isdir(self.path_images):
            raise PathException(f"The given path is not a valid: {self.path_images}")
        if not os.path.isdir(self.path_sketches):
            raise PathException(f"The given path is not a valid: {self.path_sketches}")

    def __init__(self, path_images: str, path_sketches: str, size=256, max_num_images = -1):
        print('init custom image-sketch dataset')
        self.path_images = path_images
        self.path_sketches = path_sketches
        self.resize_size = size

        self.check_dataset_folder()

        # get images
        print('scanning the images')
        self.image_paths = []
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.png')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpg')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpeg')))

        # get sketches
        print('scanning the sketches')
        self.sketch_paths = []
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.png')))
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.jpg')))
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.jpeg')))

        print('sorting the images and sketches')
        self.image_paths.sort()
        self.sketch_paths.sort()

        self.apply_transform = False

        print('limiting number of images')
        if max_num_images > 0:
            self.image_paths = self.image_paths[:max_num_images]
            self.sketch_paths = self.sketch_paths[:max_num_images]
        print('done')

        # check whether we find a sketch for each image
        assert len(self.image_paths) == len(self.sketch_paths)

        self.size = len(self.image_paths)

    def __len__(self):
        return self.size

    def preprocess(self, img):
        s = min(img.size)

        if s < self.resize_size:
            raise ValueError(f'min dim for image {s} < {self.resize_size}')

        r = self.resize_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [self.resize_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        image = PIL.Image.open(img_name).convert('RGB')
        image = self.preprocess(image)
        # image = io.imread(img_name, as_gray=False)
        # image = torch.tensor(image, dtype=torch.float32)
        # image = image / 255.0

        sketch_name = self.sketch_paths[idx]
        sketch = PIL.Image.open(sketch_name).convert('RGB')
        sketch = self.preprocess(sketch)
        # sketch = Image.open(sketch_name).convert('RGB')
        # sketch = io.imread(sketch_name, as_gray=False)
        # sketch = torch.tensor(sketch, dtype=torch.float32)
        # sketch = sketch / 255.0

        # print('shape image:', image.shape)
        # print('shape sketch:', sketch.shape)

        sample = {'image': image.squeeze(), 'image_path': img_name, 'sketch': sketch.squeeze(), 'sketch_path': sketch_name}

        return sample

    def transform(self, image: torch.tensor, sketch: torch.tensor):
        """
        Applies the same transformations to the original images and sketches.
        You will have to overwrite this function for your specific need.
        :param image:
        :param sketch:
        :return: image, sketch
        """

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            sketch = TF.hflip(sketch)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            sketch = TF.vflip(sketch)

        image = np.array(image)
        sketch = np.array(sketch)

        train_transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            ]
        )

        return train_transform(image=image, sketch=sketch)


class ImageSketchDataLoaderTrain(ImageSketchDataLoader):
    def __init__(self, path_images: str, path_sketches: str, size=256, max_num_images = -1, split=1):
        super(ImageSketchDataLoaderTrain, self).__init__(path_images, path_sketches, size, max_num_images)
        print('init custom image-sketch dataset')
        self.path_images = path_images
        self.path_sketches = path_sketches
        self.resize_size = size

        self.check_dataset_folder()

        # get images
        print('scanning the images')
        self.image_paths = []
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.png')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpg')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpeg')))

        # get sketches
        print('scanning the sketches')
        self.sketch_paths = []
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.png')))
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.jpg')))
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.jpeg')))

        print('sorting the images and sketches')
        self.image_paths.sort()
        self.sketch_paths.sort()

        self.apply_transform = False

        print('limiting number of images')
        whole_size = len(self.image_paths)

        train_size = int(whole_size * split)
        test_size = whole_size - train_size

        if max_num_images > 0:
            self.image_paths = self.image_paths[:min(max_num_images, train_size)]
            self.sketch_paths = self.sketch_paths[:min(max_num_images, train_size)]
        print('done')

        # check whether we find a sketch for each image
        assert len(self.image_paths) == len(self.sketch_paths)

        self.size = len(self.image_paths)

class ImageSketchDataLoaderTest(ImageSketchDataLoader):
    def __init__(self, path_images: str, path_sketches: str, size=256, max_num_images = -1, split=1):
        super(ImageSketchDataLoaderTrain, self).__init__(path_images, path_sketches, size, max_num_images)
        print('init custom image-sketch dataset')
        self.path_images = path_images
        self.path_sketches = path_sketches
        self.resize_size = size

        self.check_dataset_folder()

        # get images
        print('scanning the images')
        self.image_paths = []
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.png')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpg')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpeg')))

        # get sketches
        print('scanning the sketches')
        self.sketch_paths = []
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.png')))
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.jpg')))
        self.sketch_paths.extend(glob.glob(os.path.join(path_sketches, '*.jpeg')))

        print('sorting the images and sketches')
        self.image_paths.sort()
        self.sketch_paths.sort()

        self.apply_transform = False

        print('limiting number of images')
        whole_size = len(self.image_paths)

        train_size = int(whole_size * split)
        test_size = whole_size - train_size

        self.image_paths = self.image_paths[train_size:]
        self.sketch_paths = self.sketch_paths[train_size:]

        if max_num_images > 0:
            self.image_paths = self.image_paths[:max_num_images]
            self.sketch_paths = self.sketch_paths[:max_num_images]
        print('done')

        # check whether we find a sketch for each image
        assert len(self.image_paths) == len(self.sketch_paths)

        self.size = len(self.image_paths)

class RandomChoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = random.choice(self.transforms)

    def __call__(self, img):
        return self.t(img)


class RandomChoiceBatch(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        return [t(img) for img in imgs]


class ImageDataLoader(Dataset):

    def check_dataset_folder(self):
        if os.path.isdir(self.path_dataset) == False:
            raise PathException(f"The given path is not a valid: {self.path_dataset}")
        # if os.path.isdir(os.path.join(self.path_dataset, 'images')) == False:
        #     raise PathException(f"Expected to have the images saved in a folder called 'images'")
        # if os.path.isdir(os.path.join(self.path_dataset, 'sketches')) == False:
        #     raise PathException(f"Expected to have the sketches saved in a folder called 'sketches'")

    def __init__(self, path_images: str, transform=None, size=None):
        self.path_dataset = path_images

        self.check_dataset_folder()

        self.transform = transform

        # get images
        self.image_paths = []
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.png')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpg')))
        self.image_paths.extend(glob.glob(os.path.join(path_images, '*.jpeg')))

        self.image_paths = self.image_paths  # [:500]  # simulate 500 images

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
