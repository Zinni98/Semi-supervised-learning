# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import gc
import numpy as np
import copy
import json
import random
from torchvision.datasets import ImageFolder, folder
from PIL import Image
from torchvision import transforms
import math

from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from .datasetbase import BasicDataset


mean, std = {}, {}
mean['adaptiope-product'] = [0.8082, 0.7995, 0.7967]
std['adaptiope-product'] = [0.3176, 0.3218, 0.3244]
mean['adaptiope-real'] = [0.4948, 0.4466, 0.4028]
std['adaptiope-real'] = [0.2760, 0.2638, 0.2616]
img_size = 224

PRODUCT = "product_images"
REAL = "real_life"


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_adaptiope(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    num_labels = num_labels // num_classes

    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak_product = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['adaptiope-product'], std['adaptiope-product'])
    ])
    transform_weak_real = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['adaptiope-real'], std['adaptiope-real'])
    ])

    transform_strong_real = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['adaptiope-real'], std['adaptiope-real'])
    ])

    transform_val_product = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['adaptiope-product'], std['adaptiope-product'])
    ])

    data_dir = os.path.join(data_dir, name.lower())

    # Split filenames in train/val balanced
    product_root = os.path.join(data_dir, PRODUCT)
    classes, class_to_idx = folder.find_classes(product_root)
    directory = os.path.expanduser(product_root)

    train_samples = []
    val_samples = []
    TRAIN_PERCENTAGE = 0.8

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            random.shuffle(fnames)

            train_samples.extend((fname, target_class) for fname in fnames[:int(len(fnames)*TRAIN_PERCENTAGE)])
            val_samples.extend((fname, target_class) for fname in fnames[int(len(fnames)*TRAIN_PERCENTAGE):])

    lb_dset = AdaptiopeDataset(root=os.path.join(data_dir, PRODUCT), samples=train_samples, transform=transform_weak_product, ulb=False, alg=alg, num_labels=num_labels)

    ulb_dset = AdaptiopeDataset(root=os.path.join(data_dir, REAL), transform=transform_weak_real, alg=alg, ulb=True, strong_transform=transform_strong_real)

    eval_dset = AdaptiopeDataset(root=os.path.join(data_dir, PRODUCT), samples=val_samples, split="eval", transform=transform_val_product, alg=alg, ulb=False)
    return lb_dset, ulb_dset, eval_dset


class AdaptiopeDataset(BasicDataset, ImageFolder):
    def __init__(self, root, transform, ulb, alg, samples=None, strong_transform=None, num_labels=-1, split=None):
        self.alg = alg
        self.is_ulb = ulb
        self.num_labels = num_labels
        self.transform = transform
        self.root = root

        assert samples is None or ulb is False, "samples should be None for unlabeled data"

        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file, samples, split)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = samples
        self.targets = [s[1] for s in samples]

        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"

    def __sample__(self, index):
        path, target = self.data[index]
        sample = self.loader(path)
        return sample, target

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
            samples=None,
            split=None
    ):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)

        lb_idx = {}

        if samples is not None:
            if split == "eval":
                for fname, target in samples:
                    path = os.path.join(directory, target, fname)
                    if is_valid_file(path):
                        item = (path, class_to_idx[target])
                        instances.append(item)
            else:
                chunks = np.array_split(samples, len(class_to_idx.keys()))
                for ch_samples in chunks:
                    random.shuffle(ch_samples)
                    for fname, target in ch_samples[:self.num_labels]:
                        path = os.path.join(directory, target, fname)
                        if is_valid_file(path):
                            item = (path, class_to_idx[target])
                            instances.append(item)
        else:
            for target_class in sorted(class_to_idx.keys()):
                class_index = class_to_idx[target_class]
                target_dir = os.path.join(directory, target_class)
                if not os.path.isdir(target_dir):
                    continue
                for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                    random.shuffle(fnames)
                    if self.num_labels != -1:
                        fnames = fnames[:self.num_labels]
                    if self.num_labels != -1:
                        lb_idx[target_class] = fnames
                    for fname in fnames:
                        path = os.path.join(root, fname)
                        if is_valid_file(path):
                            item = path, class_index
                            instances.append(item)
            if self.num_labels != -1:
                with open('./sampled_label_idx.json', 'w') as f:
                    json.dump(lb_idx, f)
            del lb_idx
            gc.collect()
        return instances
