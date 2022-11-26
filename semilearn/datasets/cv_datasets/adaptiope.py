# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math

from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from .datasetbase import BasicDataset


mean, std = {}, {}
mean['adaptiope'] = [0.485, 0.456, 0.406]
std['adaptiope'] = [0.229, 0.224, 0.225]
img_size = 224


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

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['adaptiope'], std['adaptiope'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['adaptiope'], std['adaptiope'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['adaptiope'], std['adaptiope'])
    ])

    data_dir = os.path.join(data_dir, name.lower())

    print("SUGO")

    # Split filenames in train/val balanced
    product_root = os.path.join(data_dir, "product")
    classes, class_to_idx = ImageFolder.find_classes(product_root)
    directory = os.path.expanduser(product_root)

    train_fnames = []
    train_classes = []
    val_fnames = []
    val_classes = []
    TRAIN_PERCENTAGE = 0.8

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue 
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            random.shuffle(fnames)
            train_fnames.extend(fnames[:int(len(fnames)*TRAIN_PERCENTAGE)])
            train_classes.extend([class_index]*int(len(fnames)*TRAIN_PERCENTAGE))
            val_fnames.extend(fnames[int(len(fnames)*TRAIN_PERCENTAGE):])
            val_classes.extend([class_index]*(len(fnames)-int(len(fnames)*TRAIN_PERCENTAGE)))

            assert len(train_classes) == len(train_fnames)
            assert len(val_classes) == len(val_fnames)


    lb_dset = AdaptiopeDataset(root=os.path.join(data_dir, "product"), samples=(train_fnames, train_classes), transform=transform_weak, ulb=False, alg=alg, num_labels=num_labels)

    ulb_dset = AdaptiopeDataset(root=os.path.join(data_dir, "real"), transform=transform_weak, alg=alg, ulb=True, strong_transform=transform_strong)

    eval_dset = AdaptiopeDataset(root=os.path.join(data_dir, "product"), samples=(val_fnames, val_classes), transform=transform_val, alg=alg, ulb=False)

    return lb_dset, ulb_dset, eval_dset
    


class AdaptiopeDataset(BasicDataset, ImageFolder):
    def __init__(self, root, transform, ulb, alg, samples=None, strong_transform=None, num_labels=-1):
        self.alg = alg
        self.is_ulb = ulb
        self.num_labels = num_labels
        self.transform = transform
        self.root = root

        assert samples is None or ulb is False, "samples should be None for unlabeled data"

        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file, samples)
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
            for fname, target in zip(samples):
                path = os.path.join(root, target, fname)
                if is_valid_file(path):
                    item = (path, target)
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
