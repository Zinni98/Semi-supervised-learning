# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]


def get_cifar_os(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    if name == "cifaros100":
        name = "cifar100"
    elif name == "cifaros10":
        name = "cifar10"

    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets


    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, name,
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets


    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets
    test_data = [data for idx, data in enumerate(dset.data) if targets[idx] < num_classes]
    test_targets = [target for target in dset.targets if target < num_classes]
    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset


def split_ssl_data(args, data, targets, num_classes, name,
                   lb_num_labels, ulb_num_labels=None,
                   lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                   lb_index=None, ulb_index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, data, targets, num_classes,
                                                    name, lb_num_labels, ulb_num_labels,
                                                    lb_imbalance_ratio, ulb_imbalance_ratio)
    
    # manually set lb_idx and ulb_idx, do not use except for debug
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    
    return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]



def sample_labeled_unlabeled_data(args, data, target, num_classes, name,
                                  lb_num_labels, ulb_num_labels=None,
                                  lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                                  load_exist=False):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    dump_dir = os.path.join(base_dir, 'data', args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)
    lb_dump_path = os.path.join(dump_dir, f'lb_labels{args.num_labels}_seed{args.seed}_idx.npy')
    ulb_dump_path = os.path.join(dump_dir, f'ulb_labels{args.num_labels}_seed{args.seed}_idx.npy')

    if os.path.exists(lb_dump_path) and os.path.exists(ulb_dump_path) and load_exist:
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)
        return lb_idx, ulb_idx 

    
    # get samples per class
    if lb_imbalance_ratio == 1.0:
        # balanced setting, lb_num_labels is total number of labels for labeled data
        assert lb_num_labels % num_classes == 0, "lb_num_labels must be divideable by num_classes in balanced setting"
        lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes


    if ulb_imbalance_ratio == 1.0:
        # balanced setting
        if ulb_num_labels is not None and ulb_num_labels != 'None':
            assert ulb_num_labels % num_classes == 0, "ulb_num_labels must be divideable by num_classes in balanced setting"
            ulb_samples_per_class = [int(ulb_num_labels / num_classes)] * num_classes
        else:
            ulb_samples_per_class = None

    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        if ulb_samples_per_class is None:
            ulb_idx.extend(idx[lb_samples_per_class[c]:])
        else:
            ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c]+ulb_samples_per_class[c]])
    
    # Set the remaining classes as 
    if name == "cifar100":
        for c in range(num_classes, 100):
            idx = np.where(target == c)[0]
            ulb_idx.extend(idx)
    else:
        for c in range(num_classes, 10):
            idx = np.where(target == c)[0]
            ulb_idx.extend(idx)

    
    if isinstance(lb_idx, list):
        lb_idx = np.asarray(lb_idx)
    if isinstance(ulb_idx, list):
        ulb_idx = np.asarray(ulb_idx)

    np.save(lb_dump_path, lb_idx)
    np.save(ulb_dump_path, ulb_idx)
    
    return lb_idx, ulb_idx