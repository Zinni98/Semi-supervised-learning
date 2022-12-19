"""
Author: Thomas De Min
GitHub: tdemin16
"""

import argparse
import matplotlib as mpl
import numpy as np
import os
import pickle
import seaborn as sb
import torch
from matplotlib import pyplot as plt
from semilearn.core.utils import get_net_builder, get_dataset
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from collections import defaultdict, Counter

#! Insert here new datasets, follow the format of cifar100
DATASET_IMGS = {
    "cifar100": {
        "num_imgs": 60000,
        "img_size": 32,
        "num_classes": 100,
        "mean": [x / 255 for x in [129.3, 124.1, 112.4]],
        "std": [x / 255 for x in [68.2, 65.4, 70.4]]
    },
    "cifaros100":{
        "num_imgs": 10000,
        "img_size": 32,
        "num_classes": 80,
        "mean": [x / 255 for x in [129.3, 124.1, 112.4]],
        "std": [x / 255 for x in [68.2, 65.4, 70.4]]
    }
}

#! Default values. Change them to avoid using args,
#! otherwise keep them as they are
METHOD = "fixmatch"
DATASET = "cifaros100"
DATASET_DIR = "data"
NUM_LABELS = 400
SEED = 0
NET = "vit_small_patch2_32"
CROP_RATIO = 1
LB_IMB_RATIO = 1
ULB_IMB_RATIO = 1
BATCH_SIZE = 64
NUM_WORKERS = 3
SAVE_PATH = "assets"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_if_required(path):
    """
    Create :path: if do not exists.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def fix_weights(state_dict):
    """
    Copied from usb. It removes "module" in front of some weights
    in :state_dict:.

    Returns the sanitized :state_dict:
    """
    new_state_dict = {}
    for key, item in state_dict.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = item
        else:
            new_state_dict[key] = item
    
    return new_state_dict


def inverse_norm(args):
    """
    Compose the inverse normalize from mean and std in args.
    """
    return transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / x for x in args.std]),
        transforms.Normalize(mean=[-x for x in args.mean],
                             std=[1., 1., 1.])
    ])


def save(args, imgs):
    """
    Save images in path: 
        args.save_path/args.dataset/args.save_wrongs[0]_args.save_wrongs[1]/###.png
    
    :imgs: tensor of shape [*, 3, img_size, img_size]
    """
    name = args.save_wrongs[0] + '_' + args.save_wrongs[1]
    path = os.path.join(args.save_path, args.dataset, name)
    make_if_required(path)

    transform = inverse_norm(args)
    imgs = transform(imgs)

    for i, img in enumerate(imgs):
        img_path = os.path.join(path, str(i).zfill(3) + '.png')
        save_image(img.squeeze(0), img_path)


@torch.no_grad()
def eval(args, model, dl, labels):
    """
    Evaluate :model: on dataloader :dl:. 
    :labels: list of labels ordered according to integer annotations.

    If args.save_wrongs is defined, save images according to :save: docstring.
    """
    acc = 0
    count = 0
    estimates = []
    ground_truths = []

    imgs_to_return = None

    # indexes to track
    # pred_indx = labels.index(args.save_wrongs[1])
    # gt_index = labels.index(args.save_wrongs[0])
    correct_y = []
    for data in tqdm(dl):
        X = data['x_ulb_w']
        y = data['y_ulb']

        X = X.type(torch.FloatTensor).cuda()
        # y = y.cuda()
        logits = model(X)['logits']
        y_hat, predictions_cls = torch.softmax(logits, dim=1).max(1)
        # acc += y_hat.cpu().eq(y.cpu()).numpy().sum()
        # count += y_hat.cpu().size(0)

        # estimates.extend(y_hat.tolist())#.cpu())
        # ground_truths.extend(y.cpu())
        estimates.extend(predictions_cls.tolist())
        correct_y.extend(y.tolist())
        if args.save_wrongs is not None:
            # get all predictions and labels that match indexes to track
            pred_match = (y_hat == pred_indx)
            gt_match = (y == gt_index)
            pred_gt_match = torch.logical_and(pred_match, gt_match)
            
            if imgs_to_return is None:
                imgs_to_return = X[pred_gt_match]
            else:
                imgs_to_return = torch.cat((imgs_to_return, X[pred_gt_match]), dim=0)
        
        if args.dev:
            break

    estimates = torch.tensor(estimates)
    
    # keep only high confidence results
    mask = (estimates >= 0.95) # .type(torch.float).mean()
    estimates = torch.masked_select(estimates, mask).tolist()
    
    res = defaultdict(list)
    for idx, el in enumerate(estimates):
        res[labels[correct_y[idx]]].append(el)

    for key in res.keys():
        c = Counter(res[key])
        res[key] = [(labels[t[0]], t[1]) for t in c.most_common(3)]# labels[c.most_common(3)[0]]
    
    print(res)
    
    if args.save_wrongs is not None:
        save(args, imgs_to_return)

    return acc / count, estimates, ground_truths


def get_labels(dataset):
    # open meta file and get cifar lables
    if dataset == "cifar100" or dataset == "cifaros100":
        with open(os.path.join(DATASET_DIR, "cifar100/cifar-100-python/meta"), 'rb') as fp:
            ds = pickle.load(fp)
        return ds['fine_label_names']
    else:
        raise NotImplementedError


def confusionmatrix(args, estimates, ground_truths, labels):
    """
    Compute confusion matrix from :estimates: and :ground_truths:
    using :labels: as ticks.
    """
    cm = MulticlassConfusionMatrix(num_classes=args.num_classes)
    conf_matrix = cm(estimates, ground_truths).numpy()

    path = os.path.join(
        args.save_path,
        args.dataset
    )
    make_if_required(path)

    plt.figure(figsize=(20, 20))
    heatmap = sb.heatmap(conf_matrix, annot=True, annot_kws={"fontsize": 5}, square=True,
                         xticklabels=labels, yticklabels=labels, vmin=0, vmax=100, fmt='g')
    plt.xlabel("Predictions", fontsize=20)
    plt.ylabel("Ground Truths", fontsize=20)
    figure = heatmap.get_figure()
    figure.savefig(os.path.join(path, f"{args.method}.png"), dpi=400)


def main(args):
    print(args)

    model = get_net_builder(args.net, from_name=False)(num_classes=args.num_classes).cuda()
    weights = torch.load(args.weights, map_location="cuda:0")["model"]
    weights = fix_weights(weights)
    model.load_state_dict(weights)
    model.eval()

    ds = get_dataset(args, args.method, args.dataset, args.num_labels, args.num_classes)
    val_ds = ds['eval']
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, 
                         drop_last=False, shuffle=False, num_workers=args.num_workers)

    try:
        labels = get_labels(args.dataset)
    except NotImplementedError:
        labels = range(0, args.num_classes)

    acc, estimates, ground_truths = eval(args, model, val_dl, labels)
    print(f"\nAccuracy: {acc*100}%")

    confusionmatrix(args, estimates, ground_truths, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=METHOD,
                        help="fixmatch, remixmatch...")
    parser.add_argument("--dataset", type=str, default=DATASET, choices=list(DATASET_IMGS.keys()),
                        help="cifar100, svhn...")
    parser.add_argument("--num_labels", type=int, default=NUM_LABELS,
                        help="Number of labels of the model")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Seed, use the same of training.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to weights.")
    parser.add_argument("--net", type=str, default=NET,
                        help="Architecture used, use the same as training.")
    parser.add_argument("--crop_ratio", type=int, default=CROP_RATIO)
    parser.add_argument("--lb_imb_ratio", type=int, default=LB_IMB_RATIO)
    parser.add_argument("--ulb_imb_ratio", type=int, default=ULB_IMB_RATIO)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--save_path", type=str, default=SAVE_PATH,
                        help="Path in where to save Confusion Matrix and other assets.")
    parser.add_argument("--dev", action="store_true",
                        help="Exec only one batch to spare computation during development.")
    parser.add_argument("--save_wrongs", nargs=2,
                        help="Save wrong predictions in the form of ground_truth prediction (only one pair).")

    # workaround to add stuff to args
    tmp_args, _ = parser.parse_known_args()
    dataset = DATASET_IMGS[tmp_args.dataset]
    parser.add_argument("--img_size", type=int, default=dataset["img_size"])
    parser.add_argument("--ulb_num_labels", type=int, default=dataset["num_imgs"] - tmp_args.num_labels)
    parser.add_argument("--num_classes", type=int, default=dataset["num_classes"])
    parser.add_argument("--mean", nargs=3, default=dataset["mean"])
    parser.add_argument("--std", nargs=3, default=dataset["std"])

    args = parser.parse_args()

    main(args)