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
from tqdm import tqdm

# n_images, size, classes
DATASET_IMGS = {
    "cifar100": (60000, 32, 100),
}

METHOD = "fixmatch"
DATASET = "cifar100"
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
    if not os.path.isdir(path):
        os.makedirs(path)


def fix_weights(state_dict):
    """
    Copied from usb
    """
    new_state_dict = {}
    for key, item in state_dict.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = item
        else:
            new_state_dict[key] = item
    
    return new_state_dict


@torch.no_grad()
def eval(args, model, dl, ds_len):
    acc = 0
    estimates = []
    ground_truths = []

    for data in tqdm(dl):
        X = data['x_lb']
        y = data['y_lb']

        X = X.type(torch.FloatTensor).cuda()
        logits = model(X)['logits']
        y_hat = logits.argmax(1)

        acc += y_hat.cpu().eq(y).numpy().sum()

        estimates.extend(y_hat.cpu())
        ground_truths.extend(y.cpu())

    estimates = torch.tensor(estimates)
    ground_truths = torch.tensor(ground_truths)

    return acc / ds_len, estimates, ground_truths


def get_labels(dataset):
    if dataset == "cifar100":
        with open(os.path.join(DATASET_DIR, "cifar100/cifar-100-python/meta"), 'rb') as fp:
            ds = pickle.load(fp)
        return ds['fine_label_names']
    else:
        raise NotImplementedError


def confusionmatrix(args, estimates, ground_truths):
    cm = MulticlassConfusionMatrix(num_classes=args.num_classes)
    conf_matrix = cm(estimates, ground_truths).numpy()

    path = os.path.join(
        args.save_path,
        args.dataset
    )
    make_if_required(path)
    
    fig, ax = plt.subplots(figsize=(45, 45))
    ax.matshow(conf_matrix, cmap=mpl.colormaps['magma'], alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=25)
    plt.ylabel('Actuals', fontsize=25)
    plt.xticks(np.arange(0, args.num_classes-1))
    plt.yticks(np.arange(0, args.num_classes-1))
    plt.title('Confusion Matrix', fontsize=25)
    plt.savefig(os.path.join(path, f"{args.method}.png"))


def confusionmatrix_seaborn(args, estimates, ground_truths, labels):
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

    model = get_net_builder(args.net, from_name=False)(num_classes=dataset[2]).cuda()
    weights = torch.load(args.weights, map_location="cuda:0")["model"]
    weights = fix_weights(weights)
    model.load_state_dict(weights)
    model.eval()

    ds = get_dataset(args, args.method, args.dataset, args.num_labels, dataset[2])
    val_ds = ds['eval']
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, 
                         drop_last=False, shuffle=False, num_workers=args.num_workers)

    acc, estimates, ground_truths = eval(args, model, val_dl, len(val_ds))
    print(f"\nAccuracy: {acc*100}%")

    try:
        labels = get_labels(args.dataset)
    except NotImplementedError:
        labels = range(0, args.num_classes)

    confusionmatrix_seaborn(args, estimates, ground_truths, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=METHOD)
    parser.add_argument("--dataset", type=str, default=DATASET, choices=list(DATASET_IMGS.keys()))
    parser.add_argument("--num_labels", type=int, default=NUM_LABELS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--net", type=str, default=NET)
    parser.add_argument("--crop_ratio", type=int, default=CROP_RATIO)
    parser.add_argument("--lb_imb_ratio", type=int, default=LB_IMB_RATIO)
    parser.add_argument("--ulb_imb_ratio", type=int, default=ULB_IMB_RATIO)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--save_path", type=str, default=SAVE_PATH)

    # workaround to add stuff to args
    tmp_args, _ = parser.parse_known_args()
    dataset = DATASET_IMGS[tmp_args.dataset]
    parser.add_argument("--img_size", type=int, default=dataset[1])
    parser.add_argument("--ulb_num_labels", type=int, default=dataset[0] - tmp_args.num_labels)
    parser.add_argument("--num_classes", type=int, default=dataset[2])

    args = parser.parse_args()

    main(args)