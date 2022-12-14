
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from tqdm.auto import tqdm
import argparse

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(dataloader):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Dataset Mean and STD',
                    description = 'Compute mean and std of a dataset',
                    epilog = 'Text at the bottom of help')

    parser.add_argument("dataset")
    args = parser.parse_args()
    
    directory = os.path.expanduser(os.path.join("./data", args.dataset))
    if args.dataset == "adaptiope":
        for dataset_name in ["real_life", "product_images"]:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
            dataset = ImageFolder(root=os.path.join(directory, dataset_name), transform=transform)
            print(len(dataset))
            dataloader = DataLoader(dataset=dataset, batch_size=256, num_workers=4)
            mean, std = get_mean_and_std(dataloader)
            print(f"Mean: {mean}, Std: {std}")
    elif args.dataset == "tiny-imagenet-200":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
        dataset = ImageFolder(root=os.path.join(directory, "train"), transform=transform)
        print(len(dataset))
        dataloader = DataLoader(dataset=dataset, num_workers=4, batch_size=256)
        mean, std = get_mean_and_std(dataloader)
        print(f"Mean: {mean}, Std: {std}")

