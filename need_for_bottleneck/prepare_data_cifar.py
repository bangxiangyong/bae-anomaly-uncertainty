import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from baetorch.baetorch.evaluation import calc_auroc, calc_avgprc
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.invert import Invert
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood_v2.util.get_predictions import flatten_nll
import matplotlib.pyplot as plt
import numpy as np
import os

train_batch_size = 100
test_samples = 100


def get_ood_set(
    ood_dataset="SVHN",
    n_channels=-1,
    test_samples=100,
    shuffle=True,
    resize=None,
    base_folder="dataset",
):
    # check path dataset
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    # prepare data transformation
    if (
        n_channels == 3 and (ood_dataset == "FashionMNIST" or ood_dataset == "MNIST")
    ) or (n_channels == 1 and (ood_dataset == "CIFAR" or ood_dataset == "SVHN")):
        if resize is not None:
            data_trans_ = [
                transforms.Grayscale(num_output_channels=n_channels),
                transforms.Resize(resize),
                transforms.ToTensor(),
            ]
        else:
            data_trans_ = [
                transforms.Grayscale(num_output_channels=n_channels),
                transforms.ToTensor(),
            ]
    else:
        # Data transformation
        data_trans_ = [transforms.ToTensor()]
    data_transform = transforms.Compose(data_trans_)

    # Load ood set
    if ood_dataset == "SVHN":
        ood_loader_ = torch.utils.data.DataLoader(
            datasets.SVHN(
                os.path.join(base_folder, "data-svhn"),
                split="test",
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    elif ood_dataset == "FashionMNIST":
        ood_loader_ = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                os.path.join(base_folder, "data-fashion-mnist"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    elif ood_dataset == "MNIST":
        ood_loader_ = torch.utils.data.DataLoader(
            datasets.MNIST(
                os.path.join(base_folder, "data-mnist"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    elif ood_dataset == "CIFAR":
        ood_loader_ = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                os.path.join(base_folder, "data-cifar"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    else:
        raise NotImplemented("OOD set can be CIFAR, FashionMNIST, MNIST or SVHN only.")
    return ood_loader_


def get_id_set(
    id_dataset="CIFAR",
    n_channels=-1,
    shuffle=True,
    train_batch_size=100,
    test_samples=100,
    resize=None,
    base_folder="dataset",
):
    # check path dataset
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    # prepare data transformation
    if (
        n_channels == 3 and (id_dataset == "FashionMNIST" or id_dataset == "MNIST")
    ) or (n_channels == 1 and (id_dataset == "CIFAR" or id_dataset == "SVHN")):
        if resize is not None:
            data_trans_ = [
                transforms.Grayscale(num_output_channels=n_channels),
                transforms.Resize(resize),
                transforms.ToTensor(),
            ]
        else:
            data_trans_ = [
                transforms.Grayscale(num_output_channels=n_channels),
                transforms.ToTensor(),
            ]
    else:
        # Data transformation
        data_trans_ = [transforms.ToTensor()]
    data_transform = transforms.Compose(data_trans_)

    # Load OOD set
    if id_dataset == "SVHN":
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                os.path.join(base_folder, "data-svhn"),
                split="train",
                download=True,
                transform=data_transform,
            ),
            batch_size=train_batch_size,
            shuffle=shuffle,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                os.path.join(base_folder, "data-svhn"),
                split="test",
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    elif id_dataset == "FashionMNIST":
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                os.path.join(base_folder, "data-fashion-mnist"),
                train=True,
                download=True,
                transform=data_transform,
            ),
            batch_size=train_batch_size,
            shuffle=shuffle,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                os.path.join(base_folder, "data-fashion-mnist"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    elif id_dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                os.path.join(base_folder, "data-mnist"),
                train=True,
                download=True,
                transform=data_transform,
            ),
            batch_size=train_batch_size,
            shuffle=shuffle,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                os.path.join(base_folder, "data-mnist"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=False,
        )
    elif id_dataset == "CIFAR":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                os.path.join(base_folder, "data-cifar"),
                train=True,
                download=True,
                transform=data_transform,
            ),
            batch_size=train_batch_size,
            shuffle=shuffle,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                os.path.join(base_folder, "data-cifar"),
                train=False,
                download=True,
                transform=data_transform,
            ),
            batch_size=test_samples,
            shuffle=shuffle,
        )
    else:
        raise NotImplemented("ID set can be CIFAR, FashionMNIST, MNIST or SVHN only.")
    return train_loader, test_loader
