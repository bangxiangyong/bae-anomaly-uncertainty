from need_for_bottleneck.prepare_data_cifar import get_ood_set
import matplotlib.pyplot as plt
import numpy as np

# fmt: off
id_n_channels = {"CIFAR": 3, "SVHN": 3, "FashionMNIST": 1, "MNIST": 1}
flattened_dims = {"CIFAR": 3*32*32, "SVHN": 3*32*32, "FashionMNIST": 28*28, "MNIST": 28*28}
input_dims = {"CIFAR": 32, "SVHN": 32, "FashionMNIST": 28, "MNIST": 28}
# fmt: on

id_dataset = "CIFAR"
ood_datasets = [
    dt for dt in ["SVHN", "CIFAR", "FashionMNIST", "MNIST"] if dt != id_dataset
]
ood_datasets = ["SVHN", "CIFAR", "FashionMNIST", "MNIST"]
# ood_datasets = [dt for dt in ["MNIST"] if dt != id_dataset]
fig, axes = plt.subplots(len(ood_datasets),10)
for row_i,ood_dataset in enumerate(ood_datasets):

    ood_loader = get_ood_set(
        ood_dataset=ood_dataset,
        n_channels=id_n_channels[id_dataset],
        resize=[input_dims[id_dataset]] * 2,
    )

    for col_i in range(10):
        axes[row_i][col_i].imshow(np.moveaxis(next(iter(ood_loader))[0][0].detach().numpy(),0,2))