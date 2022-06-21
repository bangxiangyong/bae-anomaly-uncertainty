import collections

from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
import os
import numpy as np
from scipy.special import erf
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from statsmodels.distributions import ECDF
from torchvision import datasets, transforms
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2, run_auto_lr_range_v3
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import (
    Encoder,
    infer_decoder,
    Autoencoder,
)
from baetorch.baetorch.models.base_layer import DenseLayers, ConvLayers, flatten_np
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.data_model_manager import DataModelManager
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood.calc_uncertainty_ood import (
    calc_ood_threshold,
    convert_hard_predictions,
)
from uncertainty_ood.load_benchmark import load_benchmark_dt
from util.convergence import (
    bae_fit_convergence,
    plot_convergence,
    bae_semi_fit_convergence,
    bae_fit_convergence_v2,
    bae_norm_fit_convergence,
)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.metrics import matthews_corrcoef
from util.evaluate_ood import plot_histogram_ood, plot_kde_ood
from statsmodels.stats.stattools import medcouple
from sklearn.metrics import confusion_matrix
from scipy.stats import beta, gamma, lognorm, norm, uniform, expon

from util.exp_manager import ExperimentManager
from util.helper import concat_params_res
from util.uncertainty import (
    convert_prob,
    convert_cdf,
    get_pred_unc,
    get_y_results,
    evaluate_mcc_f1_unc,
    evaluate_unc_perf,
    plot_unc_tptnfpfn,
    plot_kde_auroc,
    evaluate_error_unc,
    calc_performance_unc,
    get_optimal_threshold,
    convert_hard_pred,
    get_pred_optimal,
    calc_hard_metrics,
    calc_metrics2,
    calc_spman_metrics,
    eval_retained_unc,
    eval_auroc_ood,
    rename_col_res,
    get_y_true,
)
import torch


def calc_auroc(nll_inliers_train_mean, nll_inliers_valid_mean):
    y_true = np.concatenate(
        (
            np.zeros(nll_inliers_train_mean.shape[0]),
            np.ones(nll_inliers_valid_mean.shape[0]),
        )
    )
    y_scores = np.concatenate((nll_inliers_train_mean, nll_inliers_valid_mean))

    auroc = roc_auc_score(y_true, y_scores)
    return auroc


def calc_avgprc(nll_inliers_train_mean, nll_inliers_valid_mean):
    y_true = np.concatenate(
        (
            np.zeros(nll_inliers_train_mean.shape[0]),
            np.ones(nll_inliers_valid_mean.shape[0]),
        )
    )
    y_scores = np.concatenate((nll_inliers_train_mean, nll_inliers_valid_mean))

    auroc = average_precision_score(y_true, y_scores)
    return auroc


def calc_auroc_valid(bae_ensemble, x_inliers_train, x_inliers_valid):

    # nll_inliers_train = bae_ensemble.predict_samples(
    #     x_inliers_train, select_keys=["se"]
    # )
    # nll_inliers_valid = bae_ensemble.predict_samples(
    #     x_inliers_valid, select_keys=["se"]
    # )
    #
    nll_inliers_train = bae_ensemble.predict_samples(
        x_inliers_train, select_keys=["nll_homo"]
    )
    nll_inliers_valid = bae_ensemble.predict_samples(
        x_inliers_valid, select_keys=["nll_homo"]
    )

    nll_inliers_train_mean = nll_inliers_train.mean(0)[0].mean(-1)
    nll_inliers_valid_mean = nll_inliers_valid.mean(0)[0].mean(-1)

    y_true = np.concatenate(
        (
            np.zeros(nll_inliers_train_mean.shape[0]),
            np.ones(nll_inliers_valid_mean.shape[0]),
        )
    )
    y_scores = np.concatenate((nll_inliers_train_mean, nll_inliers_valid_mean))

    auroc = roc_auc_score(y_true, y_scores)

    return auroc


# random_seed = 987
# random_seed = 33
# random_seed = 1233
# random_seed = 22
random_seed = 10

bae_set_seed(random_seed)

train_set_name = "CIFAR"
use_cuda = True
clip_data_01 = True
activation = "silu"
last_activation = "sigmoid"  # tanh
likelihood = "gaussian"  # gaussian
homo_mode = "every"
decoder_sig_enabled = True
train_size = 0.75
num_samples = 1
# weight_decay = 0.0001
weight_decay = 0.0001
lr = 0.005
# lr = 0.0085
anchored = False
# sparse = False
sparse = True
skip = False
add_se = False
norm = True

# conv_filters = [32, 64, 128]
# conv_kernel = [4, 4, 4]
# conv_stride = [2, 1, 2]

conv_filters = [32, 128]
conv_kernel = [4, 8]
conv_stride = [2, 1]

# conv_filters = [32, 64]
# conv_kernel = [4, 4]
# conv_stride = [2, 1]

latent_dim = 100

data_prescalers = {"minmax": MinMaxScaler, "std": StandardScaler, "rob": RobustScaler}
data_prescaler = "minmax"  # select data prescaler

normalised_fit = True
# semi_supervised = True
semi_supervised = False
num_train_outliers = 5

# enc_architecture = ["x2", "x4", "x5"]
# enc_architecture = ["x8", "x3", "x2"]
# enc_architecture = ["x2", "x4", "x8"]
# enc_architecture = ["x4", "x8"]
# enc_architecture = ["x2", "x2", "d2"]
enc_architecture = ["x5", "x5", "x2"]
# enc_architecture = ["x4", "x8", "d5"]
# enc_architecture = ["x8", "x4", "x2"]
# enc_architecture = ["x4", "x8", "d5"]
# enc_architecture = ["x2", "x5", "x8"]

# override last activation and clip_data01 if standard scaler is used
if data_prescaler.lower() != "minmax":
    last_activation = "none"
    clip_data_01 = False

exp_params = {
    "architecture": str(enc_architecture),
    "last_activation": last_activation,
    "num_samples": num_samples,
    "random_seed": random_seed,
    "scaler": data_prescaler,
    "clip_data_01": clip_data_01,
    "weight_decay": weight_decay,
    "anchored": anchored,
}

# ==============PREPARE DATA==========================
shuffle = True
data_transform = transforms.Compose([transforms.ToTensor()])
train_batch_size = 100
test_samples = 10000

if train_set_name == "CIFAR":
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data-cifar", train=True, download=True, transform=data_transform
        ),
        batch_size=train_batch_size,
        shuffle=shuffle,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data-cifar", train=False, download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=shuffle,
    )
    ood_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "data-svhn", split="test", download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=shuffle,
    )
elif train_set_name == "SVHN":
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "data-svhn", split="train", download=True, transform=data_transform
        ),
        batch_size=train_batch_size,
        shuffle=shuffle,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "data-svhn", split="test", download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=shuffle,
    )
    ood_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data-cifar", train=False, download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=shuffle,
    )
elif train_set_name == "FashionMNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data-fashion-mnist", train=True, download=True, transform=data_transform
        ),
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data-fashion-mnist", train=False, download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=True,
    )
    ood_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data-mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=test_samples,
        shuffle=True,
    )

elif train_set_name == "MNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data-mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data-mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=test_samples,
        shuffle=True,
    )

    ood_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data-fashion-mnist", train=False, download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=True,
    )

# =================DEFINE BAE========================
if train_set_name == "FashionMNIST" or train_set_name == "MNIST":
    input_dim = 28
    input_channel = 1
else:
    input_dim = 32
    input_channel = 3


# conv_filters = [32, 64, 128]
# conv_kernel = [4, 4, 4]
# conv_stride = [2, 1, 2]

encoder_nodes = [
    input_dim * int(factor[1]) if factor[0] == "x" else int(input_dim / int(factor[1]))
    for factor in enc_architecture
]

# model architecture
conv_architecture = [input_channel] + conv_filters

# specify encoder
# with convolutional layers and hidden dense layer
encoder = Encoder(
    [
        ConvLayers(
            input_dim=input_dim,
            conv_architecture=conv_architecture,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            activation=activation,
            last_activation=activation,
            add_se=add_se,
            norm=norm,
        ),
        # DenseLayers(
        #     architecture=[],
        #     output_size=latent_dim,
        #     activation=activation,
        #     last_activation=activation,
        # norm =norm
        # ),
    ]
)

# specify decoder-mu
decoder_mu = infer_decoder(
    encoder, activation=activation, last_activation=last_activation, norm=norm
)

if decoder_sig_enabled:
    decoder_sig = infer_decoder(
        encoder,
        activation=activation,
        last_activation="none",
        norm=True,
        last_norm=False,
    )

    # decoder_sig = torch.nn.Sequential(
    #     *[
    #         torch.nn.ConvTranspose2d(
    #             32,
    #             input_channel,
    #             kernel_size=[4, 4],
    #             stride=[2, 2],
    #             padding=[0, 0],
    #             output_padding=[0, 0],
    #             bias=False,
    #         ),
    #         # torch.nn.BatchNorm2d(3),
    #         # torch.nn.Tanh(),
    #     ]
    # )

# combine them into autoencoder
if decoder_sig_enabled:
    autoencoder = Autoencoder(encoder, decoder_mu, decoder_sig, skip=skip)
else:
    autoencoder = Autoencoder(encoder, decoder_mu, skip=skip)

# convert into BAE-Ensemble
bae_ensemble = BAE_Ensemble(
    autoencoder=autoencoder,
    anchored=anchored,
    weight_decay=weight_decay,
    num_samples=num_samples,
    likelihood=likelihood,
    learning_rate=lr,
    verbose=False,
    use_cuda=use_cuda,
    homoscedestic_mode=homo_mode,
    sparse=sparse,
)

print(autoencoder)

# ===============FIT BAE CONVERGENCE===========================
# Min lr:1.1e-06 , Max lr: 0.00234
# Min lr:9.68e-08 , Max lr: 2.55e-05 # tanh + skip - norm
# Min lr:1.93e-07 , Max lr: 0.00372 # tanh + skip + norm
# Min lr:1.02e-07 , Max lr: 0.0101 # sigmoid + skip + norm
x_id_test = next(iter(test_loader))[0].detach().cpu().numpy()
x_ood_test = next(iter(ood_loader))[0].detach().cpu().numpy()

x_id_train, label_id_train = next(iter(train_loader))
x_id_train = x_id_train.detach().cpu().numpy()
# x_id_train = next(iter(train_loader))[0].detach().cpu().numpy()
train_loader_temp = convert_dataloader(
    x_id_train,
    batch_size=int(len(x_id_train) / 2),
    shuffle=True,
)

half_iterations = np.clip(int(len(train_loader) / 2), 1, np.inf)

bae_ensemble.init_scheduler(
    half_iterations=half_iterations, min_lr=1.1e-06, max_lr=0.00234
)
bae_ensemble.scheduler_enabled = True

# minimum_lr, maximum_lr, half_iterations = run_auto_lr_range_v3(
#     train_loader, bae_ensemble, run_full=False, window_size=1, num_epochs=15
# )

# bae_ensemble.fit(
#     train_loader,
#     num_epochs=5,
#     mode="mu",
# )

if decoder_sig_enabled:
    bae_ensemble.fit(train_loader, num_epochs=10, mode="sigma", sigma_train="joint")
else:
    bae_ensemble.fit(train_loader, num_epochs=10, mode="mu")
bae_ensemble.set_cuda(False)
# bae_ensemble.fit(
#     train_loader_temp,
#     num_epochs=250,
#     mode="mu",
# )

# cvg = 0
# while cvg == 0:
#     bae_, cvg = bae_fit_convergence_v2(
#         bae_ensemble=bae_ensemble,
#         x=train_loader_temp,
#         num_epoch=100,
#         threshold=1.00,
#     )


plt.figure()
plt.plot(bae_ensemble.losses)

nll_key = "nll_homo" if not bae_ensemble.decoder_sigma_enabled else "nll_sigma"
# nll_key = "se" if not bae_ensemble.decoder_sigma_enabled else "nll_sigma"
# nll_key = "nll_homo" if not bae_ensemble.decoder_sigma_enabled else "y_sigma"

nll_train = bae_ensemble.predict_samples(
    x_id_train,
    select_keys=[
        nll_key,
        "y_mu",
    ],
)

nll_id = bae_ensemble.predict_samples(
    x_id_test,
    select_keys=[
        nll_key,
        "y_mu",
    ],
)

nll_ood = bae_ensemble.predict_samples(x_ood_test, select_keys=[nll_key, "y_mu"])

nll_id_train = nll_train.mean(0)[0]
nll_id_mean = nll_id.mean(0)[0]
nll_ood_mean = nll_ood.mean(0)[0]
y_id_train = nll_train.mean(0)[1]
y_id_mean = nll_id.mean(0)[1]
y_ood_mean = nll_ood.mean(0)[1]

# nll_id_train = nll_train.var(0)[0]
# nll_id_mean = nll_id.var(0)[0]
# nll_ood_mean = nll_ood.var(0)[0]

print(
    "AUROC VALID: {:.5f}".format(
        calc_auroc(
            nll_id_train.mean(-1).mean(-1).mean(-1),
            nll_id_mean.mean(-1).mean(-1).mean(-1),
            # nll_ood_mean.mean(-1).mean(-1).mean(-1),
        )
    )
)

print(
    "AUROC ALL: {:.5f}".format(
        calc_auroc(
            nll_id_mean.mean(-1).mean(-1).mean(-1),
            # nll_id_train.mean(-1).mean(-1).mean(-1),
            nll_ood_mean.mean(-1).mean(-1).mean(-1),
        )
    )
)

for i in range(input_channel):
    print(
        "AUROC {}: {:.5f}".format(
            i,
            calc_auroc(
                nll_id_mean.mean(-1).mean(-1)[:, i],
                nll_ood_mean.mean(-1).mean(-1)[:, i],
            ),
        )
    )

# for i in range(3):
#     print(
#         "X AUROC {}: {:.5f}".format(
#             i,
#             calc_auroc(
#                 x_id_test.mean(-1).mean(-1)[i],
#                 x_ood_test.mean(-1).mean(-1)[i],
#             ),
#         )
#     )

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
ax1.imshow(np.moveaxis(x_id_train[0], 0, 2))
ax2.imshow(np.moveaxis(y_id_train[0], 0, 2))
img1 = ax3.imshow((nll_id_train[0].mean(0)))

ax4.imshow(np.moveaxis(x_id_test[0], 0, 2))
ax5.imshow(np.moveaxis(y_id_mean[0], 0, 2))
img2 = ax6.imshow((nll_id_mean[0].mean(0)))
plt.colorbar(img2)
ax7.imshow(np.moveaxis(x_ood_test[0], 0, 2))
ax8.imshow(np.moveaxis(y_ood_mean[0], 0, 2))
img3 = ax9.imshow((nll_ood_mean[0].mean(0)))
plt.colorbar(img3)
print("NLL ID VS OOD SAMPLE")
print(nll_id_train.mean())
print(nll_id_mean.mean())
print(nll_ood_mean.mean())

print(nll_id_mean[0].mean())
print(nll_ood_mean[0].mean())

# =================HISTOGRAMS====================
plt.figure()
plt.hist(nll_id_train.mean(-1).mean(-1).mean(-1), density=True, alpha=0.5)
plt.hist(nll_id_mean.mean(-1).mean(-1).mean(-1), density=True, alpha=0.5)
plt.hist(nll_ood_mean.mean(-1).mean(-1).mean(-1), density=True, alpha=0.5)

plt.figure()
plt.boxplot(
    [
        nll_id_train.mean(-1).mean(-1).mean(-1),
        nll_id_mean.mean(-1).mean(-1).mean(-1),
        nll_ood_mean.mean(-1).mean(-1).mean(-1),
    ]
)

nll_id_train_channels = nll_id_train.mean(-1).mean(-1)
nll_id_test_channels = nll_id_mean.mean(-1).mean(-1)
nll_ood_test_channels = nll_ood_mean.mean(-1).mean(-1)

fig, axes = plt.subplots(3, 1)
axes = np.array(axes)
for i, ax in zip(range(input_channel), axes):
    ax.hist(nll_id_test_channels[:, i], density=True, alpha=0.5)
    ax.hist(nll_ood_test_channels[:, i], density=True, alpha=0.5)

# ==================CHANNEL WISE================


# nll_id_mean_scaled = nll_id_mean.reshape(10000, -1)
# nll_ood_mean_scaled = nll_ood_mean.reshape(10000, -1)
#
# minmax_scaler = MinMaxScaler().fit(nll_id_mean)
# nll_id_mean_scaled = minmax_scaler.transform(nll_id_mean_scaled)
# nll_ood_mean_scaled = minmax_scaler.transform(nll_ood_mean_scaled)
#
# nll_id_mean = nll_id_mean.reshape(10000, 3, 32, 32)
# nll_ood_mean = nll_ood_mean.reshape(10000, 3, 32, 32)

nll_id_train_pixels = nll_id_train.mean(1)
nll_id_mean_pixels = nll_id_mean.mean(1)
nll_ood_mean_pixels = nll_ood_mean.mean(1)

# nll_id_train_pixels = np.log(nll_id_train.mean(1))
# nll_id_mean_pixels = np.log(nll_id_mean.mean(1))
# nll_ood_mean_pixels = np.log(nll_ood_mean.mean(1))
#
# nll_id_train_pixels = np.log(nll_id_train[0])
# nll_id_mean_pixels = np.log(nll_id_mean[0])
# nll_ood_mean_pixels = np.log(nll_ood_mean[0])

# vmin = np.concatenate(
#     (nll_id_train_pixels, nll_id_mean_pixels, nll_ood_mean_pixels)
# ).min()
# vmax = np.concatenate(
#     (nll_id_train_pixels, nll_id_mean_pixels, nll_ood_mean_pixels)
# ).max()

dummy_samples = np.array(
    [nll_id_train_pixels[0], nll_id_mean_pixels[0], nll_ood_mean_pixels[0]]
)

vmin = dummy_samples.min()
vmax = dummy_samples.max()

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
ax1.imshow(np.moveaxis(x_id_train[0], 0, 2))
ax2.imshow(np.moveaxis(y_id_train[0], 0, 2))
img1 = ax3.imshow(nll_id_train_pixels[0], vmin=vmin, vmax=vmax)

ax4.imshow(np.moveaxis(x_id_test[0], 0, 2))
ax5.imshow(np.moveaxis(y_id_mean[0], 0, 2))
img2 = ax6.imshow(nll_id_mean_pixels[0], vmin=vmin, vmax=vmax)
plt.colorbar(img2)
ax7.imshow(np.moveaxis(x_ood_test[0], 0, 2))
ax8.imshow(np.moveaxis(y_ood_mean[0], 0, 2))
img3 = ax9.imshow(nll_ood_mean_pixels[0], vmin=vmin, vmax=vmax)

print(nll_id_mean[0].mean())
print(nll_ood_mean[0].mean())

# =================LATENT SAMPLES=================
latent_id_train = flatten_np(bae_ensemble.predict_latent_samples(x_id_train)[0])
latent_id_test = flatten_np(bae_ensemble.predict_latent_samples(x_id_test)[0])
latent_ood_test = flatten_np(bae_ensemble.predict_latent_samples(x_ood_test)[0])

latent_id_train = bae_ensemble.predict_latent_samples(x_id_train)[0]
latent_id_test = bae_ensemble.predict_latent_samples(x_id_test)[0]
latent_ood_test = bae_ensemble.predict_latent_samples(x_ood_test)[0]

# plt.figure()
#
# for i in latent_id_test[:500]:
#     plt.plot(i, color="tab:blue", alpha=0.15)
# for i in latent_ood_test[:500]:
#     plt.plot(i, color="tab:orange", alpha=0.15)

# i = -1
# plt.figure()
# plt.hist(latent_id_test[:, i], density=True, alpha=0.5)
# plt.hist(latent_ood_test[:, i], density=True, alpha=0.5)

channel_k = 5
# img1 = latent_id_train.mean(1).mean(0)
# img2 = latent_id_test.mean(1).mean(0)
# img3 = latent_ood_test.mean(1).mean(0)

img1 = latent_id_test.mean(1)[-5000:].mean(0)
img2 = latent_id_test.mean(1)[:5000].mean(0)
img3 = latent_ood_test.mean(1).mean(0)

img1 = latent_id_test.mean(1)[0]
img2 = latent_id_test.mean(1)[-1]
img3 = latent_ood_test.mean(1)[-10]

# channel_k = 5
# img1 = latent_id_train.mean(0)[channel_k]
# img2 = latent_id_test.mean(0)[channel_k]
# img3 = latent_ood_test.mean(0)[channel_k]

# img1 = latent_id_train[0, channel_k]
# img2 = latent_id_test[0, channel_k]
# img3 = latent_ood_test[0, channel_k]

vmin = np.array([img1, img2, img3]).min()
vmax = np.array([img1, img2, img3]).max()

plt.figure()
plt.imshow(img1, vmin=vmin, vmax=vmax)

plt.figure()
plt.imshow(img2, vmin=vmin, vmax=vmax)

plt.figure()
plt.imshow(img3, vmin=vmin, vmax=vmax)


# ===========PCA==============
latent_id_train = bae_ensemble.predict_latent_samples(x_id_train)[0]
latent_id_test = bae_ensemble.predict_latent_samples(x_id_test)[0]
latent_ood_test = bae_ensemble.predict_latent_samples(x_ood_test)[0]


# plt.figure()
# plt.imshow(img1)
#
# plt.figure()
# plt.imshow(img2)
#
# plt.figure()
# plt.imshow(img3)

# =================POST PROC=================

# MinMaxScaler().fit_transform()


# plt.figure()
# plt.boxplot(
#     [nll_id_mean.mean(-1).mean(-1).mean(-1), nll_ood_mean.mean(-1).mean(-1).mean(-1)]
# )
#
# i = 2
# plt.figure()
# plt.boxplot([nll_id_mean.mean(-1).mean(-1)[i], nll_ood_mean.mean(-1).mean(-1)[i]])
#
# plt.figure()
# plt.hist(nll_id_mean.mean(-1).mean(-1).mean(-1), density=True)
# plt.hist(nll_ood_mean.mean(-1).mean(-1).mean(-1), density=True)

#

# train_loader = convert_dataloader(
#     x_inliers_train,
#     batch_size=int(len(x_inliers_train) / 5)
#     if len(x_inliers_train) > 1000
#     else len(x_inliers_train),
#     shuffle=True,
# )

# train_loader = convert_dataloader(
#     x_inliers_train,
#     batch_size=int(len(x_inliers_train) / 2),
#     # batch_size=1,
#     shuffle=True,
# )


#
# # train_loader = convert_dataloader(
# #     x_inliers_train,
# #     batch_size=250,
# #     # batch_size=1,
# #     shuffle=True,
# # )
#
# # bae_ensemble.fit(train_loader, num_epochs=5)
#
# run_auto_lr_range_v3(
#     train_loader, bae_ensemble, run_full=True, window_size=1, num_epochs=15
# )
#
# num_epochs_per_cycle = 50
# fast_window = num_epochs_per_cycle
# slow_window = fast_window * 5
# n_stop_points = 10
# cvg = 0
#
# auroc_valids = []
# auroc_threshold = 0.65
#
# normalisers = []
# ii = 0
# max_ii = 25
# auroc_oods = []
#
# # bae_ensemble.fit(train_loader, num_epochs=1420)
#
#
# while cvg == 0:
#     # bae_ensemble.fit(train_loader, num_epochs=num_epochs_per_cycle)
#
#     # _, cvg = bae_fit_convergence(
#     #     bae_ensemble=bae_ensemble,
#     #     x=train_loader,
#     #     num_epoch=num_epochs_per_cycle,
#     #     fast_window=fast_window,
#     #     slow_window=slow_window,
#     #     n_stop_points=n_stop_points,
#     # )
#
#     bae_, cvg = bae_fit_convergence_v2(
#         bae_ensemble=bae_ensemble,
#         x=train_loader,
#         num_epoch=num_epochs_per_cycle,
#         threshold=1.00,
#     )
#
#     # _, cvg = bae_norm_fit_convergence(
#     #     bae_ensemble=bae_ensemble,
#     #     x=train_loader,
#     #     num_epoch=num_epochs_per_cycle,
#     #     fast_window=fast_window,
#     #     slow_window=slow_window,
#     #     n_stop_points=n_stop_points,
#     # )
#
#     if semi_supervised:
#         bae_ensemble.semisupervised_fit(
#             x_inliers=train_loader,
#             x_outliers=x_outliers_train,
#             num_epochs=int(num_epochs_per_cycle / 2),
#         )
#
#     # if normalised_fit:
#     #     for i in range(num_epochs_per_cycle):
#     #         sampled_dt = next(iter(train_loader))[0]
#     #         bae_ensemble.normalised_fit_one(sampled_dt, mode="mu")
#
#     # auroc_ood = calc_auroc_valid(bae_ensemble, x_inliers_test, x_outliers_test)
#     # print("AUROC-OOD: {:.3f}".format(auroc_ood))
#
#     nll_inliers_train = bae_ensemble.predict_samples(
#         x_inliers_train, select_keys=["nll_homo"]
#     )
#
#     # nll_inliers_train_mean = nll_inliers_train[0][0].mean(-1)
#     normaliser = np.log(np.exp(-nll_inliers_train[:, 0]).mean(-1)).mean(-1).mean(-1)
#
#     print(normaliser)
#     normalisers.append(normaliser)
#
#     auroc_valid = calc_auroc_valid(bae_ensemble, x_inliers_train, x_inliers_valid)
#     print("AUROC-VALID: {:.3f}".format(auroc_valid))
#
#     auroc_ood = calc_auroc_valid(bae_ensemble, x_inliers_test, x_outliers_test)
#     print("AUROC-OOD: {:.3f}".format(auroc_ood))
#     auroc_oods.append(auroc_ood)
#
#     # if auroc_valid >= auroc_threshold:
#     #     break
#     ii += 1
#     if ii >= max_ii:
#         break
#
# #
# #
# # print("NEXT")
# # bae_ensemble.set_learning_rate(learning_rate=bae_ensemble.min_lr)
# #
# # bae_ensemble.scheduler_enabled = False
# #
# # for lr in [0.0001, 0.0005, 0.001]:
# #     bae_ensemble.set_learning_rate(learning_rate=lr)
# #
# #     bae_ensemble.set_optimisers(
# #         bae_ensemble.autoencoder, mode="mu", sigma_train="separate"
# #     )
# #
# #     for i in range(3):
# #
# #         # bae_ensemble.normalised_fit(train_loader, num_epochs=num_epochs_per_cycle)
# #         bae_ensemble.normalised_fit_one(next(iter(train_loader))[0])
# #
# #     nll_inliers_train = bae_ensemble.predict_samples(
# #         x_inliers_train, select_keys=["se"]
# #     )
# #
# #     # nll_inliers_train_mean = nll_inliers_train[0][0].mean(-1)
# #     normaliser = np.log(np.exp(-nll_inliers_train[:, 0]).mean(-1)).mean(-1).mean(-1)
# #
# #     print(normaliser)
# #     normalisers.append(normaliser)
# #
# #     auroc_valid = calc_auroc_valid(bae_ensemble, x_inliers_train, x_inliers_valid)
# #     print("AUROC-VALID: {:.3f}".format(auroc_valid))
# #
# #     auroc_ood = calc_auroc_valid(bae_ensemble, x_inliers_test, x_outliers_test)
# #     print("AUROC-OOD: {:.3f}".format(auroc_ood))
# #     auroc_oods.append(auroc_ood)
#
#
# # for i in range(50):
# #     auroc_ood = calc_auroc_valid(bae_ensemble, x_inliers_test, x_outliers_test)
# #     print("AUROC-OOD: {:.3f}".format(auroc_ood))
# #     auroc_oods.append(auroc_ood)
# #     if normalised_fit:
# #         for i in range(num_epochs_per_cycle):
# #             sampled_dt = next(iter(train_loader))[0]
# #             bae_ensemble.normalised_fit_one(sampled_dt, mode="mu")
# #
# #     nll_inliers_train = bae_ensemble.predict_samples(
# #         x_inliers_train, select_keys=["se"]
# #     )
# #
# #     # nll_inliers_train_mean = nll_inliers_train[0][0].mean(-1)
# #     normaliser = np.log(np.exp(-nll_inliers_train[:, 0]).mean(-1)).mean(-1).mean(-1)
# #
# #     print(normaliser)
# #     normalisers.append(normaliser)
# #
# #     auroc_valid = calc_auroc_valid(bae_ensemble, x_inliers_train, x_inliers_valid)
# #     print("AUROC-VALID: {:.3f}".format(auroc_valid))
# #
# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(normalisers)
# ax2.plot(auroc_oods)
#
# # bae_ensemble = bae_
#
#
# fig, ax = plt.subplots(1, 1)
# plot_convergence(
#     losses=bae_ensemble.losses,
#     fast_window=fast_window,
#     slow_window=slow_window,
#     n_stop_points=n_stop_points,
#     ax=ax,
# )
#
# # ===============PREDICT BAE==========================
#
#
# # nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=["se"])
# # nll_inliers_test = bae_ensemble.predict_samples(x_inliers_test, select_keys=["se"])
# # nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["se"])
# # nll_outliers_test = bae_ensemble.predict_samples(x_outliers_test, select_keys=["se"])
#
# # nll_inliers_test = bae_ensemble.predict_samples(x_inliers_test, select_keys=["se"])
# # nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["se"])
# # nll_outliers_test = bae_ensemble.predict_samples(x_outliers_test, select_keys=["se"])
#
# nll_inliers_test = bae_ensemble.predict_samples(
#     x_inliers_test, select_keys=["nll_homo"]
# )
# nll_inliers_valid = bae_ensemble.predict_samples(
#     x_inliers_valid, select_keys=["nll_homo"]
# )
# nll_outliers_test = bae_ensemble.predict_samples(
#     x_outliers_test, select_keys=["nll_homo"]
# )
#
# # nll_outliers_train = bae_ensemble.predict_samples(x_outliers_train, select_keys=["se"])
#
# # nll_inliers_train_mean = nll_inliers_train.mean(0)[0].mean(-1)
# nll_inliers_test_mean = nll_inliers_test.mean(0)[0].mean(-1)
# nll_inliers_valid_mean = nll_inliers_valid.mean(0)[0].mean(-1)
# nll_outliers_test_mean = nll_outliers_test.mean(0)[0].mean(-1)
# # nll_outliers_train_mean = nll_outliers_train.mean(0)[0].mean(-1)
#
# # nll_inliers_train_var = nll_inliers_train.var(0)[0].mean(-1)
# nll_inliers_test_var = nll_inliers_test.var(0)[0].mean(-1)
# nll_inliers_valid_var = nll_inliers_valid.var(0)[0].mean(-1)
# nll_outliers_test_var = nll_outliers_test.var(0)[0].mean(-1)
# # nll_outliers_train_var = nll_outliers_train.var(0)[0].mean(-1)
#
# # nll_inliers_test_var = nll_inliers_test.mean(-1).var(0)[0]
# # nll_inliers_valid_var = nll_inliers_valid.mean(-1).var(0)[0]
# # nll_outliers_test_var = nll_outliers_test.mean(-1).var(0)[0]
#
# # =======================LAST FEATURE======================
#
#
# def predict_ood_unc(
#     nll_ref,
#     nll_inliers,
#     nll_outliers,
#     p_threshold=0.5,
#     dist_cdf="ecdf",
#     scaling=True,
#     min_level="mean",
# ):
#     # get the NLL (BAE samples)
#     prob_inliers_test, unc_inliers_test = convert_prob(
#         convert_cdf(
#             nll_ref, nll_inliers, dist=dist_cdf, scaling=scaling, min_level=min_level
#         ),
#         *(None, None)
#     )
#     prob_outliers_test, unc_outliers_test = convert_prob(
#         convert_cdf(
#             nll_ref, nll_outliers, dist=dist_cdf, scaling=scaling, min_level=min_level
#         ),
#         *(None, None)
#     )
#
#     prob_inliers_test_mean = prob_inliers_test.mean(0)
#     prob_outliers_test_mean = prob_outliers_test.mean(0)
#
#     total_unc_inliers_test = get_pred_unc(prob_inliers_test, unc_inliers_test)
#     total_unc_outliers_test = get_pred_unc(prob_outliers_test, unc_outliers_test)
#
#     y_true, y_hard_pred, y_unc, y_soft_pred = get_y_results(
#         prob_inliers_test_mean,
#         prob_outliers_test_mean,
#         total_unc_inliers_test,
#         total_unc_outliers_test,
#         p_threshold=p_threshold,
#     )
#
#     return y_true, y_hard_pred, y_unc, y_soft_pred
#
#
# # ================Evaluation==================================
# # 1 : f1 | unc
# # 2 : auprc misclassification (TYPE 1 , TYPE 2, ALL)
# # 3 : AUROC BINARY CLASSIFICATION
# exp_man = ExperimentManager()
#
# y_true = get_y_true(nll_inliers_test_mean, nll_outliers_test_mean)
#
# # auroc ood
# res_auroc_ood = eval_auroc_ood(
#     y_true,
#     {
#         "nll_mean": np.concatenate((nll_inliers_test_mean, nll_outliers_test_mean)),
#         "nll_var": np.concatenate((nll_inliers_test_var, nll_outliers_test_var)),
#     },
# )
#
# # loop over different cdf distributions and type of uncertainties
#
# for dist_cdf in [
#     "norm",
#     "uniform",
#     "expon",
#     "ecdf",
# ]:
#     for min_level in ["mean", "quartile"]:
#         scaling = False if dist_cdf == "uniform" else True
#         y_true, y_hard_pred, y_unc, y_soft_pred = predict_ood_unc(
#             nll_ref=nll_inliers_valid.mean(-1)[:, 0],
#             nll_inliers=nll_inliers_test.mean(-1)[:, 0],
#             nll_outliers=nll_outliers_test.mean(-1)[:, 0],
#             p_threshold=0.5,
#             dist_cdf=dist_cdf,
#             scaling=scaling,
#             min_level=min_level,
#         )
#
#         res_temp = eval_auroc_ood(
#             y_true,
#             {
#                 dist_cdf: y_soft_pred,
#             },
#         )
#         res_auroc_ood = res_auroc_ood.append(res_temp)
#
#         for unc_type in ["epistemic", "aleatoric", "total"]:
#
#             # rejection metric
#             y_unc_scaled = np.round(y_unc[unc_type] * 4, 2)
#             (
#                 retained_metrics,
#                 res_wmean,
#                 res_max,
#                 res_spman,
#                 res_baselines,
#             ) = eval_retained_unc(y_true, y_hard_pred, y_unc_scaled, y_soft_pred)
#
#             # append the base auroc ood
#             # res_auroc_prob = res_baselines.copy().drop(["f1", "mcc", "gmean_ss"], axis=1)
#             # res_auroc_prob.columns = [
#             #     col + "-" + dist_cdf for col in res_auroc_prob.columns
#             # ]
#             # res_auroc_ood = pd.concat((res_auroc_ood, res_auroc_prob), axis=1)
#
#             # (
#             #     retained_metrics,
#             #     res_wmean,
#             #     res_max,
#             #     res_spman,
#             #     res_baselines,
#             # ) = eval_retained_unc(
#             #     y_true,
#             #     y_hard_pred,
#             #     np.concatenate((nll_inliers_test_var, nll_outliers_test_var)),
#             #     np.concatenate((nll_inliers_test_mean, nll_outliers_test_mean)),
#             # )
#
#             res_wmean = res_wmean.drop(["threshold", "perc"], axis=1)
#             res_retained = rename_col_res(res_baselines, res_wmean, res_max, res_spman)
#
#             # misclassification error prediction @ p_threshold = 0.5
#             res_auprc_err = evaluate_error_unc(
#                 y_true, y_hard_pred, y_unc[unc_type], verbose=False
#             )
#
#             # print("---RETAINED-PERC----")
#             # print(res_retained.T)
#             # print("---AUPR-ERR-----")
#             # print(res_auprc_err.T)
#             # print("----AUROC-OOD---")
#             # print(res_auroc_ood.T)
#
#             # ===========LOG RESULTS WITH EXP MANAGER========
#             exp_params_temp = exp_params.copy()
#             exp_params_temp.update(
#                 {"dist_cdf": dist_cdf, "unc_type": unc_type, "min_level": min_level}
#             )
#
#             exp_row_err = concat_params_res(exp_params_temp, res_auprc_err)
#             exp_row_retained = concat_params_res(exp_params_temp, res_retained)
#
#             exp_man.update_csv(exp_row_err, csv_name="auprc_err.csv")
#             exp_man.update_csv(exp_row_retained, csv_name="retained.csv")
#
# exp_row_auroc_ood = concat_params_res(exp_params, res_auroc_ood)
# exp_man.update_csv(exp_row_auroc_ood, csv_name="auroc_ood.csv")
#
# # =======================PLOT CONVERSION OF NLL TO OUTLIER PERC======================
# dist_ = "norm"
# scaling_cdf = True
# cdf_valid_id = convert_cdf(
#     nll_inliers_valid.mean(-1)[:, 0],
#     nll_inliers_valid.mean(-1)[:, 0],
#     dist=dist_,
#     scaling=scaling_cdf,
# )
# cdf_test_id = convert_cdf(
#     nll_inliers_valid.mean(-1)[:, 0],
#     nll_inliers_test.mean(-1)[:, 0],
#     dist=dist_,
#     scaling=scaling_cdf,
# )
# cdf_test_ood = convert_cdf(
#     nll_inliers_valid.mean(-1)[:, 0],
#     nll_outliers_test.mean(-1)[:, 0],
#     dist=dist_,
#     scaling=scaling_cdf,
# )
#
# kk = 0
# nll_inliers = nll_inliers_test.mean(-1)[:, 0]
# # nll_inliers = nll_inliers_valid.mean(-1)[:, 0]
# nll_outliers = nll_outliers_test.mean(-1)[:, 0]
# nll_total = np.concatenate((nll_inliers, nll_outliers), axis=1)[kk]
# cdf_total = np.concatenate((cdf_test_id, cdf_test_ood), axis=1)[kk]
#
# fig, ax = plt.subplots(1, 1)
# ax.scatter(nll_total, cdf_total)
# ax2 = ax.twinx()
# ax2.hist(nll_inliers[kk], density=True, alpha=0.5)
# ax2.hist(nll_outliers[kk], density=True, alpha=0.5)
#
# print(len(np.argwhere(cdf_test_id >= 0.999)))
#
# for kk in range(num_samples):
#     print(calc_auroc(cdf_test_id[kk], cdf_test_ood[kk]))
# print(calc_auroc(cdf_test_id.mean(0), cdf_test_ood.mean(0)))
#
# plt.figure()
# plt.hist(cdf_test_id.mean(0))
# plt.hist(cdf_test_ood.mean(0))
#
# plt.figure()
# plt.hist(cdf_test_id.std(0))
# plt.hist(cdf_test_ood.std(0))
#
# # ============GMM ================
#
# import numpy as np
# from sklearn.mixture import GaussianMixture
#
# bics = []
# max_gmm_search = 3
# rd_seed = 1
# nll_scaler = RobustScaler()
# nll_inliers_valid_scaled = nll_scaler.fit_transform(
#     nll_inliers_valid.mean(0)[0].mean(-1).reshape(-1, 1)
# )
# nll_inliers_test_scaled = nll_scaler.transform(
#     nll_inliers_test.mean(0)[0].mean(-1).reshape(-1, 1)
# )
#
# for k_components in range(2, max_gmm_search + 1):
#     gm = GaussianMixture(n_components=k_components, random_state=rd_seed).fit(
#         nll_inliers_valid_scaled
#     )
#     gm_bic = gm.bic(nll_inliers_valid_scaled)
#     bics.append(gm_bic)
#     # print("BIC : {:.3f}".format(gm_bic))
#
# # select k
# best_k = np.argmin(bics) + 2
# gm_squash = GaussianMixture(n_components=best_k, random_state=rd_seed).fit(
#     nll_inliers_valid_scaled
# )
# gmm_proba = gm_squash.predict_proba(nll_inliers_test_scaled)
# select_k = np.argmax(gmm_proba[np.argmin(nll_inliers_test_scaled)])
# outlier_proba = 1 - gmm_proba[:, select_k]
#
# # ===========================================
