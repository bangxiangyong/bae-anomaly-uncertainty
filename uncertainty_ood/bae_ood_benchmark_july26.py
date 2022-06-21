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

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2, run_auto_lr_range_v3
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import (
    Encoder,
    infer_decoder,
    Autoencoder,
)
from baetorch.baetorch.models.base_layer import DenseLayers
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
random_seed = 3

bae_set_seed(random_seed)

use_cuda = True
clip_data_01 = True
activation = "leakyrelu"
last_activation = "sigmoid"  # tanh
likelihood = "gaussian"  # gaussian
homo_mode = "none"
train_size = 0.75
num_samples = 5
# weight_decay = 0.0001
weight_decay = 0.0001
lr = 0.005
# lr = 0.0085
anchored = True
# sparse = False
sparse = False
skip = False

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
# FOLDER 1 : specify data
base_folder = "od_benchmark"
mat_file_list = os.listdir(base_folder)
mat_file = mat_file_list[0]
mat = loadmat(os.path.join(base_folder, mat_file))
X = mat["X"]
y = mat["y"].ravel()

# X = X[:, [242, 273, 107, 212, 271, 252, 173, 172, 243]]
# X = X[:, [-242, -273, -107, -212, -271, -252, -173, -172, -243]]

# FOLDER 2 : specify data
# base_folder = "od_benchmark2"
# mat_file_list = os.listdir(base_folder)
# mat_file = mat_file_list[-4]
# X, y = load_benchmark_dt(base_folder, mat_file)

# get outliers and inliers
x_outliers, x_inliers = get_outliers_inliers(X, y)

# update exp params
exp_params.update({"dataset": mat_file.split("_")[0] if "_" else mat_file in mat_file})

if semi_supervised:
    x_outliers_train, x_outliers_test = train_test_split(
        x_outliers,
        train_size=np.round(num_train_outliers / len(x_outliers), 2),
        shuffle=True,
        random_state=random_seed,
    )
else:
    x_outliers_train = x_outliers.copy()
    x_outliers_test = x_outliers.copy()

x_inliers_train, x_inliers_test = train_test_split(
    x_inliers, train_size=train_size, shuffle=True, random_state=random_seed
)
x_inliers_train, x_inliers_valid = train_test_split(
    x_inliers_train, train_size=train_size, shuffle=True, random_state=random_seed
)


# =================SCALER=========================
scaler = data_prescalers[data_prescaler]()

scaler = scaler.fit(x_inliers_train)
x_inliers_train = scaler.transform(x_inliers_train)
x_inliers_valid = scaler.transform(x_inliers_valid)
x_inliers_test = scaler.transform(x_inliers_test)
x_outliers_test = scaler.transform(x_outliers_test)
x_outliers_train = scaler.transform(x_outliers_train)

if clip_data_01:
    x_inliers_train = np.clip(x_inliers_train, 0, 1)
    x_inliers_test = np.clip(x_inliers_test, 0, 1)
    x_outliers_test = np.clip(x_outliers_test, 0, 1)
    x_inliers_valid = np.clip(x_inliers_valid, 0, 1)
    x_outliers_train = np.clip(x_outliers_train, 0, 1)

# =================DEFINE BAE========================

input_dim = x_inliers_train.shape[-1]

# encoder_nodes = [input_dim*2,20]
# encoder_nodes = [input_dim*4,int(input_dim/3)]
# encoder_nodes = [input_dim*4,100]
# encoder_nodes = [input_dim*4,80]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*8,input_dim]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*6]
# encoder_nodes = [input_dim*2,input_dim*4,2]
# encoder_nodes = [input_dim*8,input_dim]
# encoder_nodes = [input_dim*6,input_dim*4,input_dim*8]
# encoder_nodes = [input_dim*6,input_dim*4]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*6]
# encoder_nodes = [input_dim*4,input_dim*4,int(input_dim/3)]
# encoder_nodes = [input_dim*8,input_dim*4,int(input_dim*5)]
# encoder_nodes = [input_dim*8,int(input_dim/3)]
# encoder_nodes = [input_dim*8,input_dim*4,int(input_dim/2)]
# encoder_nodes = [input_dim * 2, input_dim * 4, input_dim * 8]
# encoder_nodes = [input_dim * 8, input_dim * 4, input_dim * 2]
# encoder_nodes = [input_dim * factor for factor in enc_architecture]
# encoder_nodes = [
#     input_dim * factor if i + 1 < len(enc_architecture) else int(input_dim / factor)
#     for i, factor in enumerate(enc_architecture)
# ]
encoder_nodes = [
    input_dim * int(factor[1]) if factor[0] == "x" else int(input_dim / int(factor[1]))
    for factor in enc_architecture
]

encoder = Encoder(
    [
        DenseLayers(
            input_size=input_dim,
            architecture=encoder_nodes[:-1],
            output_size=encoder_nodes[-1],
            activation=activation,
            last_activation=activation,
        )
    ]
)

# specify decoder-mu
decoder_mu = infer_decoder(
    encoder, activation=activation, last_activation=last_activation
)  # symmetrical to encoder

# combine them into autoencoder
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

# ===============FIT BAE CONVERGENCE===========================
# train_loader = convert_dataloader(
#     x_inliers_train,
#     batch_size=int(len(x_inliers_train) / 5)
#     if len(x_inliers_train) > 1000
#     else len(x_inliers_train),
#     shuffle=True,
# )

train_loader = convert_dataloader(
    x_inliers_train,
    batch_size=int(len(x_inliers_train) / 2),
    # batch_size=1,
    shuffle=True,
)

# train_loader = convert_dataloader(
#     x_inliers_train,
#     batch_size=250,
#     # batch_size=1,
#     shuffle=True,
# )

# bae_ensemble.fit(train_loader, num_epochs=5)

# run_auto_lr_range_v3(
#     train_loader, bae_ensemble, run_full=True, window_size=1, num_epochs=15
# )

num_epochs_per_cycle = 50
fast_window = num_epochs_per_cycle
slow_window = fast_window * 5
n_stop_points = 10
cvg = 0

auroc_valids = []
auroc_threshold = 0.65

normalisers = []
ii = 0
max_ii = 25
auroc_oods = []

# bae_ensemble.fit(train_loader, num_epochs=1420)


while cvg == 0:
    # bae_ensemble.fit(train_loader, num_epochs=num_epochs_per_cycle)

    # _, cvg = bae_fit_convergence(
    #     bae_ensemble=bae_ensemble,
    #     x=train_loader,
    #     num_epoch=num_epochs_per_cycle,
    #     fast_window=fast_window,
    #     slow_window=slow_window,
    #     n_stop_points=n_stop_points,
    # )

    bae_, cvg = bae_fit_convergence_v2(
        bae_ensemble=bae_ensemble,
        x=train_loader,
        num_epoch=num_epochs_per_cycle,
        threshold=1.00,
    )

    # _, cvg = bae_norm_fit_convergence(
    #     bae_ensemble=bae_ensemble,
    #     x=train_loader,
    #     num_epoch=num_epochs_per_cycle,
    #     fast_window=fast_window,
    #     slow_window=slow_window,
    #     n_stop_points=n_stop_points,
    # )

    if semi_supervised:
        bae_ensemble.semisupervised_fit(
            x_inliers=train_loader,
            x_outliers=x_outliers_train,
            num_epochs=int(num_epochs_per_cycle / 2),
        )

    # if normalised_fit:
    #     for i in range(num_epochs_per_cycle):
    #         sampled_dt = next(iter(train_loader))[0]
    #         bae_ensemble.normalised_fit_one(sampled_dt, mode="mu")

    # auroc_ood = calc_auroc_valid(bae_ensemble, x_inliers_test, x_outliers_test)
    # print("AUROC-OOD: {:.3f}".format(auroc_ood))

    nll_inliers_train = bae_ensemble.predict_samples(
        x_inliers_train, select_keys=["nll_homo"]
    )

    # nll_inliers_train_mean = nll_inliers_train[0][0].mean(-1)
    normaliser = np.log(np.exp(-nll_inliers_train[:, 0]).mean(-1)).mean(-1).mean(-1)

    print(normaliser)
    normalisers.append(normaliser)

    auroc_valid = calc_auroc_valid(bae_ensemble, x_inliers_train, x_inliers_valid)
    print("AUROC-VALID: {:.3f}".format(auroc_valid))

    auroc_ood = calc_auroc_valid(bae_ensemble, x_inliers_test, x_outliers_test)
    print("AUROC-OOD: {:.3f}".format(auroc_ood))
    auroc_oods.append(auroc_ood)

    # if auroc_valid >= auroc_threshold:
    #     break
    ii += 1
    if ii >= max_ii:
        break

#
#
# print("NEXT")
# bae_ensemble.set_learning_rate(learning_rate=bae_ensemble.min_lr)
#
# bae_ensemble.scheduler_enabled = False
#
# for lr in [0.0001, 0.0005, 0.001]:
#     bae_ensemble.set_learning_rate(learning_rate=lr)
#
#     bae_ensemble.set_optimisers(
#         bae_ensemble.autoencoder, mode="mu", sigma_train="separate"
#     )
#
#     for i in range(3):
#
#         # bae_ensemble.normalised_fit(train_loader, num_epochs=num_epochs_per_cycle)
#         bae_ensemble.normalised_fit_one(next(iter(train_loader))[0])
#
#     nll_inliers_train = bae_ensemble.predict_samples(
#         x_inliers_train, select_keys=["se"]
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


# for i in range(50):
#     auroc_ood = calc_auroc_valid(bae_ensemble, x_inliers_test, x_outliers_test)
#     print("AUROC-OOD: {:.3f}".format(auroc_ood))
#     auroc_oods.append(auroc_ood)
#     if normalised_fit:
#         for i in range(num_epochs_per_cycle):
#             sampled_dt = next(iter(train_loader))[0]
#             bae_ensemble.normalised_fit_one(sampled_dt, mode="mu")
#
#     nll_inliers_train = bae_ensemble.predict_samples(
#         x_inliers_train, select_keys=["se"]
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
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(normalisers)
ax2.plot(auroc_oods)

# bae_ensemble = bae_


fig, ax = plt.subplots(1, 1)
plot_convergence(
    losses=bae_ensemble.losses,
    fast_window=fast_window,
    slow_window=slow_window,
    n_stop_points=n_stop_points,
    ax=ax,
)

# ===============PREDICT BAE==========================


# nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=["se"])
# nll_inliers_test = bae_ensemble.predict_samples(x_inliers_test, select_keys=["se"])
# nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["se"])
# nll_outliers_test = bae_ensemble.predict_samples(x_outliers_test, select_keys=["se"])

# nll_inliers_test = bae_ensemble.predict_samples(x_inliers_test, select_keys=["se"])
# nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["se"])
# nll_outliers_test = bae_ensemble.predict_samples(x_outliers_test, select_keys=["se"])

nll_inliers_test = bae_ensemble.predict_samples(
    x_inliers_test, select_keys=["nll_homo"]
)
nll_inliers_valid = bae_ensemble.predict_samples(
    x_inliers_valid, select_keys=["nll_homo"]
)
nll_outliers_test = bae_ensemble.predict_samples(
    x_outliers_test, select_keys=["nll_homo"]
)

# nll_outliers_train = bae_ensemble.predict_samples(x_outliers_train, select_keys=["se"])

# nll_inliers_train_mean = nll_inliers_train.mean(0)[0].mean(-1)
nll_inliers_test_mean = nll_inliers_test.mean(0)[0].mean(-1)
nll_inliers_valid_mean = nll_inliers_valid.mean(0)[0].mean(-1)
nll_outliers_test_mean = nll_outliers_test.mean(0)[0].mean(-1)
# nll_outliers_train_mean = nll_outliers_train.mean(0)[0].mean(-1)

# nll_inliers_train_var = nll_inliers_train.var(0)[0].mean(-1)
nll_inliers_test_var = nll_inliers_test.var(0)[0].mean(-1)
nll_inliers_valid_var = nll_inliers_valid.var(0)[0].mean(-1)
nll_outliers_test_var = nll_outliers_test.var(0)[0].mean(-1)
# nll_outliers_train_var = nll_outliers_train.var(0)[0].mean(-1)

# nll_inliers_test_var = nll_inliers_test.mean(-1).var(0)[0]
# nll_inliers_valid_var = nll_inliers_valid.mean(-1).var(0)[0]
# nll_outliers_test_var = nll_outliers_test.mean(-1).var(0)[0]

# =======================LAST FEATURE======================


def predict_ood_unc(
    nll_ref,
    nll_inliers,
    nll_outliers,
    p_threshold=0.5,
    dist_cdf="ecdf",
    scaling=True,
    min_level="mean",
):
    # get the NLL (BAE samples)
    prob_inliers_test, unc_inliers_test = convert_prob(
        convert_cdf(
            nll_ref, nll_inliers, dist=dist_cdf, scaling=scaling, min_level=min_level
        ),
        *(None, None)
    )
    prob_outliers_test, unc_outliers_test = convert_prob(
        convert_cdf(
            nll_ref, nll_outliers, dist=dist_cdf, scaling=scaling, min_level=min_level
        ),
        *(None, None)
    )

    prob_inliers_test_mean = prob_inliers_test.mean(0)
    prob_outliers_test_mean = prob_outliers_test.mean(0)

    total_unc_inliers_test = get_pred_unc(prob_inliers_test, unc_inliers_test)
    total_unc_outliers_test = get_pred_unc(prob_outliers_test, unc_outliers_test)

    y_true, y_hard_pred, y_unc, y_soft_pred = get_y_results(
        prob_inliers_test_mean,
        prob_outliers_test_mean,
        total_unc_inliers_test,
        total_unc_outliers_test,
        p_threshold=p_threshold,
    )

    return y_true, y_hard_pred, y_unc, y_soft_pred


# ================Evaluation==================================
# 1 : f1 | unc
# 2 : auprc misclassification (TYPE 1 , TYPE 2, ALL)
# 3 : AUROC BINARY CLASSIFICATION
exp_man = ExperimentManager()

y_true = get_y_true(nll_inliers_test_mean, nll_outliers_test_mean)

# auroc ood
res_auroc_ood = eval_auroc_ood(
    y_true,
    {
        "nll_mean": np.concatenate((nll_inliers_test_mean, nll_outliers_test_mean)),
        "nll_var": np.concatenate((nll_inliers_test_var, nll_outliers_test_var)),
    },
)

# loop over different cdf distributions and type of uncertainties

for dist_cdf in [
    "norm",
    "uniform",
    "expon",
    "ecdf",
]:
    for min_level in ["mean", "quartile"]:
        scaling = False if dist_cdf == "uniform" else True
        y_true, y_hard_pred, y_unc, y_soft_pred = predict_ood_unc(
            nll_ref=nll_inliers_valid.mean(-1)[:, 0],
            nll_inliers=nll_inliers_test.mean(-1)[:, 0],
            nll_outliers=nll_outliers_test.mean(-1)[:, 0],
            p_threshold=0.5,
            dist_cdf=dist_cdf,
            scaling=scaling,
            min_level=min_level,
        )

        res_temp = eval_auroc_ood(
            y_true,
            {
                dist_cdf: y_soft_pred,
            },
        )
        res_auroc_ood = res_auroc_ood.append(res_temp)

        for unc_type in ["epistemic", "aleatoric", "total"]:

            # rejection metric
            y_unc_scaled = np.round(y_unc[unc_type] * 4, 2)
            (
                retained_metrics,
                res_wmean,
                res_max,
                res_spman,
                res_baselines,
            ) = eval_retained_unc(y_true, y_hard_pred, y_unc_scaled, y_soft_pred)

            # append the base auroc ood
            # res_auroc_prob = res_baselines.copy().drop(["f1", "mcc", "gmean_ss"], axis=1)
            # res_auroc_prob.columns = [
            #     col + "-" + dist_cdf for col in res_auroc_prob.columns
            # ]
            # res_auroc_ood = pd.concat((res_auroc_ood, res_auroc_prob), axis=1)

            # (
            #     retained_metrics,
            #     res_wmean,
            #     res_max,
            #     res_spman,
            #     res_baselines,
            # ) = eval_retained_unc(
            #     y_true,
            #     y_hard_pred,
            #     np.concatenate((nll_inliers_test_var, nll_outliers_test_var)),
            #     np.concatenate((nll_inliers_test_mean, nll_outliers_test_mean)),
            # )

            res_wmean = res_wmean.drop(["threshold", "perc"], axis=1)
            res_retained = rename_col_res(res_baselines, res_wmean, res_max, res_spman)

            # misclassification error prediction @ p_threshold = 0.5
            res_auprc_err = evaluate_error_unc(
                y_true, y_hard_pred, y_unc[unc_type], verbose=False
            )

            # print("---RETAINED-PERC----")
            # print(res_retained.T)
            # print("---AUPR-ERR-----")
            # print(res_auprc_err.T)
            # print("----AUROC-OOD---")
            # print(res_auroc_ood.T)

            # ===========LOG RESULTS WITH EXP MANAGER========
            exp_params_temp = exp_params.copy()
            exp_params_temp.update(
                {"dist_cdf": dist_cdf, "unc_type": unc_type, "min_level": min_level}
            )

            exp_row_err = concat_params_res(exp_params_temp, res_auprc_err)
            exp_row_retained = concat_params_res(exp_params_temp, res_retained)

            exp_man.update_csv(exp_row_err, csv_name="auprc_err.csv")
            exp_man.update_csv(exp_row_retained, csv_name="retained.csv")

exp_row_auroc_ood = concat_params_res(exp_params, res_auroc_ood)
exp_man.update_csv(exp_row_auroc_ood, csv_name="auroc_ood.csv")

# =======================PLOT CONVERSION OF NLL TO OUTLIER PERC======================
dist_ = "norm"
scaling_cdf = True
cdf_valid_id = convert_cdf(
    nll_inliers_valid.mean(-1)[:, 0],
    nll_inliers_valid.mean(-1)[:, 0],
    dist=dist_,
    scaling=scaling_cdf,
)
cdf_test_id = convert_cdf(
    nll_inliers_valid.mean(-1)[:, 0],
    nll_inliers_test.mean(-1)[:, 0],
    dist=dist_,
    scaling=scaling_cdf,
)
cdf_test_ood = convert_cdf(
    nll_inliers_valid.mean(-1)[:, 0],
    nll_outliers_test.mean(-1)[:, 0],
    dist=dist_,
    scaling=scaling_cdf,
)

kk = 0
nll_inliers = nll_inliers_test.mean(-1)[:, 0]
# nll_inliers = nll_inliers_valid.mean(-1)[:, 0]
nll_outliers = nll_outliers_test.mean(-1)[:, 0]
nll_total = np.concatenate((nll_inliers, nll_outliers), axis=1)[kk]
cdf_total = np.concatenate((cdf_test_id, cdf_test_ood), axis=1)[kk]

fig, ax = plt.subplots(1, 1)
ax.scatter(nll_total, cdf_total)
ax2 = ax.twinx()
ax2.hist(nll_inliers[kk], density=True, alpha=0.5)
ax2.hist(nll_outliers[kk], density=True, alpha=0.5)

print(len(np.argwhere(cdf_test_id >= 0.999)))

for kk in range(num_samples):
    print(calc_auroc(cdf_test_id[kk], cdf_test_ood[kk]))
print(calc_auroc(cdf_test_id.mean(0), cdf_test_ood.mean(0)))

plt.figure()
plt.hist(cdf_test_id.mean(0))
plt.hist(cdf_test_ood.mean(0))

plt.figure()
plt.hist(cdf_test_id.std(0))
plt.hist(cdf_test_ood.std(0))

# ============GMM ================

import numpy as np
from sklearn.mixture import GaussianMixture

bics = []
max_gmm_search = 3
rd_seed = 1
nll_scaler = RobustScaler()
nll_inliers_valid_scaled = nll_scaler.fit_transform(
    nll_inliers_valid.mean(0)[0].mean(-1).reshape(-1, 1)
)
nll_inliers_test_scaled = nll_scaler.transform(
    nll_inliers_test.mean(0)[0].mean(-1).reshape(-1, 1)
)

for k_components in range(2, max_gmm_search + 1):
    gm = GaussianMixture(n_components=k_components, random_state=rd_seed).fit(
        nll_inliers_valid_scaled
    )
    gm_bic = gm.bic(nll_inliers_valid_scaled)
    bics.append(gm_bic)
    # print("BIC : {:.3f}".format(gm_bic))

# select k
best_k = np.argmin(bics) + 2
gm_squash = GaussianMixture(n_components=best_k, random_state=rd_seed).fit(
    nll_inliers_valid_scaled
)
gmm_proba = gm_squash.predict_proba(nll_inliers_test_scaled)
select_k = np.argmax(gmm_proba[np.argmin(nll_inliers_test_scaled)])
outlier_proba = 1 - gmm_proba[:, select_k]

# ===========================================
