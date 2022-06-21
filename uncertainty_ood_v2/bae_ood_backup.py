import copy
import os

from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    auc,
)
from sklearn.model_selection import train_test_split
from typing import Union

from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.evaluation import (
    calc_auroc,
    calc_avgprc,
    calc_auprc,
    calc_avgprc_perf,
    evaluate_misclas_detection,
    concat_ood_score,
    evaluate_retained_unc,
    evaluate_random_retained_unc,
    retained_top_unc_indices,
)
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood.exceed import ExCeeD
from uncertainty_ood_v2.util.get_predictions import (
    calc_e_nll,
    calc_var_nll,
    flatten_nll,
    calc_exceed,
)
from uncertainty_ood_v2.util.prepare_anomaly_datasets import (
    list_anomaly_dataset,
    load_anomaly_dataset,
)

import matplotlib.pyplot as plt
import numpy as np

# ===EXPERIMENT PARAMETERS===
# fixed variables
from util.uncertainty import (
    convert_cdf,
    calc_outlier_unc,
    convert_hard_pred,
    plot_unc_tptnfpfn,
    get_indices_error,
)

base_folder = "anomaly_datasets"
train_size = 0.8
valid_size = 0.25

# manipulated variables
random_seed = 99
dataset_id = 10
minmax_clip = True

# set random seed
bae_set_seed(random_seed)

# ===DATASET PREPARATION===
anomaly_datasets = list_anomaly_dataset()
if valid_size == 0:
    x_id_train, x_id_test, x_ood = load_anomaly_dataset(
        mat_file_id=anomaly_datasets[dataset_id],
        base_folder=base_folder,
        train_size=train_size,
        random_seed=random_seed,
        minmax_clip=minmax_clip,
    )
else:
    x_id_train, x_id_test, x_ood, x_valid = load_anomaly_dataset(
        mat_file_id=anomaly_datasets[dataset_id],
        base_folder=base_folder,
        train_size=train_size,
        random_seed=random_seed,
        valid_size=valid_size,
        minmax_clip=minmax_clip,
    )

# select features
# features = [18, 12, 17, 3]
# x_id_train = x_id_train[:, features]
# x_id_test = x_id_test[:, features]
# x_ood = x_ood[:, features]
# if valid_size > 0:
#     x_valid = x_valid[:, features]

x_id_train_loader = convert_dataloader(
    x_id_train, batch_size=len(x_id_train) // 3, shuffle=True, drop_last=True
)


# ===DEFINE BAE===

# bae params
# skip = True
skip = False
use_cuda = True
# twin_output = True
twin_output = False
# homoscedestic_mode = "every"
homoscedestic_mode = "none"
likelihood = "gaussian"
# likelihood = "laplace"
# likelihood = "bernoulli"
# likelihood = "cbernoulli"
# likelihood = "truncated_gaussian"
weight_decay = 0.0000000001
# weight_decay = 0.000000001
# weight_decay = 0.0001
# weight_decay = 0.01
# weight_decay = 0.000001
# weight_decay = 0.000
anchored = False
# anchored = True
sparse_scale = 0.0000001
# sparse_scale = 0.00
n_stochastic_samples = 100
# n_ensemble = 10
n_ensemble = 100
# n_ensemble = 1
# n_ensemble = 3
input_dim = x_id_train.shape[1]
num_epochs = 300
norm = True

chain_params = [
    {
        "base": "linear",
        "architecture": [
            input_dim,
            # input_dim * 100,
            input_dim * 8,
            input_dim * 5,
            input_dim * 2,
            # input_dim * 3,
        ],
        # "architecture": [input_dim, input_dim * 4, input_dim * 4, input_dim * 4],
        # "architecture": [input_dim, input_dim * 4, input_dim * 4, input_dim // 8],
        # "architecture": [
        #     input_dim,
        #     input_dim * 10,
        #     input_dim * 10,
        #     input_dim * 10,
        #     input_dim * 10,
        #     input_dim * 10,
        # ],
        "activation": "leakyrelu",
        "norm": norm,
    }
]

# bae_model = BAE_Ensemble(
#     chain_params=chain_params,
#     last_activation="sigmoid",
#     last_norm=False,
#     twin_output=twin_output,
#     twin_params={"activation": "none", "norm": False},
#     # twin_params={"activation": "leakyrelu", "norm": False},  # truncated gaussian
#     # twin_params={"activation": "leakyrelu", "norm": True},  # truncated gaussian
#     # twin_params={"activation": "leakyrelu", "norm": False}
#     # if likelihood == "gaussian"
#     # else {"activation": "leakyrelu", "norm": False},
#     # twin_params={"activation": "softplus", "norm": True},
#     # twin_params={"activation": "softplus", "norm": False},
#     skip=skip,
#     use_cuda=use_cuda,
#     homoscedestic_mode=homoscedestic_mode,
#     likelihood=likelihood,
#     weight_decay=weight_decay,
#     num_samples=n_ensemble,
#     sparse_scale=sparse_scale,
#     anchored=anchored,
# )


bae_model = BAE_SGHMC(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=False,
    twin_output=twin_output,
    # twin_params={"activation": "none", "norm": False},
    # twin_params={"activation": "leakyrelu", "norm": False},  # truncated gaussian
    # twin_params={"activation": "leakyrelu", "norm": True},  # truncated gaussian
    # twin_params={"activation": "leakyrelu", "norm": False}
    # if likelihood == "gaussian"
    # else {"activation": "leakyrelu", "norm": False},
    twin_params={"activation": "softplus", "norm": True},
    # twin_params={"activation": "softplus", "norm": True},
    skip=skip,
    use_cuda=use_cuda,
    homoscedestic_mode=homoscedestic_mode,
    likelihood=likelihood,
    weight_decay=weight_decay,
    num_samples=n_ensemble,
    # anchored=True,
)

min_lr, max_lr, half_iter = run_auto_lr_range_v4(
    x_id_train_loader,
    bae_model,
    window_size=1,
    num_epochs=10,
    run_full=False,
)

if isinstance(bae_model, BAE_SGHMC):
    bae_model.fit(x_id_train_loader, burn_epoch=num_epochs, sghmc_epoch=num_epochs // 2)
else:
    time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)


# === PREDICTIONS ===
# start predicting
nll_key = "nll"

bae_id_pred = bae_model.predict(x_id_test, select_keys=[nll_key])
bae_ood_pred = bae_model.predict(x_ood, select_keys=[nll_key])

# get ood scores
e_nll_id = calc_e_nll(bae_id_pred)
e_nll_ood = calc_e_nll(bae_ood_pred)
var_nll_id = calc_var_nll(bae_id_pred)
var_nll_ood = calc_var_nll(bae_ood_pred)

# convert to outlier probability
# 1. get reference distribution of NLL scores
if valid_size == 0:
    bae_id_ref_pred = bae_model.predict(x_id_train, select_keys=[nll_key])
else:
    bae_id_ref_pred = bae_model.predict(x_valid, select_keys=[nll_key])

# 2. define cdf distribution of OOD scores
cdf_dist = "uniform"
unc_type = "total"
scaling = (
    False if cdf_dist == "uniform" else True
)  # only enable outlier prob. scaling for uniform dist.

# convert to OOD probability based on selected cdf distribution
id_outprob = convert_cdf(
    flatten_nll(bae_id_ref_pred["nll"]),
    flatten_nll(bae_id_pred["nll"]),
    dist=cdf_dist,
    scaling=scaling,
    min_level="mean",
)

ood_outprob = convert_cdf(
    flatten_nll(bae_id_ref_pred["nll"]),
    flatten_nll(bae_ood_pred["nll"]),
    dist=cdf_dist,
    scaling=scaling,
    min_level="mean",
)

# Mean score of outlier probability
id_outprob_mean = id_outprob.mean(0)
ood_outprob_mean = ood_outprob.mean(0)

# Convert to outlier uncertainty
# can index by type of uncertainty : epi, alea or total
id_unc = calc_outlier_unc(id_outprob)[unc_type]
ood_unc = calc_outlier_unc(ood_outprob)[unc_type]


# === PREPARE & CONCAT MODEL PREDICTIONS ===

# concatenate prob scores
all_y_true, all_outprob_mean, all_hard_pred = concat_ood_score(
    id_outprob_mean, ood_outprob_mean, p_threshold=0.5
)

# prepare various measures of uncertainties
all_prob_unc = np.concatenate((id_unc, ood_unc))
all_var_nll_unc = np.concatenate((var_nll_id, var_nll_ood))
all_exceed_unc = np.concatenate(
    (
        calc_exceed(calc_e_nll(bae_id_ref_pred), e_nll_id),
        calc_exceed(calc_e_nll(bae_id_ref_pred), e_nll_ood),
    )
)


# === EVALUATION OF CLASSIFIER USING E-NLL & VAR NLL & WAIC ? ===
e_nll_perf = calc_avgprc_perf(*concat_ood_score(e_nll_id, e_nll_ood))
var_nll_perf = calc_avgprc_perf(*concat_ood_score(var_nll_id, var_nll_ood))

# === EVALUATION : RETAINED PERF ===
retained_percs = [0.6, 0.7, 0.8, 0.9, 1.0]
# retained_percs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
retained_prob_unc_res = evaluate_retained_unc(
    all_outprob_mean=all_outprob_mean,
    all_hard_pred=all_hard_pred,
    all_y_true=all_y_true,
    all_unc=all_prob_unc,
    retained_percs=retained_percs,
)

retained_var_nll_res = evaluate_retained_unc(
    all_outprob_mean=all_outprob_mean,
    all_hard_pred=all_hard_pred,
    all_y_true=all_y_true,
    all_unc=all_var_nll_unc,
    retained_percs=retained_percs,
)

retained_exceed_res = evaluate_retained_unc(
    all_outprob_mean=all_outprob_mean,
    all_hard_pred=all_hard_pred,
    all_y_true=all_y_true,
    all_unc=all_exceed_unc,
    retained_percs=retained_percs,
)

retained_random_res = evaluate_random_retained_unc(
    all_outprob_mean=all_outprob_mean,
    all_hard_pred=all_hard_pred,
    all_y_true=all_y_true,
    repetition=50,
    retained_percs=retained_percs,
)
print(retained_prob_unc_res["auroc"])
print(retained_var_nll_res["auroc"])
print(retained_exceed_res["auroc"])
print(retained_random_res["auroc"])

# === EVALUATION : AUPRC-ERROR ===
if n_ensemble > 1:
    misclas_probunc_res = evaluate_misclas_detection(
        all_y_true, all_hard_pred, all_prob_unc, return_boxplot=True
    )
    misclas_varnll_res = evaluate_misclas_detection(
        all_y_true, all_hard_pred, all_var_nll_unc, return_boxplot=True
    )
    misclas_exceed_res = evaluate_misclas_detection(
        all_y_true, all_hard_pred, all_exceed_unc, return_boxplot=True
    )

print(e_nll_perf)
print(var_nll_perf)

# ===========PREDICTIONS=========
concat_x = np.concatenate((x_id_test, x_ood))

nll_key = "nll"

bae_all_pred = bae_model.predict(concat_x, select_keys=[nll_key])

# get ood scores
e_nll_all = calc_e_nll(bae_all_pred)
e_nll_id = e_nll_all[: len(x_id_test)]
e_nll_ood = e_nll_all[len(x_id_test) :]

print(calc_auroc(e_nll_id, e_nll_ood))
# print(calc_auroc(e_nll_ood, e_nll_id))

# ====== FEATURE WISE AUROC ====
e_nll_id_ft = bae_id_pred["nll"].mean(0)
e_nll_ood_ft = bae_ood_pred["nll"].mean(0)


auroc_feats = [
    calc_auroc(e_nll_id_ft[:, i], e_nll_ood_ft[:, i])
    for i in range(e_nll_ood_ft.shape[-1])
]
print("AUROC-FEATS")
print(np.sort(auroc_feats))
print(np.argsort(auroc_feats))
#
# # ================================================================
#
# # === ACTIVE LEARNING ===
# retained_perc = 0.7
# (retained_unc_indices, retained_id_indices, retained_ood_indices), (
#     referred_unc_indices,
#     referred_id_indices,
#     referred_ood_indices,
# ) = retained_top_unc_indices(
#     all_y_true, all_prob_unc, retained_perc=retained_perc, return_referred=True
# )
#
# concat_x = np.concatenate((x_id_test, x_ood))
#
# # referred indices are fed back to training data.
# # retained indices are used as the new testing data.
# x_id_train_new = concat_x[referred_unc_indices][referred_id_indices]
# x_ood_train_new = concat_x[referred_unc_indices][referred_ood_indices]
# x_id_test_new = concat_x[retained_unc_indices][retained_id_indices]
# x_ood_test_new = concat_x[retained_unc_indices][retained_ood_indices]
#
# x_id_train_loader_new = convert_dataloader(
#     np.concatenate((x_id_train, x_id_train_new)),
#     batch_size=len(x_id_train) // 4,
#     shuffle=True,
# )
#
# # ========sekali========
#
# nll_key = "nll"
# bae_id_pred = bae_model.predict(x_id_test_new, select_keys=[nll_key])
# bae_ood_pred = bae_model.predict(x_ood_test_new, select_keys=[nll_key])
#
# # get ood scores
# e_nll_id = calc_e_nll(bae_id_pred)
# e_nll_ood = calc_e_nll(bae_ood_pred)
# var_nll_id = calc_var_nll(bae_id_pred)
# var_nll_ood = calc_var_nll(bae_ood_pred)
#
# e_nll_perf_new_sekali = calc_avgprc_perf(*concat_ood_score(e_nll_id, e_nll_ood))
# var_nll_perf_new_sekali = calc_avgprc_perf(*concat_ood_score(var_nll_id, var_nll_ood))
#
# # =====active fit========
# y_scaler = 1
# bae_model_new = copy.deepcopy(bae_model)
#
#
# # bae_model_new.init_anchored_weight()
# # # bae_model_new.weight_decay = 0.0000001
# bae_model_new.scheduler_enabled = False
# bae_model_new.set_learning_rate(min_lr)
# bae_model_new.set_optimisers()
#
# # min_lr, max_lr, half_iter = run_auto_lr_range_v4(
# #     train_loader=x_id_train_loader_new,
# #     bae_model=bae_model_new,
# #     y=x_ood_train_new if len(x_ood_train_new) > 0 else None,
# #     window_size=1,
# #     num_epochs=10,
# #     run_full=False,
# # )
#
# if isinstance(bae_model_new, BAE_SGHMC):
#     bae_model_new.fit(
#         x=x_id_train_loader_new,
#         y=x_ood_train_new if len(x_ood_train_new) > 0 else None,
#         burn_epoch=num_epochs,
#         sghmc_epoch=num_epochs // 2,
#         clear_sghmc_params=True,
#     )
# else:
#     # time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs + 50)
#     time_method(
#         bae_model_new.fit,
#         x=x_id_train_loader_new,
#         y=x_ood_train_new if len(x_ood_train_new) > 0 else None,
#         num_epochs=50,
#     )
#
# nll_key = "nll"
# bae_id_pred = bae_model_new.predict(x_id_test_new, select_keys=[nll_key])
# bae_ood_pred = bae_model_new.predict(x_ood_test_new, select_keys=[nll_key])
#
# # get ood scores
# e_nll_id = calc_e_nll(bae_id_pred)
# e_nll_ood = calc_e_nll(bae_ood_pred)
# var_nll_id = calc_var_nll(bae_id_pred)
# var_nll_ood = calc_var_nll(bae_ood_pred)
#
# e_nll_perf_new = calc_avgprc_perf(*concat_ood_score(e_nll_id, e_nll_ood))
# var_nll_perf_new = calc_avgprc_perf(*concat_ood_score(var_nll_id, var_nll_ood))
#
#
# print(e_nll_perf_new_sekali)
# print(e_nll_perf_new)
#
# print(var_nll_perf_new_sekali)
# print(var_nll_perf_new)
#
# # === PREDICTIONS ===
# def predict_evaluate_bae(bae_model, x_id_train, x_valid, x_id_test, x_ood):
#     # start predicting
#     nll_key = "nll"
#
#     bae_id_pred = bae_model.predict(x_id_test, select_keys=[nll_key])
#     bae_ood_pred = bae_model.predict(x_ood, select_keys=[nll_key])
#
#     # convert to outlier probability
#     # 1. get reference distribution of NLL scores
#     if valid_size == 0:
#         bae_id_ref_pred = bae_model.predict(x_id_train, select_keys=[nll_key])
#     else:
#         bae_id_ref_pred = bae_model.predict(x_valid, select_keys=[nll_key])
#
#     # 2. define cdf distribution of OOD scores
#     cdf_dist = "ecdf"
#     unc_type = "total"
#     scaling = (
#         False if cdf_dist == "uniform" else True
#     )  # only enable outlier prob. scaling for uniform dist.
#
#     # convert to OOD probability based on selected cdf distribution
#     id_outprob = convert_cdf(
#         flatten_nll(bae_id_ref_pred["nll"]),
#         flatten_nll(bae_id_pred["nll"]),
#         dist=cdf_dist,
#         scaling=scaling,
#         min_level="mean",
#     )
#
#     ood_outprob = convert_cdf(
#         flatten_nll(bae_id_ref_pred["nll"]),
#         flatten_nll(bae_ood_pred["nll"]),
#         dist=cdf_dist,
#         scaling=scaling,
#         min_level="mean",
#     )
#
#     # Mean score of outlier probability
#     id_outprob_mean = id_outprob.mean(0)
#     ood_outprob_mean = ood_outprob.mean(0)
#
#     # Convert to outlier uncertainty
#     # can index by type of uncertainty : epi, alea or total
#     id_unc = calc_outlier_unc(id_outprob)[unc_type]
#     ood_unc = calc_outlier_unc(ood_outprob)[unc_type]
#
#     # === PREPARE & CONCAT MODEL PREDICTIONS ===
#
#     # concatenate prob scores
#     all_y_true, all_outprob_mean, all_hard_pred = concat_ood_score(
#         id_outprob_mean, ood_outprob_mean, p_threshold=0.5
#     )
#
#     # prepare various measures of uncertainties
#     all_prob_unc = np.concatenate((id_unc, ood_unc))
#
#     # === EVALUATION : RETAINED PERF ===
#     retained_unc_res = evaluate_retained_unc(
#         all_outprob_mean=all_outprob_mean,
#         all_hard_pred=all_hard_pred,
#         all_y_true=all_y_true,
#         all_unc=all_prob_unc,
#     )
#     return retained_unc_res
#
#     # return calc_avgprc_perf(all_y_true, all_outprob_mean)
#
#
# # res_old = predict_evaluate_bae(bae_model, x_id_train, x_valid, x_id_test, x_ood)
#
# (
#     filtered_unc_indices,
#     retained_id_indices,
#     retained_ood_indices,
# ) = retained_top_unc_indices(
#     all_y_true, all_prob_unc, retained_perc=retained_perc, return_referred=False
# )
#
# # retained_id_outprob_mean = all_outprob_mean[filtered_unc_indices][retained_id_indices]
# # retained_ood_outprob_mean = all_outprob_mean[filtered_unc_indices][retained_ood_indices]
# # retained_y_true = all_y_true[filtered_unc_indices][
# #     np.concatenate((retained_id_indices, retained_ood_indices))
# # ]
# # retained_hard_pred = all_hard_pred[filtered_unc_indices][
# #     np.concatenate((retained_id_indices, retained_ood_indices))
# # ]
# # auroc_retained, auroc_curve = calc_auroc(
# #     retained_id_outprob_mean,
# #     retained_ood_outprob_mean,
# #     return_threshold=True,
# # )
# #
# # calc_avgprc_perf(
# #     retained_y_true,
# #     np.concatenate((retained_id_outprob_mean, retained_ood_outprob_mean)),
# # )
#
# x_id_test_new = concat_x[filtered_unc_indices][retained_id_indices]
# x_ood_test_new = concat_x[filtered_unc_indices][retained_ood_indices]
#
# nll_key = "nll"
#
# bae_id_pred = bae_model.predict(x_id_test_new, select_keys=[nll_key])
# bae_ood_pred = bae_model.predict(x_ood_test_new, select_keys=[nll_key])
#
# bae_id_pred = bae_model.predict(x_id_test, select_keys=[nll_key])
# bae_ood_pred = bae_model.predict(x_ood, select_keys=[nll_key])
#
# # convert to outlier probability
# # 1. get reference distribution of NLL scores
# if valid_size == 0:
#     bae_id_ref_pred = bae_model.predict(x_id_train, select_keys=[nll_key])
# else:
#     bae_id_ref_pred = bae_model.predict(x_valid, select_keys=[nll_key])
#
# # 2. define cdf distribution of OOD scores
# cdf_dist = "ecdf"
# unc_type = "total"
# scaling = (
#     False if cdf_dist == "uniform" else True
# )  # only enable outlier prob. scaling for uniform dist.
#
# # convert to OOD probability based on selected cdf distribution
# id_outprob = convert_cdf(
#     flatten_nll(bae_id_ref_pred["nll"]),
#     flatten_nll(bae_id_pred["nll"]),
#     dist=cdf_dist,
#     scaling=scaling,
#     min_level="mean",
# )
#
# ood_outprob = convert_cdf(
#     flatten_nll(bae_id_ref_pred["nll"]),
#     flatten_nll(bae_ood_pred["nll"]),
#     dist=cdf_dist,
#     scaling=scaling,
#     min_level="mean",
# )
#
# # Mean score of outlier probability
# id_outprob_mean = id_outprob.mean(0)
# ood_outprob_mean = ood_outprob.mean(0)
#
# # Convert to outlier uncertainty
# # can index by type of uncertainty : epi, alea or total
# id_unc = calc_outlier_unc(id_outprob)[unc_type]
# ood_unc = calc_outlier_unc(ood_outprob)[unc_type]
#
# # === PREPARE & CONCAT MODEL PREDICTIONS ===
#
# # concatenate prob scores
# all_y_true, all_outprob_mean, all_hard_pred = concat_ood_score(
#     id_outprob_mean, ood_outprob_mean, p_threshold=0.5
# )
#
# # res_old = predict_evaluate_bae(
# #     bae_model, x_id_train, x_valid, x_id_test_new, x_ood_test_new
# # )
# #
# # res_new = predict_evaluate_bae(
# #     bae_model_new, x_id_train, x_valid, x_id_test_new, x_ood_test_new
# # )
#
# print(retained_prob_unc_res["auroc"])
#
# # print(res_old["auroc"])
# # print(res_new["auroc"])
#
# calc_auroc(id_outprob_mean, ood_outprob_mean)
#
# all_outprob_mean_id = all_outprob_mean[filtered_unc_indices][retained_id_indices]
# all_outprob_mean_ood = all_outprob_mean[filtered_unc_indices][retained_ood_indices]
#
# calc_auroc(all_outprob_mean_id, all_outprob_mean_ood)
#
#
# # ==================
# for autoencoder in bae_model.autoencoder:
#     autoencoder.eval()
#
# for autoencoder in bae_model.autoencoder:
#     autoencoder.train()
#
# x_id_test_new = x_id_test[:9]
# bae_id_ref_pred = bae_model.predict(x_valid, select_keys=[nll_key])
# bae_id_pred = bae_model.predict(x_id_test_new, select_keys=[nll_key])
# id_outprob = convert_cdf(
#     flatten_nll(bae_id_ref_pred["nll"]),
#     flatten_nll(bae_id_pred["nll"]),
#     dist=cdf_dist,
#     scaling=scaling,
#     min_level="mean",
# )
# print(id_outprob.mean(0))
# print(id_outprob.mean(0).shape)
#
# print(bae_id_pred["nll"][0])
