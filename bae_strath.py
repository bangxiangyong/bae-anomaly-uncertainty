# -*- coding: utf-8 -*-
"""BAE_STRATH.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pv0awUx984VPxAhKlRnbtot8aIMlEmd8
"""

# !pip install fastai
# !pip install piqa
# import os
# !git clone -b main_v3 https://bangxiangyong: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@github.com/bangxiangyong/understanding-bae.git
# !git checkout 4125c37afab0cbde1455b08e7f3e4e2576989636
# try:
#   import os
#   os.chdir("understanding-bae/")
# except Exception as e:
#   print("Already changed working dir.")
# !git clone -b overhaul_v2 https://bangxiangyong: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@github.com/bangxiangyong/baetorch.git
# !git checkout b368a43a06caa8bc5fe2493af915d410f19ece5e
# !pip install pandas==1.3.1

import torch
import os
torch.cuda.get_device_name(0)

# from google.colab import drive
# drive.mount('/content/drive')

import shutil

# try:
#   import os
#   os.chdir("understanding-bae")
# except Exception as e:
#   print("Already changed working dir.")

google_drive_path = "temp_drive/"
local_exp_path = "experiments/"
# exp_name = "STRATH_FORGE_TGAUSS_"
# exp_name = "STRATH_FORGE_100SGHMC_"
# exp_name = "STRATH_FORGE_BOTTLENECK_"
# exp_name = "STRATH_FORGE_TESTPICKLE_"
# exp_name = "STRATH_FORGE_150_"
exp_name = "STRATH_FORGE_UNCOODV_TEST"
# exp_name = "STRATH_FORGE_BOTTLENECKV3_TEST"
# exp_name = "STRATH_FORGE_LL_"

# filenames
auroc_filename = exp_name + "AUROC.csv"
avgprc_filename = exp_name + "AVGPRC.csv"
bce_se_filename = exp_name + "BCE_VS_SE.csv"
retained_perf_filename = exp_name + "retained_perf.csv"
misclas_perf_filename = exp_name + "misclas_perf.csv"

import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import itertools

from tqdm import tqdm

from baetorch.baetorch.evaluation import (
    calc_auroc,
    calc_avgprc,
    concat_ood_score,
    calc_avgprc_perf,
    evaluate_retained_unc,
    evaluate_retained_unc_v2,
    evaluate_random_retained_unc,
    evaluate_misclas_detection,
    convert_hard_pred,
    retained_top_unc_indices,
    summarise_retained_perf,
)
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.models_v2.outlier_proba import BAE_Outlier_Proba
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood.exceed import calc_exceed, calc_exceed_v2
from uncertainty_ood_v2.util.get_predictions import flatten_nll, calc_e_nll
from uncertainty_ood_v2.util.sensor_preproc import (
    MinMaxSensor,
    FFT_Sensor,
    Resample_Sensor,
)
from util.evaluate_ood import flag_tukey_fence, evaluate_bce_se
from util.exp_manager import ExperimentManager
from util.uncertainty import convert_cdf, calc_outlier_unc, get_optimal_threshold
import datetime

bae_set_seed(100)

# data preproc. hyper params
valid_size = 0.0
tolerance_scale = 1.0
resample_factor = 10
# resample_factor = 10
n_random_seeds = 10
random_seeds = np.random.randint(0, 1000, n_random_seeds)

# scaling = ["before"]
scaling = []
mode = "forging"
# mode = "heating"
# apply_fft = False
# apply_fft = True

tukey_threshold = 1.5

target_dims_all = [1, 2, 7, 9, 12, 17]

dataset_path = "datasets/strath"

heating_traces = pickle.load(open(dataset_path + "/" + "heating_inputs.p", "rb"))
forging_traces = pickle.load(open(dataset_path + "/" + "forging_inputs.p", "rb"))
column_names = pickle.load(open(dataset_path + "/" + "column_names.p", "rb"))
cmm_data = pickle.load(open(dataset_path + "/" + "strath_outputs_v2_err.p", "rb")).values
cmm_data = np.abs(cmm_data)

# fmt: off
heating_sensors_all = np.array([63, 64, 70, 71, 79, 76, 77, 78]) - 1
heating_sensors_pairs = np.array([[63, 70], [64, 71]]) - 1
forging_sensors_all = np.array(
    [3, 4, 6, 7, 10, 18, 26, 34, 42, 50, 66, 12, 28, 20, 36, 44, 52, 68, 14, 22, 30, 38, 46, 54, 15, 23, 31, 39, 47, 55,
     16, 24, 32, 40, 57, 65, 72, 80, 82, 83]) - 1
forging_sensor_pairs = np.array([[10, 34], [26, 50], [12, 36], [28, 52], [22, 46], [23, 47]]) - 1
# fmt: on

for sensor_pair in forging_sensor_pairs:
    for trace_i, forging_trace in enumerate(forging_traces):
        forging_traces[trace_i][:, sensor_pair[1]] = (
            forging_trace[:, sensor_pair[1]] - forging_trace[:, sensor_pair[0]]
        )

for sensor_pair in heating_sensors_pairs:
    for trace_i, heating_trace in enumerate(heating_traces):
        heating_traces[trace_i][:, sensor_pair[1]] = (
            heating_trace[:, sensor_pair[1]] - heating_trace[:, sensor_pair[0]]
        )

# fmt: off
full_likelihood = ["mse","homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss", "bernoulli", "cbernoulli"]
homoscedestic_mode_map = { "bernoulli": "none", "cbernoulli": "none", "homo-gauss": "every","hetero-gauss": "none", "homo-tgauss": "none", "hetero-tgauss": "none", "mse": "none",}
likelihood_map = { "bernoulli": "bernoulli", "cbernoulli": "cbernoulli", "homo-gauss": "gaussian", "hetero-gauss": "gaussian", "homo-tgauss": "truncated_gaussian", "hetero-tgauss": "truncated_gaussian", "mse": "gaussian",}
twin_output_map = { "bernoulli": False, "cbernoulli": False, "homo-gauss": False, "hetero-gauss": True, "homo-tgauss": False,"hetero-tgauss": True, "mse": False,}
# fmt: on

# Grid search

# FULL GRID : TOO EXPENSIVE
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 14, 71, 64, [13, 14, 71, 64]],
#     "target_dim": [2],
#     "resample_factor": [5],
#     "skip": [True, False],
#     "latent_factor": [0.25, 0.5, 1, 2],
#     "bae_type": ["ae", "ens", "mcd", "sghmc", "vi", "vae"],
#     "full_likelihood": full_likelihood,
# }

# EVALUATE EFFECT OF LIKELIHOOD AND VARNLL
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 71, [13, 71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [False],
#     "latent_factor": [0.5],
#     "bae_type": ["ae","ens","vae", "sghmc","vi","mcd"],
#     "full_likelihood": ["mse","hetero-gauss","bernoulli","cbernoulli"],
# }

# EVALUATE SGHMC
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 71, [13, 71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [True,False],
#     "latent_factor": [0.5],
#     "bae_type": ["sghmc"],
#     "full_likelihood": ["mse","hetero-gauss","bernoulli","cbernoulli","homo-tgauss","hetero-tgauss"],
# }

# EVALUATE UNC OOD
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13,71,[13, 71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [True,False],
#     "latent_factor": [0.5],
#     "bae_type": ["ae","ens","vae", "sghmc","vi","mcd"],
#     "full_likelihood": ["mse"],
# }

# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [True],
#     "latent_factor": [0.5],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
# }

# EVALUATE EFFECT OF TGAUSS LIKELIHOOD
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 71, [13, 71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [True,False],
#     "latent_factor": [0.5],
#     "bae_type": ["ae","ens","vae", "sghmc","vi","mcd"],
#     "full_likelihood": ["homo-tgauss","hetero-tgauss"],
# }

# EVALUATE NEED FOR BOTTLE NECK
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 71, [13, 71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [True,False],
#     "latent_factor": [0.25,0.5,1.0,2.0],
#     "bae_type": ["ae","ens","vae"],
#     "full_likelihood": ["mse"],
# }

# EVALUATE RESAMPLING FACTOR
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13,71,[13, 71]],
#     "target_dim": [2],
#     "resample_factor": [1,2,5,10],
#     "skip": [True,False],
#     "latent_factor": [0.5],
#     "bae_type": ["ae","ens","vae"],
#     "full_likelihood": ["mse"],
# }

# EVALUATE SENSOR IMPORTANCE
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [2, 25, 11, 13, 54, 71, 82, [2, 25, 11, 13, 54, 71, 82]],
#     "target_dim": [2],
#     "resample_factor": [10],
#     "skip": [True,False],
#     "latent_factor": [0.5],
#     "bae_type": ["ens"],
#     "full_likelihood": ["mse"],

# }

# EVALUATE NORM LAYERS
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13,71,[13, 71]],
#     "target_dim": [2],
#     "resample_factor": [10],
#     "skip": [True,False],
#     "latent_factor": [0.5],
#     "bae_type": ["ae","ens"],
#     "full_likelihood": ["mse"],
#     "norm": ["layer","weight","none"]
# }

# # ====NEW GRIDS====
# # GRID - RESAMPLING
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13,71,[13, 71]],
#     "target_dim": [2],
#     "resample_factor": [1,2,5,10],
#     "skip": [True,False],
#     "latent_factor": [0.5],
#     "bae_type": ["ae","ens","vae"],
#     "full_likelihood": ["mse"],
#     "eval_ood_unc": [False]
# }

# # GRID - BOTTLENECK
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 25, 71, [13, 25, 71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [True,False],
#     "latent_factor": [0.1,0.5,1.0,10],
#     "bae_type": ["ae","ens","vae", "vi","mcd"],
#     "full_likelihood": ["mse"],
#     "eval_ood_unc": [False]
# }

# # GRID - LL
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 25, 71, [13, 25, 71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [-1],
#     "latent_factor": [0.5],
#     "bae_type": ["ae","ens","vae", "vi","mcd"],
#     "full_likelihood": ["mse","hetero-gauss","bernoulli","cbernoulli"],
#     "eval_ood_unc": [False]
# }

# GRID - UNC-OOD
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 71, [13, 71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [-1],
#     "latent_factor": [0.5],
#     "bae_type": ["ae", "ens", "mcd", "sghmc", "vi", "vae"],
#     "full_likelihood": ["mse"],
#     "eval_ood_unc": [True]
# }

# # UNC OOD NEWEST
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 25, 71, [13, 25, 71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [-1],
#     "latent_factor": [0.5],
#     "bae_type": ["ae", "ens", "mcd", "sghmc", "vi", "vae"],
#     "full_likelihood": ["mse"],
#     "eval_ood_unc": [True]
# }

# UNC OOD NEWEST (20220616)
grid = {
    "random_seed": random_seeds,
    "apply_fft": [False],
    "mode": ["forging"],
    "ss_id": [13, 25, 71, [13, 25, 71]],
    "target_dim": [2],
    "resample_factor": [resample_factor],
    "skip": [-1],
    "latent_factor": [0.5],
    "bae_type": ["ae", "ens", "mcd", "vi", "vae"],
    "full_likelihood": ["mse"],
    "eval_ood_unc": [True]
}

bae_type_classes = {
    "ens": BAE_Ensemble,
    "mcd": BAE_MCDropout,
    "sghmc": BAE_SGHMC,
    "vi": BAE_VI,
    "vae": VAE,
    "ae": BAE_Ensemble,
}

n_bae_samples_map = {
    "ens": 10,
    "mcd": 100,
    "sghmc": 50,
    "vi": 100,
    "vae": 100,
    "ae": 1,
}



# Loop over all grid search combinations
for rep, values in enumerate(tqdm(itertools.product(*grid.values()))):

    # setup the grid
    exp_params = dict(zip(grid.keys(), values))
    print(exp_params)

    # unpack exp params
    random_seed = exp_params["random_seed"]
    apply_fft = exp_params["apply_fft"]
    mode = exp_params["mode"]
    sensor_i = exp_params["ss_id"]

    # SPECIAL CASE: SS_ID 13-WITH NO-SKIP
    if isinstance(exp_params["skip"], int) and exp_params["skip"] == -1:
      if (sensor_i == 13) or (sensor_i == 25):
        exp_params.update({"skip":False})
      else:
        exp_params.update({"skip":True})

    target_dim = exp_params["target_dim"]
    resample_factor = exp_params["resample_factor"]
    skip = exp_params["skip"]
    latent_factor = exp_params["latent_factor"]
    bae_type = exp_params["bae_type"]
    full_likelihood_i = exp_params["full_likelihood"]
    eval_ood_unc = exp_params["eval_ood_unc"]



    # whether to evaluate OOD uncertainty
    if eval_ood_unc:
        pickle_files = [
            auroc_filename,
            avgprc_filename,
            bce_se_filename,
            retained_perf_filename,
            misclas_perf_filename,
        ]
    else:
        pickle_files = [auroc_filename, avgprc_filename, bce_se_filename]

    if "norm" in exp_params.keys():
      norm = exp_params["norm"]
    else:
      norm = "layer"

    twin_output = twin_output_map[full_likelihood_i]
    homoscedestic_mode = homoscedestic_mode_map[full_likelihood_i]
    likelihood = likelihood_map[full_likelihood_i]
    n_bae_samples = n_bae_samples_map[bae_type]

    # refresh exp manager random seed
    exp_man = ExperimentManager(folder_name=local_exp_path, random_seed=random_seed+np.random.randint(10000))

    # continue unpacking
    bae_set_seed(random_seed)

    target_dim = [target_dim] if isinstance(target_dim, int) else target_dim
    heating_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)
    forging_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)

    # get heating/forging traces
    x_heating = [heating_trace[:, heating_sensors] for heating_trace in heating_traces]
    x_forging = [forging_trace[:, forging_sensors] for forging_trace in forging_traces]

    # cut to smallest trace length
    min_heat_length = np.array([len(trace) for trace in x_heating]).min()
    min_forge_length = np.array([len(trace) for trace in x_forging]).min()

    x_heating = np.array(
        [heating_trace[:min_heat_length] for heating_trace in x_heating]
    )
    x_forging = np.array(
        [forging_trace[:min_forge_length] for forging_trace in x_forging]
    )

    # ood and id
    tukey_flags = np.apply_along_axis(
        flag_tukey_fence, arr=cmm_data, axis=0, level=tukey_threshold
    )
    tukey_flags_filtered = (
        tukey_flags[:, target_dim] if len(target_dim) > 0 else tukey_flags
    )
    ood_args = np.unique(np.argwhere(tukey_flags_filtered.sum(1) >= 1)[:, 0])
    id_args = np.unique(np.argwhere(tukey_flags_filtered.sum(1) == 0)[:, 0])

    x_heating_train = x_heating[id_args]
    x_forging_train = x_forging[id_args]

    x_heating_ood = x_heating[ood_args]
    x_forging_ood = x_forging[ood_args]

    # ====================================
    x_heating_train = np.moveaxis(x_heating_train, 1, 2)
    x_forging_train = np.moveaxis(x_forging_train, 1, 2)

    x_heating_ood = np.moveaxis(x_heating_ood, 1, 2)
    x_forging_ood = np.moveaxis(x_forging_ood, 1, 2)

    if mode == "forging":
        x_id_train = x_forging_train
        x_ood_test = x_forging_ood

    elif mode == "heating":
        x_id_train = x_heating_train
        x_ood_test = x_heating_ood

    # x_id_train, x_id_test = train_test_split(x_id_train, train_size=0.75, shuffle=True)
    x_id_train, x_id_test = train_test_split(
        x_id_train, train_size=0.70, shuffle=True, random_state=random_seed
    )

    if apply_fft:
        x_id_train = FFT_Sensor().transform(x_id_train)
        x_id_test = FFT_Sensor().transform(x_id_test)
        x_ood_test = FFT_Sensor().transform(x_ood_test)

    # resample
    if resample_factor > 1:
        x_id_train = Resample_Sensor().transform(x_id_train, n=resample_factor)
        x_id_test = Resample_Sensor().transform(x_id_test, n=resample_factor)
        x_ood_test = Resample_Sensor().transform(x_ood_test, n=resample_factor)

    # min max
    sensor_scaler = MinMaxSensor(num_sensors=x_id_train.shape[1], axis=1)
    x_id_train = sensor_scaler.fit_transform(x_id_train)
    x_id_test = sensor_scaler.transform(x_id_test)
    x_ood_test = sensor_scaler.transform(x_ood_test)

    # ===============FIT BAE===============

    use_cuda = True
    weight_decay = 0.0000000001
    anchored = True if bae_type == "ens" else False
    bias = False
    se_block = False
    self_att = False
    self_att_transpose_only = False
    num_epochs = 100
    activation = "leakyrelu"
    lr = 0.001
    bae_keys = {"dropout_rate": 0.01} if bae_type == "mcd" else {}
    # if (bae_type == "mcd") or (bae_type == "vae") or (bae_type == "vi"):
    #   bae_keys.update({"num_train_samples":5})
    input_dim = x_id_train.shape[-1]
    latent_dim = int(np.product(x_id_train.shape[1:]) * latent_factor)
    chain_params = [
        #         {
        #     "base": "conv1d",
        #     "input_dim": input_dim,
        #     "conv_channels": [x_id_train.shape[1], 10, 20],
        #     "conv_stride": [2, 2],
        #     "conv_kernel": [5, 2],
        #     "activation": activation,
        #     "norm": norm,
        #     "se_block": se_block,
        #     "order": ["base", "norm", "activation"],
        #     "bias": bias,
        #     "last_norm": norm,
        # },
        {
            "base": "conv1d",
            "input_dim": input_dim,
            "conv_channels": [x_id_train.shape[1], 10, 20],
            "conv_stride": [2, 2],
            "conv_kernel": [8, 2],
            "activation": activation,
            "norm": norm,
            "se_block": se_block,
            "order": ["base", "norm", "activation"],
            "bias": bias,
            "last_norm": norm,
        },

        {
            "base": "linear",
            "architecture": [1000, latent_dim],
            "activation": activation,
            "norm": norm,
            "last_norm": norm,
        },
    ]

    bae_model = bae_type_classes[bae_type](
        chain_params=chain_params,
        last_activation="sigmoid",
        last_norm=norm,
        twin_output=twin_output,
        twin_params={"activation": "selu", "norm": "none"},
        skip=skip,
        use_cuda=use_cuda,
        scaler_enabled=False,
        homoscedestic_mode=homoscedestic_mode,
        likelihood=likelihood,
        weight_decay=weight_decay,
        num_samples=n_bae_samples,
        anchored=anchored,
        learning_rate=lr,
        stochastic_seed=random_seed,
        **bae_keys
    )

    x_id_train_loader = convert_dataloader(
        x_id_train,
        batch_size=len(x_id_train) // 5,
        shuffle=True,
    )

    if isinstance(bae_model, BAE_SGHMC):
        bae_model.fit(
            x_id_train_loader,
            burn_epoch=int(num_epochs*2/3),
            sghmc_epoch=num_epochs // 3,
            clear_sghmc_params=True,
        )
    else:
        time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)

    # === PREDICTIONS ===

    # start predicting
    nll_key = "nll"

    bae_id_pred = bae_model.predict(x_id_test, select_keys=[nll_key,"y_mu"])
    bae_ood_pred = bae_model.predict(x_ood_test, select_keys=[nll_key,"y_mu"])


    # get ood scores
    e_nll_id = flatten_nll(bae_id_pred[nll_key]).mean(0)
    e_nll_ood = flatten_nll(bae_ood_pred[nll_key]).mean(0)
    var_nll_id = flatten_nll(bae_id_pred[nll_key]).var(0)
    var_nll_ood = flatten_nll(bae_ood_pred[nll_key]).var(0)

    # var_x_id = flatten_nll(bae_id_pred["y_mu"].var(0),max_dim=1)
    # var_x_ood = flatten_nll(bae_ood_pred["y_mu"].var(0),max_dim=1)

    eval_auroc = {
        "E_AUROC": calc_auroc(e_nll_id, e_nll_ood),
        "V_AUROC": calc_auroc(var_nll_id, var_nll_ood),
        "WAIC_AUROC": calc_auroc(var_nll_id+e_nll_id, var_nll_ood+e_nll_ood),
        # "VX_AUROC":calc_auroc(var_x_id, var_x_ood),
    }

    eval_avgprc = {
        "E_AUROC": calc_avgprc(e_nll_id, e_nll_ood),
        "V_AUROC": calc_avgprc(var_nll_id, var_nll_ood),
        "WAIC_AUROC": calc_avgprc(var_nll_id+e_nll_id, var_nll_ood+e_nll_ood),
        # "VX_AUROC":calc_avgprc(var_x_id, var_x_ood),
    }

    # res = exp_man.concat_params_res(exp_params, eval_res)
    # exp_man.update_csv(exp_params=res, csv_name=auroc_filename)
    exp_man.update_csv(exp_params=exp_man.concat_params_res(exp_params, eval_auroc), csv_name=auroc_filename)
    exp_man.update_csv(exp_params=exp_man.concat_params_res(exp_params, eval_avgprc), csv_name=avgprc_filename)

    # special case for evaluating bce vs mse
    if (
        likelihood == "gaussian" and not twin_output and homoscedestic_mode == "none"
    ) or likelihood == "bernoulli":
        eval_res = evaluate_bce_se(bae_model, x_id_test, x_ood_test)


        res = exp_man.concat_params_res(exp_params, eval_res)
        exp_man.update_csv(exp_params=res, csv_name=bce_se_filename)

    # === EVALUATE OUTLIER UNCERTAINTY ===
    if eval_ood_unc:
      # convert to outlier probability
      # 1. get reference distribution of NLL scores
      if valid_size == 0:
          bae_id_ref_pred = bae_model.predict(x_id_train, select_keys=[nll_key])

      all_y_true = np.concatenate((np.zeros_like(e_nll_id), np.ones_like(e_nll_ood)))
      all_var_nll_unc = np.concatenate((var_nll_id, var_nll_ood))
      concat_e_nll = concat_ood_score(e_nll_id, e_nll_ood)[1]

      # 2. define cdf distribution of OOD scores
      for cdf_dist in ["norm", "uniform", "ecdf","expon"]:
          bae_proba_model = BAE_Outlier_Proba(
              dist_type=cdf_dist,
              norm_scaling=True,
              fit_per_bae_sample=False if (bae_type == "vae") else True,
          )
          bae_proba_model.fit(bae_id_ref_pred[nll_key])
          for norm_scaling in [True, False]:
              id_proba_mean, id_proba_unc = bae_proba_model.predict(
                  bae_id_pred[nll_key], norm_scaling=norm_scaling
              )
              ood_proba_mean, ood_proba_unc = bae_proba_model.predict(
                  bae_ood_pred[nll_key], norm_scaling=norm_scaling
              )

              # CONVERT HARD PRED
              all_proba_mean = np.concatenate((id_proba_mean, ood_proba_mean))

              auroc, (fpr, tpr, thresholds) = calc_auroc(id_proba_mean,ood_proba_mean,return_threshold=True)
              optim_threshold = thresholds[np.argmax(tpr - fpr)]

              # all_hard_proba_pred = convert_hard_pred(all_proba_mean, p_threshold=0.5)
              all_hard_proba_pred = convert_hard_pred(all_proba_mean, p_threshold=optim_threshold)


              # EXCEED UNCERTAINTY
              all_exceed_unc = calc_exceed(
                  len(calc_e_nll(bae_id_ref_pred)),
                  all_proba_mean,
                  all_hard_proba_pred,
                  contamination=0.0,
              )

              # Evalute uncertainty performances
              # retained_percs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
              retained_percs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
              # retained_percs = [0.6, 0.7, 0.8, 0.9, 1.0]
              # retained_percs = [0.7, 0.8, 0.9, 1.0]

              # Evaluate retained performance
              retained_varnll_res = evaluate_retained_unc_v2(
                  all_outprob_mean=concat_e_nll,
                  all_hard_pred=all_hard_proba_pred,
                  all_y_true=all_y_true,
                  all_unc=all_var_nll_unc,
              )

              retained_exceed_res = evaluate_retained_unc_v2(
                  all_outprob_mean=concat_e_nll,
                  all_hard_pred=all_hard_proba_pred,
                  all_y_true=all_y_true,
                  all_unc=all_exceed_unc,
              )

              # retained_random_res = evaluate_random_retained_unc(
              #     all_outprob_mean=concat_e_nll,
              #     all_hard_pred=all_hard_proba_pred,
              #     all_y_true=all_y_true,
              #     repetition=10,
              #     retained_percs=retained_percs,
              # )

              # evaluate misclassification detection
              misclas_varnll_res = evaluate_misclas_detection(
                  all_y_true,
                  all_hard_proba_pred,
                  all_var_nll_unc,
                  return_boxplot=True,
              )
              misclas_exceed_res = evaluate_misclas_detection(
                  all_y_true,
                  all_hard_proba_pred,
                  all_exceed_unc,
                  return_boxplot=True,
              )

              # Save all results in dicts
              retained_res_all = {}
              misclas_res_all = {}
              retained_res_all.update(
                  {
                      "varnll": retained_varnll_res,
                      "exceed": retained_exceed_res,
                      # "random": retained_random_res,
                  }
              )
              misclas_res_all.update(
                  {
                      "varnll": misclas_varnll_res,
                      "exceed": misclas_exceed_res,
                  }
              )

              for proba_unc_key in ["epi", "alea", "total"]:
                  all_proba_unc = np.concatenate(
                      (id_proba_unc[proba_unc_key], ood_proba_unc[proba_unc_key])
                  )

                  retained_prob_unc_res = evaluate_retained_unc_v2(
                      all_outprob_mean=concat_e_nll,
                      # all_outprob_mean=all_proba_mean,
                      all_hard_pred=all_hard_proba_pred,
                      all_y_true=all_y_true,
                      all_unc=all_proba_unc,
                      round_deci=200,
                  )

                  misclas_prob_unc_res = evaluate_misclas_detection(
                      all_y_true,
                      all_hard_proba_pred,
                      all_proba_unc,
                      return_boxplot=True,
                  )

                  retained_res_all.update(
                      {"proba-" + proba_unc_key: retained_prob_unc_res}
                  )
                  misclas_res_all.update({"proba-" + proba_unc_key: misclas_prob_unc_res})

              # Save uncertainty evaluation results in CSV
              unc_method = {"dist": cdf_dist, "norm": norm_scaling}
              base_method_columns = exp_man.concat_params_res(exp_params, unc_method)
              pickle_retained = exp_man.encode(exp_man.concat_params_res(exp_params, unc_method,{"restype":"retained","date":datetime.datetime.now()}))
              pickle_misclas = exp_man.encode(exp_man.concat_params_res(exp_params, unc_method,{"restype":"misclas","date":datetime.datetime.now()}))
              print(pickle_retained)

              # handle retained results
              for unc_method_name in retained_res_all.keys():
                  summary_ret_res = summarise_retained_perf(
                      retained_res_all[unc_method_name], flatten_key=True
                  )
                  retained_csv = exp_man.concat_params_res(
                      base_method_columns,
                      {"unc_method": unc_method_name},
                      summary_ret_res,
                  )
                  exp_man.update_csv(
                      retained_csv,
                      insert_pickle=pickle_retained,
                      csv_name=exp_name + "retained_perf.csv",
                  )
              exp_man.encode_pickle(pickle_retained, data=retained_res_all)

              # handle misclas results
              for unc_method_name in misclas_res_all.keys():
                  misclas_csv = exp_man.concat_params_res(
                      base_method_columns,
                      {"unc_method": unc_method_name},
                      misclas_res_all[unc_method_name]["all_err"],
                  )
                  exp_man.update_csv(
                      misclas_csv,
                      insert_pickle=pickle_misclas,
                      csv_name=exp_name + "misclas_perf.csv",
                  )
              exp_man.encode_pickle(pickle_misclas, data=misclas_res_all)


    # save checkpoint every 10 hyper param reps
    if rep % 10 == 0:
      for filename in pickle_files:
          shutil.copy(os.path.join(local_exp_path, filename), os.path.join(google_drive_path, filename))

# save final checkpoint
print("DONE ALL EXPERIMENT RUNS")
for filename in pickle_files:
    shutil.copy(os.path.join(local_exp_path, filename), os.path.join(google_drive_path, filename))

local_pickle_path = os.path.join(local_exp_path, "pickles")
for pickle_file in os.listdir(local_pickle_path):
    shutil.copy(os.path.join(local_pickle_path,pickle_file), os.path.join(google_drive_path, "pickles", pickle_file))

os.path.join(google_drive_path, auroc_filename)

os.path.join(local_exp_path, auroc_filename)

# saves files to gg drive
for filename in [auroc_filename, bce_se_filename, retained_perf_filename, misclas_perf_filename]:
    shutil.copy(os.path.join(local_exp_path, filename), os.path.join(google_drive_path, filename))

local_pickle_path = os.path.join(local_exp_path, "pickles")
for pickle_file in os.listdir(local_pickle_path):
    shutil.copy(os.path.join(local_pickle_path,pickle_file), os.path.join(google_drive_path, "pickles", pickle_file))

# check if all pickles exist
pickle_files = os.listdir(os.path.join(local_exp_path, "pickles"))

res_csv = pd.read_csv(os.path.join(local_exp_path, retained_perf_filename))["pickle"].unique()

match = [0 if file in pickle_files else 1 for file in res_csv]
print(np.sum(match))

res_csv["pickle"].unique()

# delete pickles from local folder
# for file in os.listdir("../experiments/pickles"):
#   os.remove("../experiments/pickles/"+file)

import datetime
str(datetime.datetime.now())