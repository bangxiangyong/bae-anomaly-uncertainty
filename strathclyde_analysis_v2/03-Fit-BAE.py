import itertools
import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from baetorch.baetorch.evaluation import calc_auroc
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.outlier_proba import BAE_Outlier_Proba
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from strathclyde_analysis_v2.evaluate_outlier_uncertainty import evaluate_ood_unc
from uncertainty_ood_v2.util.get_predictions import flatten_nll
from uncertainty_ood_v2.util.sensor_preproc import (
    MinMaxSensor,
    FFT_Sensor,
    Resample_Sensor,
)
from util.evaluate_ood import flag_tukey_fence
from util.exp_manager import ExperimentManager

bae_set_seed(123)

# data preproc. hyper params
resample_factor = 10
mode = "forging"
pickle_path = "pickles"

tukey_threshold = 1.5

target_dims_all = [1, 2, 7, 9, 12, 17]

heating_traces = pickle.load(open(pickle_path + "/" + "heating_inputs.p", "rb"))
forging_traces = pickle.load(open(pickle_path + "/" + "forging_inputs.p", "rb"))
column_names = pickle.load(open(pickle_path + "/" + "column_names.p", "rb"))
cmm_data = pickle.load(open(pickle_path + "/" + "strath_outputs_v2.p", "rb")).values
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


n_random_seeds = 1
random_seeds = np.random.randint(0, 1000, n_random_seeds)

# fmt: off
full_likelihood = ["mse","homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss", "bernoulli", "cbernoulli", "beta"]
homoscedestic_mode_map = { "bernoulli": "none", "cbernoulli": "none", "homo-gauss": "every","hetero-gauss": "none", "homo-tgauss": "none", "hetero-tgauss": "none", "mse": "none","beta":"none"}
likelihood_map = { "bernoulli": "bernoulli", "cbernoulli": "cbernoulli", "homo-gauss": "gaussian", "hetero-gauss": "gaussian", "homo-tgauss": "truncated_gaussian", "hetero-tgauss": "truncated_gaussian", "mse": "gaussian","beta":"beta"}
twin_output_map = { "bernoulli": False, "cbernoulli": False, "homo-gauss": False, "hetero-gauss": True, "homo-tgauss": False,"hetero-tgauss": True, "mse": False,"beta":True}
# fmt: on

min_max_clip = True
use_auto_lr = False

# Hyperparameter grids for running experiment.
# Uncomment and run for the required results in subsequent analysis.
# For each grid completion, please move the results into respective subfolder (`sensors`,`resampling`,`latent`,`likelihood`) inside a `results` folder.


# GRID : SENSOR SELECTION
grid = {
    "random_seed": random_seeds,
    "apply_fft": [False],
    "mode": ["forging"],
    "ss_id": [2, 25, 11, 13, 54, 71, 82],
    "target_dim": [2],
    "resample_factor": [resample_factor],
    "skip": [True, False],
    "latent_factor": [0.5],
    "bae_type": ["ae", "ens"],
    "full_likelihood": ["mse"],
}

# GRID : RESAMPLING
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 71, [13,71]],
#     "target_dim": [2],
#     "resample_factor": [1,2,5,10],
#     "skip": [True, False],
#     "latent_factor": [0.50],
#     "bae_type": ["ae", "ens", "vae"],
#     "full_likelihood": ["mse"],
# }

# GRID : BOTTLENECK
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 71, [13,71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [True, False],
#     "latent_factor": [0.25,0.50,1,2],
#     "bae_type": ["ae", "ens", "vae"],
#     "full_likelihood": ["mse"],
# }

# GRID : FULL AE ,LIKELIHOOD, SKIPS,
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 71, [13,71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [True, False],
#     "latent_factor": [0.5],
#     "bae_type": ["ae", "ens", "mcd", "sghmc", "vi", "vae"],
#     "full_likelihood": ["mse", "hetero-gauss", "bernoulli", "cbernoulli"],
# }

# GRID: SINGLE SAMPLE
grid = {
    "random_seed": random_seeds,
    "apply_fft": [False],
    "mode": ["forging"],
    "ss_id": [13],
    "target_dim": [2],
    "resample_factor": [resample_factor],
    "skip": [False],
    "latent_factor": [0.5],
    "bae_type": ["ens"],
    "full_likelihood": ["mse"],
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
    "ens": 5,
    "mcd": 100,
    "sghmc": 5,
    "vi": 100,
    "vae": 100,
    "ae": 1,
}

exp_name = "STRATH_FORGE_"

print(column_names[[13, 71]])
exp_man = ExperimentManager(folder_name="experiments")

# Loop over all grid search combinations
for values in tqdm(itertools.product(*grid.values())):

    # setup the grid
    exp_params = dict(zip(grid.keys(), values))
    print(exp_params)

    # unpack exp params
    random_seed = exp_params["random_seed"]
    apply_fft = exp_params["apply_fft"]
    mode = exp_params["mode"]
    sensor_i = exp_params["ss_id"]
    target_dim = exp_params["target_dim"]
    resample_factor = exp_params["resample_factor"]
    skip = exp_params["skip"]
    latent_factor = exp_params["latent_factor"]
    bae_type = exp_params["bae_type"]
    full_likelihood_i = exp_params["full_likelihood"]
    twin_output = twin_output_map[full_likelihood_i]
    homoscedestic_mode = homoscedestic_mode_map[full_likelihood_i]
    likelihood = likelihood_map[full_likelihood_i]
    n_bae_samples = n_bae_samples_map[bae_type]

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

    # data splitting
    x_id_train, x_id_test = train_test_split(
        x_id_train, train_size=0.70, shuffle=True, random_state=random_seed
    )

    # option to apply fft
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
    sensor_scaler = MinMaxSensor(
        num_sensors=x_id_train.shape[1], axis=1, clip=min_max_clip
    )
    x_id_train = sensor_scaler.fit_transform(x_id_train)
    x_id_test = sensor_scaler.transform(x_id_test)
    x_ood_test = sensor_scaler.transform(x_ood_test)

    # ===============FIT BAE===============

    use_cuda = True
    weight_decay = 0.00000000001
    anchored = True if bae_type == "ens" else False
    bias = False
    se_block = False
    norm = "layer"
    self_att = False
    self_att_transpose_only = False
    num_epochs = 100
    activation = "leakyrelu"
    lr = 0.001
    dropout = 0.005

    input_dim = x_id_train.shape[-1]
    latent_dim = int(np.product(x_id_train.shape[1:]) * latent_factor)

    chain_params = [
        {
            "base": "conv1d",
            "input_dim": input_dim,
            "conv_channels": [x_id_train.shape[1], 10, 20],
            "conv_stride": [2, 2],
            "conv_kernel": [5, 2],
            "activation": activation,
            "norm": norm,
            "se_block": se_block,
            "order": ["base", "norm", "activation"],
            "bias": bias,
            "last_norm": norm,
        },
        {
            "base": "linear",
            "architecture": [250, latent_dim],
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
        # dropout_rate=dropout,
    )

    x_id_train_loader = convert_dataloader(
        x_id_train,
        batch_size=len(x_id_train) // 5,
        shuffle=True,
    )

    if use_auto_lr:
        min_lr, max_lr, half_iter = run_auto_lr_range_v4(
            x_id_train_loader,
            bae_model,
            window_size=1,
            num_epochs=10,
            run_full=False,
        )

    if isinstance(bae_model, BAE_SGHMC):
        bae_model.fit(
            x_id_train_loader,
            burn_epoch=int(num_epochs * 2 / 3),
            sghmc_epoch=int(num_epochs / 3),
            clear_sghmc_params=True,
        )
    else:
        time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)

    # predict and evaluate
    (e_nll_id, e_nll_ood, var_nll_id, var_nll_ood), (
        eval_auroc,
        retained_res_all,
        misclas_res_all,
    ) = evaluate_ood_unc(
        bae_model=bae_model,
        x_id_train=x_id_train,
        x_id_test=x_id_test,
        x_ood_test=x_ood_test,
        exp_name=exp_name,
        exp_params=exp_params,
        eval_ood_unc=True,
        exp_man=exp_man,
        ret_flatten_nll=True,
        cdf_dists=["norm", "uniform", "ecdf", "expon"],
        norm_scalings=[True, False],
    )

    # predict and evaluate
    (e_nll_id, e_nll_ood, var_nll_id, var_nll_ood), (
        eval_auroc,
        retained_res_all,
        misclas_res_all,
    ) = evaluate_ood_unc(
        bae_model=bae_model,
        x_id_train=x_id_train,
        x_id_test=x_id_test,
        x_ood_test=x_ood_test,
        exp_name=exp_name,
        exp_params=exp_params,
        eval_ood_unc=True,
        exp_man=exp_man,
        ret_flatten_nll=True,
        cdf_dists=["norm", "uniform", "ecdf", "expon"],
        norm_scalings=[True],
    )

# bae_set_seed(15)
nll_key = "nll"
bae_id_ref_pred = bae_model.predict(x_id_train, select_keys=["nll"])
bae_id_pred = bae_model.predict(x_id_test, select_keys=["nll"])
bae_ood_pred = bae_model.predict(x_ood_test, select_keys=["nll"])
norm_scaling = True
# bae_proba_model = BAE_Outlier_Proba(
#     dist_type="norm",
#     norm_scaling=True,
#     fit_per_bae_sample=False if (isinstance(bae_model, VAE)) else True,
# )

bae_proba_model = BAE_Outlier_Proba(
    dist_type="norm",
    norm_scaling=True,
    fit_per_bae_sample=True,
)

bae_proba_model.fit(bae_id_ref_pred[nll_key])
id_proba_train_mean, id_proba_train_unc = bae_proba_model.predict(
    bae_id_ref_pred[nll_key], norm_scaling=norm_scaling
)
id_proba_mean, id_proba_unc = bae_proba_model.predict(
    bae_id_pred[nll_key], norm_scaling=norm_scaling
)
ood_proba_mean, ood_proba_unc = bae_proba_model.predict(
    bae_ood_pred[nll_key], norm_scaling=norm_scaling
)

# np.argwhere(e_nll_ood > np.percentile(pred_train, 85))

# retained_res_all["proba-total"]["auroc"]


# PLOT THE PROBA MAPPING
import matplotlib.pyplot as plt

# sample_x = np.arange(0,1,0.001)
# example = sample_x
# bae_proba_model.predict()

outp_all =[]
plt.figure()
for i in range(bae_model.num_samples):
    nll_inp = np.unique(flatten_nll(bae_id_ref_pred[nll_key])[i])
    outp_ = bae_proba_model.dist_[i].predict(nll_inp, norm_scaling=norm_scaling)
    outp_all.append(outp_)
    plt.plot(nll_inp,outp_, color="tab:blue",alpha=0.5)
# plt.plot(np.mean(np.array(outp_all),0),color="tab:orange")
outp_all = np.array(outp_all)


# outp_all =[]
# plt.figure()
# for i in range(bae_model.num_samples):
#     nll_inp = flatten_nll(bae_id_ref_pred[nll_key])[i]
#     outp_ = bae_proba_model.dist_[i].predict(nll_inp, norm_scaling=norm_scaling)
#     outp_all.append(outp_)
#     plt.plot(nll_inp,outp_, color="tab:blue",alpha=0.5)
# outp_all = np.array(outp_all)


outlier_probas = bae_proba_model.predict_proba_samples(bae_id_ref_pred[nll_key],norm_scaling=norm_scaling)

print(outlier_probas.max(0))

print(outlier_probas.min(0))

print(outlier_probas.mean(0))

print(retained_res_all["proba-total"]["auroc"])
