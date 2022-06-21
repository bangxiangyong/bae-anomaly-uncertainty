import itertools
import pickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from baetorch.baetorch.evaluation import (
    calc_auroc,
    concat_ood_score,
    evaluate_random_retained_unc,
    evaluate_misclas_detection,
    convert_hard_pred,
    summarise_retained_perf,
    evaluate_retained_unc_v2,
)
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
from uncertainty_ood.exceed import calc_exceed
from uncertainty_ood_v2.util.get_predictions import flatten_nll, calc_e_nll
from uncertainty_ood_v2.util.sensor_preproc import (
    MinMaxSensor,
    FFT_Sensor,
    Resample_Sensor,
)
from uncertainty_ood_v2.util.sensor_preproc import (
    Resample_Sensor_Fourier,
)
from util.evaluate_ood import evaluate_bce_se
from util.exp_manager import ExperimentManager
import os

bae_set_seed(100)

# args for script
total_sensors = 17
path_data = "pickles/raw_data.p"
pickle_path = "pickles/"
data_raw = pickle.load(open(pickle_path + "raw_data.p", "rb"))
num_cycles = 2205
sensor_xis = 1
seq_xis = 2

# exp name and filenames
exp_name = "ZEMA_HYD_SS10"
auroc_filename = exp_name + "AUROC.csv"
bce_se_filename = exp_name + "BCE_VS_SE.csv"
retained_perf_filename = exp_name + "retained_perf.csv"
misclas_perf_filename = exp_name + "misclas_perf.csv"
sensor_auroc_filename = exp_name + "sensors_auroc.csv"
ood_level_auroc_filename = exp_name + "level_auroc.csv"

apply_fft = False

# Loop over all grid search combinations
# fmt: off
n_random_seeds = 1
random_seeds = np.random.randint(0, 1000, n_random_seeds)
full_likelihood = ["mse", "homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss", "bernoulli", "cbernoulli",
                   "beta"]
homoscedestic_mode_map = {"bernoulli": "none", "cbernoulli": "none", "homo-gauss": "every", "hetero-gauss": "none",
                          "homo-tgauss": "none", "hetero-tgauss": "none", "mse": "none", "beta": "none"}
likelihood_map = {"bernoulli": "bernoulli", "cbernoulli": "cbernoulli", "homo-gauss": "gaussian",
                  "hetero-gauss": "gaussian", "homo-tgauss": "truncated_gaussian",
                  "hetero-tgauss": "truncated_gaussian", "mse": "gaussian", "beta": "beta"}
twin_output_map = {"bernoulli": False, "cbernoulli": False, "homo-gauss": False, "hetero-gauss": True,
                   "homo-tgauss": False, "hetero-tgauss": True, "mse": False, "beta": True}
# fmt: on

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
    "sghmc": 100,
    "vi": 100,
    "vae": 100,
    "ae": 1,
}

# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [apply_fft],
#     # "ss_id": [
#     #     *np.arange(total_sensors).astype(int),
#     #     list(np.arange(total_sensors).astype(int)),
#     # ],
#     "ss_id": [0],
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": [60],
#     "skip": [False, True],
#     "latent_factor": [0.5],
#     "bae_type": ["ae", "ens"],
#     "full_likelihood": ["mse"],
# }

# grid = {
#     "random_seed": [1],
#     "apply_fft": [apply_fft],
#     # "ss_id": [
#     #     *np.arange(total_sensors).astype(int),
#     #     list(np.arange(total_sensors).astype(int)),
#     # ],
#     "ss_id": list(range(17)) + [list(range(17))],
#     "target_dim": [0],
#     "resample_factor": [60],
#     "skip": [False],
#     "latent_factor": [0.5],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
# }

# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [apply_fft],
#     "ss_id": [-1],
#     "target_dim": [0],
#     "resample_factor": [562],
#     "skip": [True],
#     "latent_factor": [0.5],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
# }

# GRID SELECT SENSORS
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [apply_fft],
#     "ss_id": [-1],
#     "target_dim": [0],
#     "resample_factor": [60],
#     "skip": [True],
#     "latent_factor": [0.5],
#     "bae_type": ["sghmc"],
#     "full_likelihood": ["mse"],
# }

grid = {
    "random_seed": random_seeds,
    "apply_fft": [apply_fft],
    "ss_id": [-1],
    "target_dim": [1],
    "resample_factor": [60],
    "skip": [True],
    "latent_factor": [0.5],
    "bae_type": ["sghmc"],
    "full_likelihood": ["mse"],
    "eval_ood_unc": [True],
}


target_dim_ssid = [12, 12, 5, 12]

for values in tqdm(itertools.product(*grid.values())):
    # setup the grid
    exp_params = dict(zip(grid.keys(), values))
    print(exp_params)

    # unpack exp params
    random_seed = exp_params["random_seed"]
    apply_fft = exp_params["apply_fft"]
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

    # refresh exp manager random seed
    exp_man = ExperimentManager(folder_name="experiments", random_seed=random_seed+np.random.randint(10000))

    # continue unpacking
    bae_set_seed(random_seed)

    # whether to evaluate OOD uncertainty
    eval_ood_unc = exp_params["eval_ood_unc"]
    if eval_ood_unc:
        pickle_files = [
            auroc_filename,
            bce_se_filename,
            retained_perf_filename,
            misclas_perf_filename,
            sensor_auroc_filename,
        ]
    else:
        pickle_files = [auroc_filename, bce_se_filename, sensor_auroc_filename]

    # conditional ss_id (-1) on target_dim
    if sensor_i == -1:
        sensor_i = target_dim_ssid[target_dim]
        exp_params.update({"ss_id": sensor_i})

    # Apply Resample here
    # use encode/load pickle of the resampled x data
    if not os.path.exists(os.path.join("x_resampled", "pickles")):
        os.mkdir(os.path.join("x_resampled", "pickles"))

    resample_maps = {
        "Hz_1": {"n": 1, "mode": "up"},
        "Hz_10": {"n": 10, "mode": "down"},
        "Hz_100": {"n": 100, "mode": "down"},
    }

    exp_man_xsample = ExperimentManager(folder_name="x_resampled")
    pickle_x_rs = exp_man_xsample.encode({"resample": resample_factor})
    if pickle_x_rs not in os.listdir(os.path.join("x_resampled","pickles")):
        x_resampled = None
        for id_, key in enumerate(["Hz_1", "Hz_10", "Hz_100"]):
            if apply_fft:
                x_temp = FFT_Sensor().transform(data_raw[key])
            else:
                x_temp = np.copy(data_raw[key])
            x_resampled_ = Resample_Sensor_Fourier().transform(
                x_temp, seq_len=resample_factor, seq_axis=seq_xis
            )
            if id_ == 0:
                x_resampled = x_resampled_
            else:
                x_resampled = np.concatenate(
                    (x_resampled, x_resampled_), axis=sensor_xis
                )
        exp_man_xsample.encode_pickle(pickle_x_rs, data=x_resampled)
    else:
        x_resampled = exp_man_xsample.load_encoded_pickle(pickle_x_rs)

    # select sensors
    if isinstance(sensor_i, int):
        x_resampled_select = x_resampled[:, [sensor_i]]
    else:
        x_resampled_select = x_resampled[:, sensor_i]
    y_target = np.copy(data_raw["target"])

    # split inliers and outliers
    # get the y_arg_ood where only target dim is faulty,
    # and all other dims are healthy

    y_arg_ood = np.argwhere(
        (y_target[:, target_dim] > 0)
        & (y_target[:, [i for i in range(4) if i != target_dim]].sum(1) == 0)
    )[:, 0]
    x_inliers = x_resampled_select[np.argwhere(y_target[:, target_dim] == 0)[:, 0]]
    x_outliers = x_resampled_select[y_arg_ood]

    x_id_train, x_id_test = train_test_split(
        x_inliers, random_state=random_seed, shuffle=True, train_size=0.70
    )
    x_ood_test = x_outliers

    # === MIN MAX SCALER ===
    min_max_clip = True
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
    norm = "none"
    self_att = False
    self_att_transpose_only = False
    num_epochs = 100
    activation = "leakyrelu"
    lr = 0.001

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
    )

    x_id_train_loader = convert_dataloader(
        x_id_train, batch_size=len(x_id_train) // 5, shuffle=True, drop_last=True
    )

    min_lr, max_lr, half_iter = run_auto_lr_range_v4(
        x_id_train_loader,
        bae_model,
        window_size=1,
        num_epochs=10,
        run_full=False,
        plot=False,
        verbose=False,
        save_mecha="copy" if bae_type == "vae" else "file",
    )

    if isinstance(bae_model, BAE_SGHMC):
        bae_model.fit(
            x_id_train_loader,
            burn_epoch=int(num_epochs * 2 / 3),
            sghmc_epoch=num_epochs // 3,
            clear_sghmc_params=True,
        )
    else:
        time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)

    # === PREDICTIONS ===
    exp_man = ExperimentManager(folder_name="experiments")

    # predict and evaluate
    (bae_id_pred, bae_ood_pred), (
        (eval_auroc, retained_res_all, misclas_res_all)
    ) = evaluate_ood_unc(
        bae_model=bae_model,
        x_id_train=x_id_train,
        x_id_test=x_id_test,
        x_ood_test=x_ood_test,
        exp_name=exp_name,
        exp_params=exp_params,
        eval_ood_unc=eval_ood_unc,
        exp_man=exp_man,
        ret_flatten_nll=False,
        cdf_dists=["norm", "uniform", "ecdf", "expon"],
        round_deci=250,
    )

    # eval per sensor
    nll_key = "nll"
    sensor_aurocs_ = {}
    for i in range(x_inliers.shape[sensor_xis]):
        e_nll_id_ = flatten_nll(bae_id_pred[nll_key].mean(0)[:, [i]])
        e_nll_ood_ = flatten_nll(bae_ood_pred[nll_key].mean(0)[:, [i]])

        sensor_aurocs_.update({str(i): calc_auroc(e_nll_id_, e_nll_ood_)})

    # save eval per sensor
    pickle_sensor_auroc = exp_man.encode(exp_params)
    exp_man.update_csv(
        exp_params,
        insert_pickle=pickle_sensor_auroc,
        csv_name=sensor_auroc_filename,
    )
    exp_man.encode_pickle(pickle_sensor_auroc, data=sensor_aurocs_)

    print(eval_auroc)

    # eval per level of ood
    e_nll_id = flatten_nll(bae_id_pred[nll_key]).mean(0)
    e_nll_ood = flatten_nll(bae_ood_pred[nll_key]).mean(0)
    v_nll_id = flatten_nll(bae_id_pred[nll_key]).var(0)
    v_nll_ood = flatten_nll(bae_ood_pred[nll_key]).var(0)

    y_ood_levels = np.unique(y_target[y_arg_ood, target_dim]).astype(int)
    for level in y_ood_levels:
        e_ood_level = e_nll_ood[
            np.argwhere(y_target[y_arg_ood, target_dim] == level)[:, 0]
        ]
        v_ood_level = v_nll_ood[
            np.argwhere(y_target[y_arg_ood, target_dim] == level)[:, 0]
        ]
        auroc_levels = {
            "OOD_LEVEL": level,
            "E_AUROC": calc_auroc(e_nll_id, e_ood_level),
            "V_AUROC": calc_auroc(v_nll_id, v_ood_level),
        }
        exp_man.update_csv(
            exp_man.concat_params_res(exp_params, auroc_levels),
            csv_name=ood_level_auroc_filename,
        )
        print(auroc_levels)


# CALC AUROC DIY
# plt.figure()
# y_unc_concat = np.concatenate((v_nll_id, v_ood_level))
# uniq_thresholds = np.unique(y_unc_concat)

# uniq_thresholds = np.linspace(np.min(y_unc_concat), np.max(y_unc_concat), 100)


# desc_score_indices = np.argsort(y_unc_concat, kind="mergesort")[::-1]
# y_unc_concat = y_unc_concat[desc_score_indices]
# # y_true = y_true[desc_score_indices]
#
# print(len(uniq_thresholds))
#
# # distinct_value_indices = np.where(np.diff(y_unc_concat))[0]
# # threshold_idxs = np.r_[distinct_value_indices, y_unc_concat.size - 1]
#
# optimal_idxs = np.where(
#     np.r_[True, np.diff(y_unc_concat, 2)]
# )[0]
#
