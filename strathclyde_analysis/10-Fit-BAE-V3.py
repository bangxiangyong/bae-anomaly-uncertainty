import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import itertools
import seaborn as sns
from tqdm import tqdm

from baetorch.baetorch.evaluation import (
    calc_auroc,
    concat_ood_score,
    calc_avgprc_perf,
    evaluate_retained_unc,
    evaluate_random_retained_unc,
    evaluate_misclas_detection,
    convert_hard_pred,
    retained_top_unc_indices,
    summarise_retained_perf,
    evaluate_retained_unc_v2,
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


bae_set_seed(123)

# data preproc. hyper params
valid_size = 0.0
tolerance_scale = 1.0
resample_factor = 1
# scaling = ["before"]
scaling = []
mode = "forging"
# mode = "heating"
pickle_path = "pickles"
# apply_fft = False
# apply_fft = True

tukey_threshold = 1.5

target_dims_all = [1, 2, 7, 9, 12, 17]

heating_traces = pickle.load(open(pickle_path + "/" + "heating_inputs.p", "rb"))
forging_traces = pickle.load(open(pickle_path + "/" + "forging_inputs.p", "rb"))
column_names = pickle.load(open(pickle_path + "/" + "column_names.p", "rb"))
cmm_data = pickle.load(open(pickle_path + "/" + "strath_outputs_v2_err.p", "rb")).values
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

# Grid search
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13, 14, 71, 64, [13, 14, 71, 64]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [True, False],
#     "latent_factor": [0.25, 0.5, 1, 2],
#     "bae_type": ["ae", "ens", "mcd", "sghmc", "vi", "vae"],
#     "full_likelihood": full_likelihood,
# }

# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     # "ss_id": [13, 64, [13, 64]],
#     "ss_id": [13, 71, [13, 71]],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [True],
#     "latent_factor": [0.5],
#     "bae_type": ["ae"],
#     "full_likelihood": ["hetero-gauss"],
# }

# extreme resampling
# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "mode": ["forging"],
#     "ss_id": [13],
#     # "ss_id": [[13, 71]],
#     "target_dim": [2],
#     "resample_factor": [10],
#     "skip": [False],
#     "latent_factor": [0.5],
#     "bae_type": ["ens"],
#     "full_likelihood": ["mse", "hetero-gauss", "bernoulli", "cbernoulli"],
# }

grid = {
    "random_seed": random_seeds,
    "apply_fft": [False],
    "mode": ["forging"],
    "ss_id": [71],
    # "ss_id": [[13, 71]],
    "target_dim": [2],
    "resample_factor": [10],
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
    # lr = 0.000000000001

    input_dim = x_id_train.shape[-1]
    latent_dim = int(np.product(x_id_train.shape[1:]) * latent_factor)
    # latent_dim = 32
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
            # "architecture": [latent_dim, latent_dim // 2],
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
        # twin_params={"activation": "none", "norm": "none"},
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
        x_id_train,
        batch_size=len(x_id_train) // 5,
        shuffle=True,
    )

    # min_lr, max_lr, half_iter = run_auto_lr_range_v4(
    #     x_id_train_loader,
    #     bae_model,
    #     window_size=1,
    #     num_epochs=10,
    #     run_full=False,
    # )

    if isinstance(bae_model, BAE_SGHMC):
        bae_model.fit(
            x_id_train_loader,
            burn_epoch=num_epochs,
            sghmc_epoch=num_epochs // 2,
            clear_sghmc_params=True,
        )
    else:
        time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)

    # === PREDICTIONS ===

    # start predicting
    nll_key = "nll"

    bae_id_pred = bae_model.predict(x_id_test, select_keys=[nll_key])
    bae_ood_pred = bae_model.predict(x_ood_test, select_keys=[nll_key])

    # get ood scores
    e_nll_id = flatten_nll(bae_id_pred[nll_key]).mean(0)
    e_nll_ood = flatten_nll(bae_ood_pred[nll_key]).mean(0)
    var_nll_id = flatten_nll(bae_id_pred[nll_key]).var(0)
    var_nll_ood = flatten_nll(bae_ood_pred[nll_key]).var(0)

    eval_res = {
        "E_AUROC": calc_auroc(e_nll_id, e_nll_ood),
        "V_AUROC": calc_auroc(var_nll_id, var_nll_ood),
    }

    exp_man = ExperimentManager()
    res = exp_man.concat_params_res(exp_params, eval_res)
    exp_man.update_csv(exp_params=res, csv_name=exp_name + "AUROC.csv")

    # special case for evaluating bce vs mse
    if (
        likelihood == "gaussian" and not twin_output and homoscedestic_mode == "none"
    ) or likelihood == "bernoulli":
        eval_res = evaluate_bce_se(bae_model, x_id_test, x_ood_test)

        exp_man = ExperimentManager()
        res = exp_man.concat_params_res(exp_params, eval_res)
        exp_man.update_csv(exp_params=res, csv_name=exp_name + "BCE_VS_SE.csv")

    # === EVALUATE OUTLIER UNCERTAINTY ===
    # convert to outlier probability
    # 1. get reference distribution of NLL scores
    if valid_size == 0:
        bae_id_ref_pred = bae_model.predict(x_id_train, select_keys=[nll_key])

    all_y_true = np.concatenate(
        (np.zeros_like(e_nll_id), np.ones_like(e_nll_ood))
    ).astype(int)
    all_var_nll_unc = np.concatenate((var_nll_id, var_nll_ood))
    concat_e_nll = concat_ood_score(e_nll_id, e_nll_ood)[1]

    # 2. define cdf distribution of OOD scores
    for cdf_dist in ["norm", "uniform", "ecdf"]:
        # for cdf_dist in ["ecdf"]:
        bae_proba_model = BAE_Outlier_Proba(
            dist_type=cdf_dist,
            norm_scaling=True,
            fit_per_bae_sample=False if (bae_type == "vae") else True,
        )
        bae_proba_model.fit(bae_id_ref_pred[nll_key])

        # for norm_scaling in [True, False]:
        for norm_scaling in [True]:
            id_proba_mean, id_proba_unc = bae_proba_model.predict(
                bae_id_pred[nll_key], norm_scaling=norm_scaling
            )
            ood_proba_mean, ood_proba_unc = bae_proba_model.predict(
                bae_ood_pred[nll_key], norm_scaling=norm_scaling
            )

            # CONVERT HARD PRED
            all_proba_mean = np.concatenate((id_proba_mean, ood_proba_mean))
            all_hard_proba_pred = convert_hard_pred(all_proba_mean, p_threshold=0.5)

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

            retained_random_res = evaluate_random_retained_unc(
                all_outprob_mean=concat_e_nll,
                all_hard_pred=all_hard_proba_pred,
                all_y_true=all_y_true,
                repetition=10,
                retained_percs=retained_percs,
            )

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
                    "random": retained_random_res,
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
                    all_hard_pred=all_hard_proba_pred,
                    all_y_true=all_y_true,
                    all_unc=all_proba_unc,
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
            exp_man = ExperimentManager()
            unc_method = {"dist": cdf_dist, "norm": norm_scaling}
            base_method_columns = exp_man.concat_params_res(exp_params, unc_method)
            pickle_retained = exp_man.encode(str(base_method_columns) + "-retained")
            pickle_misclas = exp_man.encode(str(base_method_columns) + "-misclas")

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
            pickle_name = exp_man.encode_pickle(pickle_retained, data=retained_res_all)

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
            pickle_name = exp_man.encode_pickle(pickle_misclas, data=misclas_res_all)

print("AUROC")
print(retained_prob_unc_res["auroc"])

# ========EVALUATE RETAINED ?
all_unc = all_exceed_unc
retained_perc = 0.8
(
    filtered_unc_indices,
    retained_id_indices,
    retained_ood_indices,
) = retained_top_unc_indices(
    all_y_true, all_unc, retained_perc=retained_perc, return_referred=False
)


# # loop across retained percentages
# for retained_perc in retained_percs:
#     try:
#         (
#             filtered_unc_indices,
#             retained_id_indices,
#             retained_ood_indices,
#         ) = retained_top_unc_indices(
#             all_y_true, all_unc, retained_perc=retained_perc, return_referred=False
#         )
#
#         retained_id_outprob_mean = all_outprob_mean[filtered_unc_indices][
#             retained_id_indices
#         ]
#         retained_ood_outprob_mean = all_outprob_mean[filtered_unc_indices][
#             retained_ood_indices
#         ]
#
#         retained_y_true = all_y_true[filtered_unc_indices][
#             np.concatenate((retained_id_indices, retained_ood_indices))
#         ].astype(int)
#         retained_hard_pred = all_hard_pred[filtered_unc_indices][
#             np.concatenate((retained_id_indices, retained_ood_indices))
#         ].astype(int)


# ============

# PLOT HISTO SAMPLE
plt.figure()

sns.histplot(e_nll_id, stat="probability", color="tab:blue")
sns.histplot(e_nll_ood, stat="probability", color="tab:orange")
# plt.hist(e_nll_id, color="tab:blue", density=True)
# plt.hist(e_nll_ood, color="tab:orange", density=True)

# weights = np.ones_like(e_nll_id) / (len(e_nll_id))
# plt.hist(e_nll_id, weights=weights)
# weights = np.ones_like(e_nll_ood) / (len(e_nll_ood))
# plt.hist(e_nll_ood, weights=weights)

plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
plt.legend(["Within-tolerance", "Out-of-tolerance"])
plt.ylabel("Probability density")
plt.suptitle("AUROC:" + str(calc_auroc(e_nll_id, e_nll_ood).round(3)))

# sensor_i = 1
# plt.figure()
# for sample_i in x_id_test:
#     plt.plot(sample_i[sensor_i], color="tab:blue", alpha=0.5)
# for sample_i in x_ood_test:
#     plt.plot(sample_i[sensor_i], color="tab:orange", alpha=0.5)
#
# plt.figure()


# plt.figure()
# plt.scatter(x=e_nll_id, y=var_nll_id)
# plt.scatter(x=e_nll_ood, y=var_nll_ood)
# ===============PLOT RESULTS================

# plt.figure()
# plt.hist(e_nll_id, density=True, alpha=0.75)
# plt.hist(e_nll_ood, density=True, alpha=0.75)
#
# plt.figure()
# plt.boxplot([e_nll_id, e_nll_ood])

forging_sensors = [2, 25, 33, 65, 11, 13, 14, 38, 54, 64, 71, 82]
filtered_forging_indices = []
for i in forging_sensors:
    forging_sensor_ = column_names[i]
    if "NOM" not in forging_sensor_:
        filtered_forging_indices.append(i)

# x_forging
# pearsonr()

sensor_i = filtered_forging_indices

heating_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)
forging_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)

# get heating/forging traces
x_heating = [heating_trace[:, heating_sensors] for heating_trace in heating_traces]
x_forging = [forging_trace[:, forging_sensors] for forging_trace in forging_traces]

# cut to smallest trace length
min_heat_length = np.array([len(trace) for trace in x_heating]).min()
min_forge_length = np.array([len(trace) for trace in x_forging]).min()

x_heating = np.array([heating_trace[:min_heat_length] for heating_trace in x_heating])
x_forging = np.array([forging_trace[:min_forge_length] for forging_trace in x_forging])

# === ANALYSE FORGING SENSORS CORRELATION ===
forging_sensor_labels = [
    column_names[i].split("[")[0].strip().replace("_", "-").replace(" ", "-")
    for i in filtered_forging_indices
]
forge_sensor_pairs = {
    nam: i for nam, i in zip(forging_sensor_labels, filtered_forging_indices)
}
forging_sensor_corrs = []
for x_forging_i in x_forging:
    forge_sensor_corr = pd.DataFrame(x_forging_i)
    forging_sensor_corrs.append(forge_sensor_corr.corr().values)
forging_sensor_corrs = np.array(forging_sensor_corrs).mean(0).round(2) + 0.0


# matrix
# matrix = np.triu(forge_sensor_corrs)
# sns.heatmap(forge_sensor_corrs, annot=True, mask=matrix)

import seaborn as sns

plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True

# mask
mask = np.triu(np.ones_like(forging_sensor_corrs, dtype=np.bool_))
# adjust mask and df
mask = mask[1:, :-1]
corr = forging_sensor_corrs[1:, :-1].copy()
# plot heatmap
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="BrBG",
    vmin=-1,
    vmax=1,
    cbar_kws={"shrink": 0.8},
    yticklabels=forging_sensor_labels[1:],
    xticklabels=forging_sensor_labels[:-1],
)
plt.tight_layout()


# remove sensors
remove_sensors = [14, 64, 65, 38, 64]
final_sensor_list = [i for i in forging_sensors if i not in remove_sensors]

# ==================

plt.figure()
plt.hist(x_id_test.flatten(), density=True, alpha=0.5)
plt.hist(x_ood_test.flatten(), density=True, alpha=0.5)

# =======PLOT SAMPLE OF VISUALISATION======
# start predicting
nll_key = "nll"

bae_id_pred = bae_model.predict(x_id_test, select_keys=[nll_key, "y_mu"])
bae_ood_pred = bae_model.predict(x_ood_test, select_keys=[nll_key, "y_mu"])

plt.figure()
for x, y_pred in zip(x_id_test, bae_id_pred[nll_key].mean(0)):
    plt.plot(x[0], color="tab:blue", alpha=0.7)
    # plt.plot(y_pred[0], color="tab:blue", alpha=0.7)

for x, y_pred in zip(x_ood_test, bae_ood_pred[nll_key].mean(0)):
    plt.plot(x[0], color="tab:orange", alpha=0.7)
    # plt.plot(y_pred[0], color="tab:orange", alpha=0.7)


# ============ILLUSTRATE PROBLEM WITH EXCEED===============


plt.figure()
plt.hist(all_exceed_unc, density=True, bins=20, alpha=0.75)
plt.hist(all_proba_unc * 4, density=True, bins=20, alpha=0.75)


# ============NEW EVALUATION=================
proba_unc_key = "total"
all_proba_unc = np.concatenate(
    (id_proba_unc[proba_unc_key], ood_proba_unc[proba_unc_key])
)
retained_aurocs = []
retained_percs = []
# all_proba_unc_ = np.copy(all_exceed_unc)
all_proba_unc_ = np.copy(all_proba_unc)

thresholds = np.unique(all_proba_unc_)

for threshold in thresholds:
    retained_indices = np.argwhere(all_proba_unc_ <= threshold)[:, 0]
    retained_y_trues = all_y_true[retained_indices]
    perc_retained = len(retained_indices) / len(all_proba_unc_)

    retained_id_indices = np.argwhere(retained_y_trues == 0)[:, 0]
    retained_ood_indices = np.argwhere(retained_y_trues == 1)[:, 0]

    if len(retained_ood_indices) >= 3:
        nll_id = concat_e_nll[retained_indices][retained_id_indices]
        nll_ood = concat_e_nll[retained_indices][retained_ood_indices]

        auroc_retained = calc_auroc(nll_id, nll_ood)

        retained_aurocs.append(auroc_retained)
        retained_percs.append(perc_retained)

retained_aurocs = np.array(retained_aurocs)
retained_percs = np.array(retained_percs)

w_auroc = np.average(retained_aurocs, weights=retained_percs)

print(w_auroc)

# plot retained percentage
plt.figure()
plt.plot(retained_percs, retained_aurocs)

plt.figure()
plt.scatter(all_proba_unc, all_exceed_unc)


plt.figure()
plt.plot(retained_varnll_res["valid_perc"], retained_varnll_res["auroc"])
plt.plot(retained_prob_unc_res["valid_perc"], retained_prob_unc_res["auroc"])


# =====================================================
from scipy.stats import binom


def calc_exceed(len_train, test_prob, test_hard_pred, contamination):
    n = len_train
    n_anom = int(n * contamination)  # expected anomalies

    # conf_func = np.vectorize(lambda p: p ** n)
    conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
    exWise_conf = conf_func(test_prob)
    np.place(exWise_conf, test_hard_pred == 0, 1 - exWise_conf[test_hard_pred == 0])
    # return 1 - exWise_conf
    return exWise_conf


contamination = 0
n = len(x_id_train)
n_anom = int(n * contamination)
exWise_conf = 1 - binom.cdf(n - n_anom, n, all_proba_mean)
exWise_conf = np.power(all_proba_mean, n)
# conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
# exWise_conf = conf_func(all_proba_mean)
# exWise_conf =

np.place(
    exWise_conf, all_hard_proba_pred == 0, 1 - exWise_conf[all_hard_proba_pred == 0]
)

# xceed_test = calc_exceed(len(x_id_train), all_proba_mean, all_hard_proba_pred, 0)
