import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.evaluation import calc_auroc
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood_v2.util.sensor_preproc import (
    MinMaxSensor,
    FFT_Sensor,
    Resample_Sensor,
)
from util.evaluate_ood import flag_tukey_fence
from util.exp_manager import ExperimentManager


def apply_along_sensors(method, data, axis=1, **kwargs):
    transformed_data = np.array(
        [
            np.apply_along_axis(method, arr=data[:, :, 0], axis=axis, **kwargs)
            for i in range(data.shape[-1])
        ]
    )
    transformed_data = np.moveaxis(transformed_data, 0, -1)
    return transformed_data


def resample_data(data_series, n=10):
    temp_df = pd.DataFrame(data_series)
    resampled_data = (
        temp_df.groupby(np.arange(len(temp_df)) // n).mean().values.squeeze(-1)
    )
    return resampled_data


def normalise_minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# data preproc. hyper params
# random_seed = 775
# random_seed = 85
# random_seed = 111111
random_seed = 9
bae_set_seed(random_seed)


tolerance_scale = 1.0
resample_factor = 1
# scaling = ["before"]
scaling = []
mode = "forging"
# mode = "heating"
pickle_path = "pickles"
# apply_fft = False
apply_fft = True
sensor_i = 10

target_dims_all = [1, 2, 7, 9, 12, 17]


heating_traces = pickle.load(open(pickle_path + "/" + "heating_inputs.p", "rb"))
forging_traces = pickle.load(open(pickle_path + "/" + "forging_inputs.p", "rb"))
column_names = pickle.load(open(pickle_path + "/" + "column_names.p", "rb"))
cmm_data = pickle.load(open(pickle_path + "/" + "strath_outputs_v2_err.p", "rb")).values
cmm_data = np.abs(cmm_data)

target_dims = [2]

heating_features = np.argwhere(column_names == "ForgingBox_Temp")

print(column_names[[77, 76, 81, 2, 3, 5, 6]])
# print(column_names[[9, 8, 10]])
# heating_sensors = np.array([75])
heating_sensors = np.array([75, 77, 76])
# heating_sensors = np.array([75])
# heating_sensors = np.array([75])
# forging_sensors = np.array([2, 3, 5, 6])  # 236 (75, 2 , 7?)
# forging_sensors = np.array([6])  # 236 (75, 2 , 7?)
# forging_sensors = np.array([81,82])  # 236 (75, 2 , 7?)
# forging_sensors = np.array([6])  # 76, 81 corresponds to part 70 breakdown!
# forging_sensors = np.array([6])
# forging_sensors = np.array([9])  # 2021 AUG - SEEMS TO BE GOOD WITH FFT
# forging_sensors = np.array([11])
# forging_sensors = np.array([17])
forging_sensors = np.array([sensor_i])
# fmt: off
heating_sensors_all = np.array([63,64,70,71,79,76,77,78])-1
heating_sensors_pairs = np.array([[63,70],[64,71]])-1
forging_sensors_all = np.array([3,4,6,7,10,18,26,34,42,50,66,12,28,20,36,44,52,68,14,22,30,38,46,54,15,23,31,39,47,55,16,24,32,40,57,65,72,80,82,83])-1
forging_sensor_pairs = np.array([[10,34], [26,50], [12,36],[28,52],[22,46],[23,47]])-1
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


# plot some samples
# plt.figure()
# for heating_trace in heating_traces:
#     plt.plot(heating_trace[:, heating_sensors], color="tab:blue", alpha=0.7)
#
# plt.figure()
# for forging_trace in forging_traces:
#     plt.plot(forging_trace[:, forging_sensors], color="tab:blue", alpha=0.7)

# get heating/forging traces
x_heating = [heating_trace[:, heating_sensors] for heating_trace in heating_traces]
x_forging = [forging_trace[:, forging_sensors] for forging_trace in forging_traces]

# cut to smallest trace length
min_heat_length = np.array([len(trace) for trace in x_heating]).min()
min_forge_length = np.array([len(trace) for trace in x_forging]).min()

x_heating = np.array([heating_trace[:min_heat_length] for heating_trace in x_heating])
x_forging = np.array([forging_trace[:min_forge_length] for forging_trace in x_forging])

# scale min max here?
if "before" in scaling:
    x_heating = normalise_minmax(x_heating)
    x_forging = normalise_minmax(x_forging)


plt.figure()
plt.boxplot(list(np.moveaxis(cmm_data, 0, 1)))


# ood_args = np.argwhere((y_all < y_all.mean()-tolerance_scale*y_all.std()) | (y_all > y_all.mean()+tolerance_scale*y_all.std()))[:,0]
# id_args = np.argwhere((y_all >= y_all.mean()-tolerance_scale*y_all.std()) & (y_all <= y_all.mean()+tolerance_scale*y_all.std()))[:,0]

# ood and id
tukey_flags = np.apply_along_axis(flag_tukey_fence, arr=cmm_data, axis=0, level=1.00)
tukey_flags_filtered = (
    tukey_flags[:, target_dims] if len(target_dims) > 0 else tukey_flags
)
ood_args = np.argwhere(tukey_flags_filtered.sum(1) >= 1)[:, 0]
id_args = np.argwhere(tukey_flags_filtered.sum(1) == 0)[:, 0]

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
    x_id_train, train_size=0.80, shuffle=True, random_state=random_seed
)
# x_id_test = x_id_train

if apply_fft:
    x_id_train = FFT_Sensor().transform(x_id_train)
    x_id_test = FFT_Sensor().transform(x_id_test)
    x_ood_test = FFT_Sensor().transform(x_ood_test)

# sensor FFT
# N = x_id_train.shape[-1]
# trace = x_id_train[0, 0]
#
# trace_fft = 2.0 / N * np.abs(fft(trace)[: N // 2])
# trace_fft = trace_fft[1:]
#
#
# plt.figure()
# plt.plot(fft_x_id_train[0, 0])
#
# plt.figure()
# plt.plot(trace_fft)

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


plt.figure()
for sample in x_id_train[:, 0]:
    plt.plot(sample, color="tab:blue", alpha=0.5)
for sample in x_id_test[:, 0]:
    plt.plot(sample, color="tab:green", alpha=0.5)
for sample in x_ood_test[:, 0]:
    plt.plot(sample, color="tab:orange", alpha=0.5)


# ===============FIT BAE===============

skip = True
# skip = False
use_cuda = True
twin_output = False
# twin_output = True
# homoscedestic_mode = "every"
# homoscedestic_mode = "single"
homoscedestic_mode = "none"
clip_data_01 = True
# likelihood = "ssim"
likelihood = "gaussian"
# likelihood = "laplace"
# likelihood = "bernoulli"
# likelihood = "cbernoulli"
# likelihood = "truncated_gaussian"
# weight_decay = 0.0000000001
weight_decay = 0.00000000001
# weight_decay = 0.000001
# weight_decay = 0.0000001
# weight_decay = 0.01
# weight_decay = 0.000001
anchored = False
# anchored = True
# sparse_scale = 0.0000001
sparse_scale = 0.00
n_stochastic_samples = 100
n_ensemble = 5
# n_ensemble = 1
bias = False
se_block = False
norm = "layer"
self_att = False
self_att_transpose_only = False
num_epochs = 50
activation = "selu"
dropout = 0.0
lr = 0.001

input_dim = x_id_train.shape[-1]
latent_dim = 500
chain_params = [
    {
        "base": "conv1d",
        "input_dim": input_dim,
        "conv_channels": [x_id_train.shape[1], 10],
        "conv_stride": [2],
        "conv_kernel": [5],
        "activation": activation,
        "norm": norm,
        "se_block": se_block,
        # "order": ["base", "activation", "norm"],
        "order": ["base", "norm", "activation"],
        # "order": ["norm", "base", "activation"],
        "bias": bias,
        "dropout": dropout,
        "last_norm": norm,
    },
    {
        "base": "linear",
        "architecture": [latent_dim, latent_dim // 2],
        "activation": activation,
        "norm": norm,
        "last_norm": norm,
    },
]

bae_model = BAE_Ensemble(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=norm,
    twin_output=twin_output,
    # twin_params={"activation": "none", "norm": False},
    # twin_params={"activation": "leakyrelu", "norm": "none"},  # truncated gaussian
    # twin_params={"activation": "leakyrelu", "norm": True},  # truncated gaussian
    twin_params={"activation": "softplus", "norm": "none"},
    # twin_params={"activation": "softplus", "norm": True},
    skip=skip,
    use_cuda=use_cuda,
    scaler_enabled=False,
    homoscedestic_mode=homoscedestic_mode,
    likelihood=likelihood,
    weight_decay=weight_decay,
    num_samples=n_ensemble,
    sparse_scale=sparse_scale,
    anchored=anchored,
    learning_rate=lr,
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
    bae_model.fit(x_id_train_loader, burn_epoch=num_epochs, sghmc_epoch=num_epochs // 2)
else:
    # time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs + 50)
    time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)


# === PREDICTIONS ===
# switch to evaluation mode
for autoencoder in bae_model.autoencoder:
    autoencoder.eval()

# start predicting
nll_key = "nll"

bae_id_train = bae_model.predict(x_id_train, select_keys=[nll_key])
bae_id_pred = bae_model.predict(x_id_test, select_keys=[nll_key])
bae_ood_pred = bae_model.predict(x_ood_test, select_keys=[nll_key])

# === EVALUATION ===
e_nll_train = bae_id_train[nll_key].mean(0).mean(-1).mean(-1)
e_nll_id = bae_id_pred[nll_key].mean(0).mean(-1).mean(-1)
e_nll_ood = bae_ood_pred[nll_key].mean(0).mean(-1).mean(-1)

var_nll_train = bae_id_train[nll_key].var(0).mean(-1).mean(-1)
var_nll_id = bae_id_pred[nll_key].var(0).mean(-1).mean(-1)
var_nll_ood = bae_ood_pred[nll_key].var(0).mean(-1).mean(-1)

# print("AUROC (TRAIN-OOD):" + str(calc_auroc(e_nll_train, e_nll_ood)))
print("AUROC NLL (TRAIN-TEST):" + str(calc_auroc(e_nll_train, e_nll_id)))
print("AUROC NLL (TEST-OOD):" + str(calc_auroc(e_nll_id, e_nll_ood)))

print("AUROC VAR (TRAIN-TEST):" + str(calc_auroc(var_nll_train, var_nll_id)))
print("AUROC VAR (TEST-OOD):" + str(calc_auroc(var_nll_id, var_nll_ood)))

print("NUMBER OF OOD:" + str(len(x_ood_test)))
print(ood_args)
print("NUM UNIQUE OOD PER DIM: ")

print(
    pd.DataFrame(
        [{v: k for k, v in zip(tukey_flags.sum(0), np.arange(tukey_flags.shape[-1]))}]
    ).T
)

exp_man = ExperimentManager()
exp_man.update_csv(exp_params={}, csv_name="STRATH_BAE_TEST1.csv")

# ===============PLOT RESULTS================

# plt.figure()
# plt.hist(e_nll_id, density=True, alpha=0.75)
# plt.hist(e_nll_ood, density=True, alpha=0.75)
#
# plt.figure()
# plt.boxplot([e_nll_id, e_nll_ood])
