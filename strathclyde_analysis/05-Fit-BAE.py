import pickle as pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy as copy

import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.evaluation import calc_auroc
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import (
    Encoder,
    infer_decoder,
    Autoencoder,
)
from baetorch.baetorch.models.base_layer import flatten_np, Conv1DLayers, DenseLayers
from util.evaluate_ood import flag_tukey_fence


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
# target_dim = 2
target_dim = 3
tolerance_scale = 1.0
resample_fft = 1
scaling = ["before"]
mode = "forging"
pickle_path = "pickles"

heating_traces = pickle.load(open(pickle_path + "/" + "heating_inputs.p", "rb"))
forging_traces = pickle.load(open(pickle_path + "/" + "forging_inputs.p", "rb"))
column_names = pickle.load(open(pickle_path + "/" + "column_names.p", "rb"))
cmm_data = pickle.load(open(pickle_path + "/" + "strath_outputs.p", "rb")).values

heating_features = np.argwhere(column_names == "ForgingBox_Temp")

print(column_names[[77, 76, 81, 2, 3, 5, 6]])
# heating_sensors = np.array([76])
heating_sensors = np.array([77, 76])
# forging_sensors = np.array([2, 3, 5, 6])  # 236 (75, 2 , 7?)
# forging_sensors = np.array([6])  # 236 (75, 2 , 7?)
# forging_sensors = np.array([6, 81])  # 236 (75, 2 , 7?)
forging_sensors = np.array([6])  # 76, 81 corresponds to part 70 breakdown!
# forging_sensors = np.array(
#     [
#         8,
#     ]
# )  # 76, 81 corresponds to part 70 breakdown!

# plot some samples
plt.figure()
for heating_trace in heating_traces:
    plt.plot(heating_trace[:, heating_sensors])

plt.figure()
for forging_trace in forging_traces:
    plt.plot(forging_trace[:, forging_sensors])

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

# properly form train/test data

y_all = np.expand_dims(cmm_data[:, target_dim], -1)

plt.figure()
plt.hist(y_all)

# ood_args = np.argwhere((y_all < y_all.mean()-tolerance_scale*y_all.std()) | (y_all > y_all.mean()+tolerance_scale*y_all.std()))[:,0]
# id_args = np.argwhere((y_all >= y_all.mean()-tolerance_scale*y_all.std()) & (y_all <= y_all.mean()+tolerance_scale*y_all.std()))[:,0]

# num_train = 62
# id_args = np.arange(3, num_train)
# ood_args = np.arange(num_train, 80)

tukey_flags = np.apply_along_axis(flag_tukey_fence, arr=cmm_data, axis=0)
ood_args = np.argwhere(tukey_flags.sum(1) >= 1)[:, 0]
id_args = np.argwhere(tukey_flags.sum(1) == 0)[:, 0]

x_heating_train = x_heating[id_args]
x_forging_train = x_forging[id_args]

x_heating_ood = x_heating[ood_args]
x_forging_ood = x_forging[ood_args]

y_train = y_all[id_args]
y_ood = y_all[ood_args]

# expand dims if only one sensor selected
if len(x_heating_train.shape) != 3:
    x_heating_train = np.expand_dims(x_heating_train, -1)
    x_forging_train = np.expand_dims(x_forging_train, -1)
    x_heating_ood = np.expand_dims(x_heating_ood, -1)
    x_forging_ood = np.expand_dims(x_forging_ood, -1)

# perform scaling
if "after" in scaling:
    heating_scaler = MinMaxScaler()
    forging_scaler = MinMaxScaler()

    x_heating_train_shape = x_heating_train.shape
    x_forging_train_shape = x_forging_train.shape
    x_heating_ood_shape = x_heating_ood.shape
    x_forging_ood_shape = x_forging_ood.shape

    x_heating_train = heating_scaler.fit_transform(flatten_np(x_heating_train)).reshape(
        x_heating_train_shape
    )
    x_forging_train = forging_scaler.fit_transform(flatten_np(x_forging_train)).reshape(
        x_forging_train_shape
    )

    x_heating_ood = heating_scaler.transform(flatten_np(x_heating_ood)).reshape(
        x_heating_ood_shape
    )
    x_forging_ood = forging_scaler.transform(flatten_np(x_forging_ood)).reshape(
        x_forging_ood_shape
    )

    x_heating_ood = np.clip(x_heating_ood, 0, 1)
    x_forging_ood = np.clip(x_forging_ood, 0, 1)
    x_forging_train = np.clip(x_forging_train, 0, 1)
    x_heating_train = np.clip(x_heating_train, 0, 1)

# perform resampling
if resample_fft > 1:
    x_heating_train = apply_along_sensors(
        resample_data, x_heating_train, axis=1, n=resample_fft
    )
    x_forging_train = apply_along_sensors(
        resample_data, x_forging_train, axis=1, n=resample_fft
    )
    x_heating_ood = apply_along_sensors(
        resample_data, x_heating_ood, axis=1, n=resample_fft
    )
    x_forging_ood = apply_along_sensors(
        resample_data, x_forging_ood, axis=1, n=resample_fft
    )


# ================================Fit BAE============================================

if mode == "forging":
    x_heating_train = x_forging_train
    x_heating_ood = x_forging_ood

# x_heating_train, x_heating_test = train_test_split(
#     x_heating_train, train_size=0.8, random_state=123, shuffle=True
# )

input_dim = x_heating_train.shape[1]
dense_architecture = [1000]
conv_architecture = [x_heating_train.shape[2], 5, 10]
latent_dim = 100
homoscedestic_mode = "none"
bae_name = "STRATH_BAE"
use_cuda = torch.cuda.is_available()
weight_decay = 0.0000015
num_ensembles = 1
learning_rate = 0.001
num_epochs = 300
encoder = Encoder(
    [
        Conv1DLayers(
            input_dim=input_dim,
            conv_architecture=conv_architecture,
            # conv_kernel=[100, 50],
            # activation="selu",
            # conv_stride=[2, 2],
            conv_kernel=[100, 10],
            activation="leakyrelu",
            conv_stride=[2, 2],
        ),
        DenseLayers(
            architecture=dense_architecture,
            output_size=latent_dim,
            activation="leakyrelu",
            last_activation="leakyrelu",
        ),
    ]
)
decoder_mu = infer_decoder(encoder, activation="leakyrelu", last_activation="sigmoid")

autoencoder = Autoencoder(encoder, decoder_mu, homoscedestic_mode=homoscedestic_mode)


bae_ensemble = BAE_Ensemble(
    model_name=bae_name,
    autoencoder=autoencoder,
    use_cuda=use_cuda,
    anchored=True,
    weight_decay=weight_decay,
    num_samples=num_ensembles,
    likelihood="gaussian",
    homoscedestic_mode=homoscedestic_mode,
    output_clamp=(-1000, 1000),
    learning_rate=learning_rate,
)

bae_ensemble.fit(np.moveaxis(x_heating_train, 1, 2), num_epochs=num_epochs)
# bae_ensemble.fit(np.moveaxis(x_heating_train, 1, 2), num_epochs=num_epochs)

# bae_ensemble.fit(x_heating_train, num_epochs=num_epochs)

nll_ood = bae_ensemble.predict_samples(
    np.moveaxis(x_heating_ood, 1, 2), select_keys=["nll_homo"]
)
# nll_train = bae_ensemble.predict_samples(
#     np.moveaxis(x_heating_test, 1, 2), select_keys=["nll_homo"]
# )
nll_train = bae_ensemble.predict_samples(
    np.moveaxis(x_heating_train, 1, 2), select_keys=["nll_homo"]
)

nll_ood_mu = nll_ood.mean(0).mean(-1).mean(-1)[0]
nll_train_mu = nll_train.mean(0).mean(-1).mean(-1)[0]

nll_ood_std = nll_ood.var(0).mean(-1).mean(-1)[0]
nll_train_std = nll_train.var(0).mean(-1).mean(-1)[0]


plt.figure()
plt.hist(nll_train_mu, alpha=0.7, density=True)
plt.hist(nll_ood_mu, alpha=0.7, density=True)

print("AUROC :" + str(calc_auroc(nll_train_mu, nll_ood_mu)))
print("AUROC :" + str(calc_auroc(nll_train_std, nll_ood_std)))


plt.figure()
plt.hist(nll_train_std, alpha=0.7, density=True)
plt.hist(nll_ood_std, alpha=0.7, density=True)


plt.figure()
plt.boxplot([nll_train_mu, nll_ood_mu])

plt.figure()
plt.plot(nll_ood.mean(0)[0][0, :, 0])
plt.plot(nll_train.mean(0)[0][0, :, 0])


# latent_ood = bae_ensemble.predict_latent(x_heating_ood, transform_pca=False)[0]
# latent_train = bae_ensemble.predict_latent(x_heating_train, transform_pca=False)[0]
#
# plt.figure()
# plt.scatter(latent_train[:, 0], latent_train[:, 1])
# plt.scatter(latent_ood[:, 0], latent_ood[:, 1])
#
# y_pred_train = bae_ensemble.predict_samples(x_heating_train, select_keys=["y_mu"])
# y_pred_ood = bae_ensemble.predict_samples(x_heating_ood, select_keys=["y_mu"])
#
# plt.figure()
# plt.plot(x_heating_train[1, :, 0])
# plt.plot(y_pred_train.mean(0)[0][1, :, 0])
#
# plt.figure()
# plt.plot(x_heating_ood[1, :, 0])
# plt.plot(y_pred_ood.mean(0)[0][1, :, 0])
#
# # plt.plot(y_pred_ood.mean(0)[0][0,:,1])
#
# fig, axes = plt.subplots(6, 3)
# axes = axes.flatten()
# for i, ax in enumerate(axes):
#     ax.scatter(np.arange(80), cmm_data[:, i])
#
# plt.figure()
# plt.scatter(id_args, nll_train_mu)
# plt.scatter(ood_args, nll_ood_mu)
#
# nll_all_mu = np.concatenate((nll_train_mu, nll_ood_mu))
#
# chosen_y_cmm = 2
# cmm_all = cmm_data[np.concatenate((id_args, ood_args)), chosen_y_cmm]
# plt.figure()
# plt.scatter(cmm_all, nll_all_mu)
#
# print(spearmanr(cmm_all, nll_all_mu)[0])
# print(pearsonr(cmm_all, nll_all_mu)[0])
#
#
# cmm_diff = np.diff(cmm_all)
# nll_diff = np.diff(nll_all_mu)
# plt.figure()
# plt.scatter(cmm_diff, nll_diff)
