import itertools
import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood_v2.util.sensor_preproc import (
    MinMaxSensor,
    FFT_Sensor,
    Resample_Sensor,
)
from util.evaluate_ood import flag_tukey_fence

bae_set_seed(33)

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
grid = {
    "random_seed": random_seeds,
    "apply_fft": [False],
    "mode": ["forging"],
    "ss_id": [13, 71],
    "target_dim": [2],
    "resample_factor": [1, 10, 100, 500],
    "skip": [False],
    "latent_factor": [0.25],
    "bae_type": ["ae"],
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

# STORE RESULTS
downsampling_res = []

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
        x_id_train,
        batch_size=len(x_id_train) // 5,
        shuffle=True,
    )

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

    bae_id_pred = bae_model.predict(x_id_test, select_keys=["se", "bce", "y_mu"])
    bae_ood_pred = bae_model.predict(x_ood_test, select_keys=["se", "bce", "y_mu"])

    # append results
    downsampling_res.append(
        {
            "x_id_test": x_id_test,
            "x_ood_test": x_ood_test,
            "bae_id_pred": bae_id_pred,
            "bae_ood_pred": bae_ood_pred,
            "resample_factor": resample_factor,
            "ss_id": sensor_i,
        }
    )


# =======PLOT SAMPLE OF VISUALISATION======
figsize = (8, 5)
legends = []
fig, axes = plt.subplots(3, 2, sharex=True, figsize=figsize)

for col_i, ss in enumerate(grid["ss_id"]):
    downsampling_res_ss = [res for res in downsampling_res if res["ss_id"] == ss]
    for resample_i in range(len(downsampling_res_ss)):
        n = downsampling_res_ss[resample_i]["resample_factor"]

        x_indices = np.arange(0, 5619, n)

        # plt.plot(x_indices, downsampling_res[i]["bae_id_pred"]["bce"].mean(0)[0][0])
        # plt.plot(x_indices, downsampling_res[i]["bae_id_pred"]["bce"].mean(0)[0][0])
        axes[0, col_i].plot(
            x_indices, downsampling_res_ss[resample_i]["x_id_test"][0][0]
        )
        axes[1, col_i].plot(
            x_indices,
            downsampling_res_ss[resample_i]["bae_id_pred"]["se"].mean(0)[0][0],
        )
        axes[2, col_i].plot(
            x_indices,
            downsampling_res_ss[resample_i]["bae_id_pred"]["bce"].mean(0)[0][0],
        )
        legends.append(str(n))

axes[0, 0].set_title("(a) L-ACTpos sensor", fontsize="medium")
axes[0, 1].set_title("(b) Feedback-SPA", fontsize="medium")
axes[0, 0].legend(legends, fontsize="small")

axes[0, 0].set_ylabel("Normalised " + r"$x_i$")
axes[1, 0].set_ylabel(r"$\mathbb{E}_{\theta}\mathrm{(NLL)}$ , N$(\hat{x},1)$")
axes[2, 0].set_ylabel(r"$\mathbb{E}_{\theta}\mathrm{(NLL)}$ , Ber$(\hat{x})$")

axes[2, 0].set_xlabel("Time " + r"($1\times{10}$ms)")
axes[2, 1].set_xlabel("Time " + r"($1\times{10}$ms)")
fig.tight_layout()
fig.savefig("visualise-downsample.png", dpi=500)
