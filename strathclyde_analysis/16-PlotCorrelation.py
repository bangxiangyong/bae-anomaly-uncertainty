import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baetorch.baetorch.util.seed import bae_set_seed

bae_set_seed(100)

# data preproc. hyper params
valid_size = 0.0
tolerance_scale = 1.0
resample_factor = 10
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

plt.figure(figsize=(7, 5))
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
plt.savefig("forge-sensor-corr.png", dpi=500)

# remove sensors
remove_sensors = [14, 64, 65, 38, 64]
final_sensor_list = [i for i in forging_sensors if i not in remove_sensors]

print(column_names[remove_sensors])
