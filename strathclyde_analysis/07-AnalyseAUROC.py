# With results from 06, find out which sensors are important to which target dimensions

import pandas as pd
import numpy as np

# data_path = "experiments\\STRATH_BAE_TEST_HEATING.csv"
data_path = "experiments\\STRATH_BAE_TEST_FORGING.csv"
auroc_res = pd.read_csv(data_path)

unique_dimensions = np.unique(auroc_res["target_dim"])
auroc_threshold = 0.7
apply_fft = False

filtered_res = []
for dim in unique_dimensions:
    slice_dim = auroc_res[
        (auroc_res["target_dim"] == dim)
        & (auroc_res["E_AUROC"] >= auroc_threshold)
        & (auroc_res["apply_fft"] == apply_fft)
    ]
    filtered_res.append(slice_dim)

filtered_res = pd.concat(filtered_res)

print(filtered_res)

# create target and sensor list
for target_dim in np.unique(filtered_res["target_dim"]):
    if target_dim == 2:
        sensors = filtered_res[filtered_res["target_dim"] == target_dim]["ss_id"].values
