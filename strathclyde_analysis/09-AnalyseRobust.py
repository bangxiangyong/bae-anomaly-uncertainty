# With results from 06, find out which sensors are important to which target dimensions

import pandas as pd
import numpy as np

data_path = "experiments\\STRATH_BAE_TEST_forging_ROBUST.csv"
# data_path = "experiments\\STRATH_BAE_TEST_FORGING.csv"
auroc_res = pd.read_csv(data_path)

unique_dimensions = np.unique(auroc_res["target_dim"])
auroc_threshold = 0.80

filtered_res = []
for dim in unique_dimensions:
    print("-------------------")
    print(dim)
    slice_dim = auroc_res[(auroc_res["target_dim"] == dim)]
    # print(slice_dim.groupby("ss_id").max())
    print(slice_dim.groupby("ss_id").mean())
    # print(slice_dim.groupby("ss_id").min())
    print(slice_dim.groupby("ss_id").std())

#   filtered_res.append(slice_dim)
#
#   filtered_res = pd.concat(filtered_res)
#
# print(filtered_res)
