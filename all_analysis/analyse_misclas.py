import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import sem

# WIP
base_folder = "results/STRATH_UNCOOD/"
base_filename = "STRATH_FORGE_UNCOOD"
# base_folder = "results/ZEMA_UNCOOD/"
# base_filename = "ZEMA_HYD_UNCOOD"

# =========================MISCLASS DETECTION============================

res_misclas = pd.read_csv("results/STRATH_UNCOOD/STRATH_FORGE_UNCOOD_misclas_perf.csv")
bae_type = "sghmc"
full_likelihood = "mse"
ss_id = "[13, 71]"
# ss_id = "71"
res_misclas = res_misclas[
    (res_misclas["bae_type"] == bae_type)
    & (res_misclas["full_likelihood"] == full_likelihood)
    & (res_misclas["ss_id"] == ss_id)
]
res_misclas["prc-ratio"] = res_misclas["avgprc"] / res_misclas["baseline"]
all_auroc = res_misclas.groupby(["dist", "norm", "unc_method"]).mean()["auroc"]
all_prc_ratio = res_misclas.groupby(["dist", "norm", "unc_method"]).mean()["prc-ratio"]

# apply filter
unc_method = "exceed"
type_err = "all_err"
norm_scaling = True
dist = "norm"
filter_res_retained = res_misclas[
    (res_misclas["bae_type"] == bae_type)
    & (res_misclas["norm"] == norm_scaling)
    & (res_misclas["dist"] == dist)
]

type2_pos = []
type2_neg = []

# load pickle
pickle_files = filter_res_retained["pickle"].unique()
pickle_folder = "results/STRATH_UNCOOD/pickles/"
for file in pickle_files:
    pickle_dict = pickle.load(
        open(
            os.path.join(pickle_folder, file),
            "rb",
        )
    )
    if type_err in pickle_dict[unc_method]["y_unc_boxplot"].keys():
        type2_pos.append(pickle_dict[unc_method]["y_unc_boxplot"][type_err][0])
        type2_neg.append(pickle_dict[unc_method]["y_unc_boxplot"][type_err][1])
type2_pos = np.hstack(type2_pos)
type2_neg = np.hstack(type2_neg)

plt.figure()
plt.violinplot([type2_pos * 4, type2_neg * 4])

plt.figure()
plt.boxplot([type2_pos * 4, type2_neg * 4])

plt.figure()
# plt.hist(type2_pos)
plt.hist(type2_neg)


# plt.figure()
# # plt.boxplot([val for val in pickle_dict["proba-total"]["y_unc_boxplot"]["all_err"]])
# plt.violinplot(
#     [val * 4 for val in pickle_dict["proba-total"]["y_unc_boxplot"]["all_err"]]
# )
#
# plt.figure()
# # plt.boxplot([val for val in pickle_dict["proba-total"]["y_unc_boxplot"]["all_err"]])
# plt.violinplot(
#     [val * 4 for val in pickle_dict["proba-total"]["y_unc_boxplot"]["type2"]]
# )
