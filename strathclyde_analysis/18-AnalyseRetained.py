import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# res = pd.read_csv("results/latent/STRATH_FORGE_retained_perf.csv")
res = pd.read_csv("results/likelihood/STRATH_FORGE_retained_perf.csv")

bae_type = "ens"
full_likelihood = "mse"
ss_id = "[13, 71]"
res = res[
    (res["bae_type"] == bae_type)
    & (res["full_likelihood"] == full_likelihood)
    & (res["ss_id"] == ss_id)
]

res["diff-auroc"] = res["weighted-auroc"] - res["baseline-auroc"]
# res["diff-auroc"] = res["weighted-gss"] - res["baseline-gss"]
# res["diff-auroc"] = res["weighted-f1_score"] - res["baseline-f1_score"]

filter_norm_scaling = True
dist = "ecdf"
# dist = "norm"

num_count = res[
    (res["norm"] == False) & (res["dist"] == dist) & (res["unc_method"] == "exceed")
]["diff-auroc"]
num_count2 = res[
    (res["norm"] == True) & (res["dist"] == dist) & (res["unc_method"] == "proba-total")
]["diff-auroc"]


print((num_count > 0).sum() / len(num_count) * 100)
print((num_count2 > 0).sum() / len(num_count2) * 100)

all = res.groupby(["dist", "norm", "unc_method"]).mean()["weighted-gss"]
# all = res.groupby(["dist", "norm", "unc_method"]).mean()["max-gss"]

# all = res.groupby(["dist", "norm", "unc_method"]).mean()["max-auroc"]
# all = res.groupby(["dist", "norm", "unc_method"]).mean()["baseline-auroc"]


# =========================MISCLASS DETECTION============================

res_misclas = pd.read_csv("results/likelihood/STRATH_FORGE_misclas_perf.csv")
bae_type = "ens"
full_likelihood = "mse"
ss_id = "13"
res_misclas = res_misclas[
    (res_misclas["bae_type"] == bae_type)
    & (res_misclas["full_likelihood"] == full_likelihood)
    & (res_misclas["ss_id"] == ss_id)
]
res_misclas["prc-ratio"] = res_misclas["avgprc"] / res_misclas["baseline"]
all = res_misclas.groupby(["dist", "norm", "unc_method"]).mean()["auroc"]
all = res_misclas.groupby(["dist", "norm", "unc_method"]).mean()["prc-ratio"]

# =========================PLOT RESULTS================================

res_retained = pd.read_csv("results/retained/STRATH_FORGE_UNCODV2_retained_perf.csv")

# filter
bae_type = "ens"
full_likelihood = "mse"
ss_id = "[13, 71]"
norm_scaling = False
dist = "norm"
res_retained = res_retained[
    (res_retained["bae_type"] == bae_type)
    & (res_retained["full_likelihood"] == full_likelihood)
    & (res_retained["ss_id"] == ss_id)
    & (res_retained["norm"] == norm_scaling)
    & (res_retained["dist"] == dist)
]

# load pickle
pickle_files = np.unique(res_retained["pickle"].values)

for i, file in enumerate(pickle_files):
    pickle_dict = pickle.load(
        open(
            os.path.join("results/retained/pickles/", file),
            "rb",
        )
    )
