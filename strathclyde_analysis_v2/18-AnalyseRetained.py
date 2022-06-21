import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import sem

# WIP


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

plt.figure()
unc_method = "proba-total"
perf_key = "auroc"
norm_scaling = True
dist = "norm"
# ss_id = "[13, 71]"
ss_id = "13"
# ss_id = "71"

res_retained = pd.read_csv("results/retained/STRATH_FORGE_UNCODV6_retained_perf.csv")
# res_retained = pd.read_csv("results/retained/STRATH_FORGE_UNCODV3_misclas_perf.csv")
# legends = []
# global_max = []
# for bae_type in res_retained["bae_type"].unique():
#     # apply filter
#     full_likelihood = "mse"
#     filter_res_retained = res_retained[
#         (res_retained["bae_type"] == bae_type)
#         & (res_retained["full_likelihood"] == full_likelihood)
#         & (res_retained["ss_id"] == ss_id)
#         & (res_retained["norm"] == norm_scaling)
#         & (res_retained["dist"] == dist)
#     ]
#
#     # load pickle
#     pickle_files = filter_res_retained["pickle"].unique()
#     all_max_percs = []
#     all_valid_percs = []
#     all_auroc = []
#     all_interpolate_f = []
#
#     for i, file in enumerate(pickle_files):
#         pickle_dict = pickle.load(
#             open(
#                 os.path.join("results/retained/pickles/", file),
#                 "rb",
#             )
#         )
#         valid_perc = 1 - np.array(pickle_dict[unc_method]["valid_perc"])
#         auroc = pickle_dict[unc_method][perf_key]
#         all_valid_percs.append(valid_perc)
#         all_auroc.append(auroc)
#         all_max_percs.append(np.max(valid_perc))
#         all_interpolate_f.append(interpolate.interp1d(valid_perc, auroc))
#
#     interplates_mean = []
#     interplates_sem = []
#     valid_xi = []
#     for x_i in np.arange(0, 1.05, 0.025):
#         temp = []
#         for f_i, max_i in zip(all_interpolate_f, all_max_percs):
#             # check if more than max
#             if x_i <= max_i:
#                 temp.append(f_i(x_i))
#         if len(temp) >= 3:
#             interplates_mean.append(np.mean(temp))
#             interplates_sem.append(sem(temp))
#             valid_xi.append(x_i)
#     global_max.append(np.max(valid_xi))
#     interplates_mean = np.array(interplates_mean)
#     interplates_sem = np.array(interplates_sem)
#
#     plt.plot(valid_xi, interplates_mean)
#     plt.fill_between(
#         valid_xi,
#         np.clip(interplates_mean - interplates_sem, 0, 1),
#         np.clip(interplates_mean + interplates_sem, 0, 1),
#         alpha=0.15,
#     )
#     legends.append(bae_type)
#
# plt.xlim(right=np.min(global_max))
# plt.legend(legends)


def plot_retained_perf(
    res_retained,
    bae_type,
    norm_scaling,
    dist,
    unc_method="proba-total",
    perf_key="auroc",
    ax=None,
):

    # apply filter
    filter_res_retained = res_retained[
        (res_retained["bae_type"] == bae_type)
        & (res_retained["norm"] == norm_scaling)
        & (res_retained["dist"] == dist)
    ]

    # load pickle
    pickle_files = filter_res_retained["pickle"].unique()
    all_max_percs = []
    all_valid_percs = []
    all_auroc = []
    all_interpolate_f = []

    for i, file in enumerate(pickle_files):
        pickle_dict = pickle.load(
            open(
                os.path.join("results/retained/pickles/", file),
                "rb",
            )
        )
        valid_perc = 1 - np.array(pickle_dict[unc_method]["valid_perc"])
        auroc = pickle_dict[unc_method][perf_key]
        all_valid_percs.append(valid_perc)
        all_auroc.append(auroc)
        all_max_percs.append(np.max(valid_perc))
        all_interpolate_f.append(interpolate.interp1d(valid_perc, auroc))

    interplates_mean = []
    interplates_sem = []
    valid_xi = []
    for x_i in np.arange(0, 1.05, 0.025):
        temp = []
        for f_i, max_i in zip(all_interpolate_f, all_max_percs):
            # check if more than max
            if x_i <= max_i:
                temp.append(f_i(x_i))
        if len(temp) >= 3:
            interplates_mean.append(np.mean(temp))
            interplates_sem.append(sem(temp))
            valid_xi.append(x_i)
    max_valid_prob = np.max(valid_xi)
    interplates_mean = np.array(interplates_mean)
    interplates_sem = np.array(interplates_sem)

    # actual plot
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(valid_xi, interplates_mean)
    ax.fill_between(
        valid_xi,
        np.clip(interplates_mean - interplates_sem, 0, 1),
        np.clip(interplates_mean + interplates_sem, 0, 1),
        alpha=0.15,
    )
    return max_valid_prob


# plt.figure()
# unc_method = "proba-total"
# perf_key = "gss"
# norm_scaling = True
# dist = "norm"
# ss_id = "[13, 71]"
ss_id = "13"
# ss_id = "71"

res_retained = pd.read_csv("results/retained/STRATH_FORGE_UNCODV6_retained_perf.csv")
# filter_res_retained = res_retained[res_retained["ss_id"] == ss_id]

# === PLOT FOR EACH MODEL ===
bae_type_map = {
    "ae": "Det. AE",
    "ens": "BAE, Ensemble",
    "mcd": "BAE, MC-Dropout",
    "vi": "BAE, BayesBB",
    "sghmc": "BAE, SGHMC",
    "vae": "VAE",
}

unc_method = "proba-total"
perf_key = "auroc"
# norm_scaling = True
# dist = "ecdf"



# filter_res_retained_ = filter_res_retained.groupby(["dist","norm"]).mean(0)["weighted-"+perf_key].reset_index()
# best_dist = np.argmax(filter_res_retained_["weighted-gss"])
# filter_res_retained_ = filter_res_retained_.iloc[best_dist]

figsize = (8, 3)
fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
for ss_id, ax in zip(res_retained["ss_id"].unique(), axes):
    legends = []
    global_max = []
    filter_res_retained = res_retained[res_retained["ss_id"] == ss_id]
    for bae_type in res_retained["bae_type"].unique():
        # filter best dist + norm scale
        best_dist_retained_ = filter_res_retained[filter_res_retained["bae_type"] == bae_type]
        best_dist_retained_ = best_dist_retained_.groupby(["dist", "norm"]).mean(0)[
            "weighted-" + perf_key].reset_index()
        best_dist = np.argmax(best_dist_retained_["weighted-" + perf_key])
        best_dist_retained_ = best_dist_retained_.iloc[best_dist]

        norm_scaling = best_dist_retained_["norm"]
        dist = best_dist_retained_["dist"]
        print(best_dist_retained_)

        if bae_type != "jj":
            max_valid_prob = plot_retained_perf(
                filter_res_retained,
                bae_type,
                norm_scaling,
                dist,
                unc_method=unc_method,
                perf_key=perf_key,
                ax=ax,
            )
            global_max.append(max_valid_prob)
            legends.append(bae_type)
    ax.set_xlim(0, np.min(global_max))
legends = [bae_type_map[bae] for bae in legends]
axes[2].legend(legends, fontsize="small")
axes[1].set_xlabel("Referred rate (%)")
axes[0].set_ylabel("AUROC")
fig.tight_layout()


# ===PLOT FOR EACH DISTRIBUTIONS===
dist_map = {
    "uniform": "Uniform",
    "expon": "Exponential",
    "norm": "Gaussian",
    "ecdf": "ECDF",
}

unc_method = "proba-total"
unc_method = "varnll"
perf_key = "auroc"
bae_type = "sghmc"
norm_scaling = False
# dist = "norm"
# res_retained["dist-nscale"] = res_retained["dist"]+res_retained["norm"].astype(str)
figsize = (8, 3)
fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
for ss_id, ax in zip(res_retained["ss_id"].unique(), axes):
    legends = []
    global_max = []
    filter_res_retained = res_retained[res_retained["ss_id"] == ss_id]
    for dist in res_retained["dist"].unique():
        for norm_scaling in res_retained["norm"].unique():
            if bae_type != "ae":
                # plot
                max_valid_prob = plot_retained_perf(
                    filter_res_retained,
                    bae_type,
                    norm_scaling,
                    dist,
                    unc_method=unc_method,
                    perf_key=perf_key,
                    ax=ax,
                )
                global_max.append(max_valid_prob)

                # update label
                dist_label = dist_map[dist]
                if norm_scaling:
                    dist_label = dist_label + "+Norm"
                legends.append(dist_label)
    ax.set_xlim(0, np.min(global_max))
# legends = [dist_map[dist] for dist in legends]
axes[2].legend(legends, fontsize="small")
axes[1].set_xlabel("Referred rate (%)")
axes[0].set_ylabel("AUROC")
fig.tight_layout()

# === PLOT VARNLL METHOD ===


unc_method = "proba-total"
perf_key = "auroc"

figsize = (8, 3)
fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
for ss_id, ax in zip(res_retained["ss_id"].unique(), axes):
    legends = []
    global_max = []
    filter_res_retained = res_retained[res_retained["ss_id"] == ss_id]
    for bae_type in res_retained["bae_type"].unique():
        # filter best dist + norm scale
        best_dist_retained_ = filter_res_retained[filter_res_retained["bae_type"] == bae_type]
        best_dist_retained_ = best_dist_retained_.groupby(["dist", "norm"]).mean(0)[
            "weighted-" + perf_key].reset_index()
        best_dist = np.argmax(best_dist_retained_["weighted-" + perf_key])
        best_dist_retained_ = best_dist_retained_.iloc[best_dist]

        norm_scaling = best_dist_retained_["norm"]
        dist = best_dist_retained_["dist"]
        print(best_dist_retained_)

        if bae_type != "ae":
            max_valid_prob = plot_retained_perf(
                filter_res_retained,
                bae_type,
                norm_scaling,
                dist,
                unc_method=unc_method,
                perf_key=perf_key,
                ax=ax,
            )
            global_max.append(max_valid_prob)
            legends.append(bae_type)
    ax.set_xlim(0, np.min(global_max))
legends = [bae_type_map[bae] for bae in legends]
axes[2].legend(legends, fontsize="small")
axes[1].set_xlabel("Referred rate (%)")
axes[0].set_ylabel("AUROC")
fig.tight_layout()




# collection of aurocs vs percs
#
# f = interpolate.interp1d(
#     1 - np.array(pickle_dict[unc_method]["valid_perc"]),
#     pickle_dict[unc_method]["auroc"],
# )
#
#
# new_xis = np.linspace(0, 1.0, 20)
# plt.figure()
# plt.plot(new_xis, f(new_xis))
# plt.scatter(
#     1 - np.array(pickle_dict[unc_method]["valid_perc"]),
#     pickle_dict[unc_method]["auroc"],
# )
#
# # ---
#
#
# plt.figure()
# plt.plot(1 - new_xis, np.interp(new_xis, valid_perc, auroc))
