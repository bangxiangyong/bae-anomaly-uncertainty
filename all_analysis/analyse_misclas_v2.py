import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import sem
import seaborn as sns


from baetorch.baetorch.evaluation import calc_auroc

# dataset = "ZEMA"
dataset = "STRATH"

# WIP
base_folder = "results/STRATH_UNCOOD/"
base_filename = "STRATH_FORGE_UNCOOD"
# base_folder = "results/ZEMA_UNCOOD/"
# base_filename = "ZEMA_HYD_UNCOOD"

param_maps = {
    "STRATH": {
        "base_folder": "results/STRATH_UNCOOD/",
        "base_filename": "STRATH_FORGE_UNCOOD",
        "target_col": "ss_id",
        "latex_name": "strath",
        "fig_width": 7.5,
    },
    "ZEMA": {
        "base_folder": "results/ZEMA_UNCOOD_V2/",
        "base_filename": "ZEMA_HYD_UNCOODV2",
        "target_col": "target_dim",
        "latex_name": "zema",
        "fig_width": 9,
    },
}

# LOAD FROM PARAM MAP
base_folder = param_maps[dataset]["base_folder"]
base_filename = param_maps[dataset]["base_filename"]
target_col = param_maps[dataset]["target_col"]
latex_name = param_maps[dataset]["latex_name"]
fig_width = param_maps[dataset]["fig_width"]

# =========================MISCLASS DETECTION============================

# search for best GSS
# perf_key = "f1_score"
perf_key = "gss"
# unc_method = "varnll"
# unc_method = "exceed"
unc_method = "proba-total"
# unc_method = "proba-alea"

full_likelihood = "mse"
# ss_id = "[13, 71]"
# type_err = "type1"
# type_err = "type2"
# type_err = "all_err"
target_col = param_maps[dataset]["target_col"]

res_retained = pd.read_csv(base_folder + base_filename + "_retained_perf_AMENDED.csv")
# res_retained = pd.read_csv(base_folder + base_filename + "_retained_perf.csv")
res_retained["bae_type"].unique()

misclas_traces = {}

err_labels = {
    "type1": {"pos": "TP", "neg": "FP"},
    "type2": {"pos": "TN", "neg": "FN"},
    "all_err": {"pos": "TP+TN", "neg": "FP+FN"},
}

# construct new pd
new_pd = []
# bae_types = ["vi"]
bae_types = ["ae", "ens", "sghmc", "vi", "vae", "mcd"]
target_cols = res_retained[target_col].unique()
# target_cols = [res_retained[target_col].unique()[2]]

# for type_err in ["type1", "type2", "all_err"]:

new_table = []

for bae_type in bae_types:
    for target_key in target_cols:
        filtered_res_retained = res_retained[
            (res_retained["bae_type"] == bae_type)
            & (res_retained["unc_method"] == unc_method)
            & (res_retained[target_col] == target_key)
        ]
        filtered_res_retained = (
            filtered_res_retained.groupby(["norm", "dist"]).mean().reset_index()
        )
        best_row = filtered_res_retained.iloc[
            filtered_res_retained["weighted-" + perf_key].argmax()
        ]

        norm = best_row["norm"]
        dist = best_row["dist"]

        # res misclas
        res_misclas = pd.read_csv(base_folder + base_filename + "_misclas_perf.csv")

        filter_res_misclas = res_misclas[
            (res_misclas["bae_type"] == bae_type)
            & (res_misclas["full_likelihood"] == full_likelihood)
            & (res_misclas[target_col] == target_key)
            & (res_misclas["norm"] == norm)
            & (res_misclas["dist"] == dist)
        ]

        # tabulate table of results
        filter_res_misclas_mean = (
            filter_res_misclas.groupby(["unc_method"]).mean().reset_index()
        )
        filter_res_misclas_mean["avgprc-ratio"] = (
            filter_res_misclas_mean["avgprc"] / filter_res_misclas_mean["baseline"]
        )

        # for unc_method_ in filter_res_misclas_mean["unc_method"]:
        #     temp_ = filter_res_misclas_mean[
        #         filter_res_misclas_mean["unc_method"] == unc_method_
        #     ]
        #     new_table.append(
        #         {
        #             "bae_type": bae_type,
        #             "unc_method": unc_method_,
        #             "auroc": temp_["auroc"].item(),
        #             "avgprc": temp_["avgprc"].item(),
        #             "baseline": temp_["baseline"].item(),
        #             "avgprc-ratio": temp_["avgprc-ratio"].item(),
        #             target_col: target_key,
        #         }
        #     )

        for unc_method_ in ["proba-total"]:
            temp_ = filter_res_misclas_mean[
                filter_res_misclas_mean["unc_method"] == unc_method_
            ]
            new_table.append(
                {
                    "bae_type": bae_type,
                    "unc_method": unc_method_,
                    "auroc": temp_["auroc"].item(),
                    "avgprc": temp_["avgprc"].item(),
                    "baseline": temp_["baseline"].item(),
                    "avgprc-ratio": temp_["avgprc-ratio"].item(),
                    target_col: target_key,
                }
            )
        # append new df
        for type_err in ["type1", "type2", "all_err"]:
            # for type_err in ["all_err"]:
            type2_pos = []
            type2_neg = []

            # load pickle
            pickle_files = filter_res_misclas["pickle"].unique()
            pickle_folder = base_folder + "pickles/"
            for file in pickle_files:
                pickle_dict = pickle.load(
                    open(
                        os.path.join(pickle_folder, file),
                        "rb",
                    )
                )
                if type_err in pickle_dict[unc_method]["y_unc_boxplot"].keys():
                    type2_pos.append(
                        pickle_dict[unc_method]["y_unc_boxplot"][type_err][0]
                    )
                    type2_neg.append(
                        pickle_dict[unc_method]["y_unc_boxplot"][type_err][1]
                    )
            if len(type2_pos) > 0 and len(type2_neg) > 0:
                type2_pos = np.hstack(type2_pos)
                type2_neg = np.hstack(type2_neg)
            misclas_traces = {}
            misclas_traces.update(
                {
                    target_key: {
                        "trace_pos": type2_pos,
                        "trace_neg": type2_neg,
                        "bae_type": bae_type,
                    }
                }
            )

            # APPEND NEW PD
            for key in misclas_traces.keys():
                for pos in misclas_traces[key]["trace_pos"]:
                    new_pd.append(
                        {
                            "err": err_labels[type_err]["pos"],
                            "task": key,
                            "unc": pos * 4,
                            "bae_type": misclas_traces[key]["bae_type"],
                        }
                    )
                for neg in misclas_traces[key]["trace_neg"]:
                    new_pd.append(
                        {
                            "err": err_labels[type_err]["neg"],
                            "task": key,
                            "unc": neg * 4,
                            "bae_type": misclas_traces[key]["bae_type"],
                        }
                    )
new_pd = pd.DataFrame(new_pd)
new_table = pd.DataFrame(new_table)

# plot boxplot
# group_order = ["TP+TN", "FP+FN"]
group_order = ["TP", "TN", "FP", "FN", "TP+TN", "FP+FN"]
# group_order = ["TP", "TN", "FP", "FN"]
# plt.figure()

# ax = sns.boxplot(
#     x="err",
#     y="unc",
#     hue="task",
#     data=new_pd,
#     order=group_order,
# )

# ax = sns.boxplot(
#     x="err",
#     y="unc",
#     hue="bae_type",
#     data=new_pd,
#     order=group_order,
# )

for task_i in new_pd["task"].unique():
    filtered_new_pd = new_pd[new_pd["task"] == task_i]
    plt.figure()
    ax = sns.boxplot(
        x="err",
        y="unc",
        hue="bae_type",
        data=filtered_new_pd,
        order=group_order,
    )
    ax.set_title(str(task_i))

task = "[13, 71]"
tptn = new_pd[
    (new_pd["err"] == "TP+TN")
    & (new_pd["bae_type"] == "mcd")
    & (new_pd["task"] == task)
]["unc"]
fpfn = new_pd[
    (new_pd["err"] == "FP+FN")
    & (new_pd["bae_type"] == "mcd")
    & (new_pd["task"] == task)
]["unc"]

print(calc_auroc(tptn, fpfn))


# best_perf = []
prefix_perf = "weighted-"
new_table = []
bae_types = ["ens"]
target_cols = [task]
for bae_type in bae_types:
    for target_key in target_cols:
        for unc_method_ in ["proba-epi", "proba-alea", "proba-total"]:
            filtered_res_retained = res_retained[
                (res_retained["bae_type"] == bae_type)
                & (res_retained["unc_method"] == unc_method_)
                & (res_retained[target_col] == target_key)
            ]
            filtered_res_retained = (
                filtered_res_retained.groupby(["norm", "dist"]).mean().reset_index()
            )
            # best_row = filtered_res_retained.iloc[
            #     filtered_res_retained[prefix_perf + perf_key].argmax()
            # ]
            best_row = filtered_res_retained.iloc[
                filtered_res_retained["weighted-" + perf_key].argmax()
            ]
            norm = best_row["norm"]
            dist = best_row["dist"]

            # res misclas
            res_misclas = pd.read_csv(base_folder + base_filename + "_misclas_perf.csv")

            filter_res_misclas = res_misclas[
                (res_misclas["bae_type"] == bae_type)
                & (res_misclas["full_likelihood"] == full_likelihood)
                & (res_misclas[target_col] == target_key)
                & (res_misclas["norm"] == norm)
                & (res_misclas["dist"] == dist)
                & (res_misclas["unc_method"] == unc_method_)
            ]

            # tabulate table of results
            filter_res_misclas_mean = (
                filter_res_misclas.groupby(["unc_method"]).mean().reset_index()
            )
            filter_res_misclas_mean["avgprc-ratio"] = (
                filter_res_misclas_mean["avgprc"] / filter_res_misclas_mean["baseline"]
            )

            temp_ = filter_res_misclas_mean[
                filter_res_misclas_mean["unc_method"] == unc_method_
            ]
            new_table.append(
                {
                    "bae_type": bae_type,
                    "unc_method": unc_method_,
                    "auroc": temp_["auroc"].item(),
                    "avgprc": temp_["avgprc"].item(),
                    "baseline": temp_["baseline"].item(),
                    "avgprc-ratio": temp_["avgprc-ratio"].item(),
                    target_col: target_key,
                    prefix_perf + perf_key: best_row[prefix_perf + perf_key],
                }
            )
new_table = pd.DataFrame(new_table)
