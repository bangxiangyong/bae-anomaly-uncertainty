import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import sem

savefig = True
show_legend = True
# show_legend=False
# savefig = False

# dataset = "STRATH"
dataset = "ZEMA"

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
bae_type_map = {
    "ae": "Det. AE",
    "ens": "BAE, Ensemble",
    "mcd": "BAE, MC-Dropout",
    "vi": "BAE, BayesBB",
    "sghmc": "BAE, SGHMC",
    "vae": "VAE",
}
unc_map = {
    "proba-epi": "Epistemic",
    "proba-alea": "Aleatoric",
    "proba-total": "Total",
    "varnll": "Var(NLL)",
    "exceed": "Binomial CDF",
}
dist_map = {
    "uniform": "Uniform",
    "expon": "Exponen.",
    "norm": "Gaussian",
    "ecdf": "ECDF",
}
subtitle_labels = {
    "STRATH": [
        "(a) L-ACTpos",
        "(b) Feedback-SPA",
        "(c) L-ACTpos & Feedback-SPA",
    ],
    "ZEMA": [
        "(a) Cooler",
        "(b) Valve",
        "(c) Pump",
        "(d) Accumulator",
    ],
}

# arc_ylabel = {"gss": r"$W_{GSS}$", "f1_score": r"$W_{F1}$", "auroc": "AUROC"}
arc_ylabel = {"gss": r"$G_{SS}$", "f1_score": r"$F_{1}$", "auroc": "AUROC"}
arc_xlabel = "Rejection rate (%)"

# LOAD FROM PARAM MAP
base_folder = param_maps[dataset]["base_folder"]
base_filename = param_maps[dataset]["base_filename"]
target_col = param_maps[dataset]["target_col"]
latex_name = param_maps[dataset]["latex_name"]
fig_width = param_maps[dataset]["fig_width"]

# =========================MISCLASS DETECTION============================

res_misclas = pd.read_csv("results/STRATH_UNCOOD/STRATH_FORGE_UNCOOD_misclas_perf.csv")
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


def plot_retained_perf(
    res_retained,
    bae_type,
    norm_scaling,
    dist,
    unc_method="proba-total",
    perf_key="auroc",
    ax=None,
    pickle_folder="results/STRATH_UNCOOD/pickles/",
    max_xlim=1.0,
    dotted=True,
):
    # if unc_method != "random":
    # apply filter
    filter_res_retained = res_retained[
        (res_retained["bae_type"] == bae_type)
        & (res_retained["norm"] == norm_scaling)
        & (res_retained["dist"] == dist)
    ]
    # else:
    #     filter_res_retained = res_retained
    # load pickle
    pickle_files = filter_res_retained["pickle"].unique()
    all_max_percs = []
    all_valid_percs = []
    all_auroc = []
    all_interpolate_f = []

    for i, file in enumerate(pickle_files):
        pickle_dict = pickle.load(
            open(
                os.path.join(pickle_folder, file),
                "rb",
            )
        )
        valid_perc = 1 - np.array(pickle_dict[unc_method]["valid_perc"])
        auroc = pickle_dict[unc_method][perf_key]
        if len(valid_perc) > 1:
            all_valid_percs.append(valid_perc)
            all_auroc.append(auroc)
            all_max_percs.append(np.max(valid_perc))
            all_interpolate_f.append(interpolate.interp1d(valid_perc, auroc))

    interplates_mean = []
    interplates_sem = []
    valid_xi = []
    # for x_i in np.arange(0, 1.05, 0.025):
    for x_i in np.arange(0, max_xlim + 0.05, 0.05):
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
    markersize = 4.0
    alpha_line = 0.8
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if dotted and norm_scaling:
        ax.plot(
            valid_xi, interplates_mean, "v--", markersize=markersize, alpha=alpha_line
        )
    else:
        ax.plot(
            valid_xi, interplates_mean, "o-", markersize=markersize, alpha=alpha_line
        )

    ax.fill_between(
        valid_xi,
        np.clip(interplates_mean - interplates_sem, 0, 1),
        np.clip(interplates_mean + interplates_sem, 0, 1),
        alpha=0.25,
    )

    # config x ticks
    # plt.xticks(valid_xi, np.round(np.array(valid_xi) * 100, 2))
    xticks_ = np.arange(0, max_xlim + 0.1, 0.10)
    ax.set_xticks(xticks_)
    ax.set_xticklabels((xticks_ * 100).astype(int))
    ax.set_xlim(0, max_xlim)

    return (max_valid_prob, interplates_mean)


# LOAD RESULTS FOR CHOSEN DATASET
res_retained = pd.read_csv(base_folder + base_filename + "_retained_perf_AMENDED.csv")

# === PLOT FOR EACH MODEL ===


unc_method = "proba-total"
# perf_key = "gss"
perf_key = "f1_score"
# perf_key = "auroc"

max_rej_perc = 0.5

figsize = (fig_width, 2.25)
fig, axes = plt.subplots(
    1, len(res_retained[target_col].unique()), figsize=figsize, sharey=False
)
for ss_id, ax, subtitle in zip(
    res_retained[target_col].unique(), axes, subtitle_labels[dataset]
):
    legends = []
    global_max = []
    filter_res_retained = res_retained[res_retained[target_col] == ss_id]
    for bae_type in res_retained["bae_type"].unique():

        # filter best dist + norm scale
        best_dist_retained_ = filter_res_retained[
            filter_res_retained["bae_type"] == bae_type
        ]
        best_dist_retained_ = (
            best_dist_retained_.groupby(["dist", "norm"])
            .mean(0)["weighted-" + perf_key]
            .reset_index()
        )
        best_dist = np.argmax(best_dist_retained_["weighted-" + perf_key])
        best_dist_retained_ = best_dist_retained_.iloc[best_dist]

        norm_scaling = best_dist_retained_["norm"]
        dist = best_dist_retained_["dist"]
        print(best_dist_retained_)

        if bae_type != "jj":
            max_valid_prob, __ = plot_retained_perf(
                filter_res_retained,
                bae_type,
                norm_scaling,
                dist,
                unc_method=unc_method,
                perf_key=perf_key,
                ax=ax,
                pickle_folder=base_folder + "pickles/",
                max_xlim=0.5,
                dotted=False,
            )
            global_max.append(max_valid_prob)
            legends.append(bae_type)

    # max_valid_prob, __ = plot_retained_perf(
    #     filter_res_retained,
    #     # bae_type="ens",
    #     # norm_scaling=True,
    #     # dist="norm",
    #     unc_method="random",
    #     perf_key=perf_key,
    #     ax=ax,
    #     pickle_folder=base_folder + "pickles/",
    #     max_xlim=0.5,
    #     dotted=False,
    # )

    # ax.set_xlim(0, np.min(global_max))
    ax.set_title(subtitle, fontsize="small")
    ax.set_xlabel(arc_xlabel)
    # ax.set_aspect('equal', 'box')

# legends += ["Random"]
axes[0].set_ylabel(arc_ylabel[perf_key])

# set legends
if show_legend:
    legends = [bae_type_map[bae] for bae in legends]
    box = axes[-1].get_position()
    axes[-1].set_position([box.x0, box.y0, box.width * 0.98, box.height])
    axes[-1].legend(
        legends, fontsize="x-small", loc="center left", bbox_to_anchor=(1, 0.5)
    )

fig.tight_layout()
if savefig:
    fig_suffix = perf_key if not show_legend else perf_key + "_LG_"
    fig.savefig("arc_" + dataset + "_model_" + fig_suffix + ".png", dpi=500)

# ===PLOT FOR EACH DISTRIBUTIONS===


# unc_method = "proba-total"
# unc_method = "varnll"
# perf_key = "gss"
bae_type = "ens"


fig, axes = plt.subplots(
    1, len(res_retained[target_col].unique()), figsize=figsize, sharey=False
)
for ss_id, ax, subtitle in zip(
    res_retained[target_col].unique(), axes, subtitle_labels[dataset]
):
    legends = []
    global_max = []
    filter_res_retained = res_retained[res_retained[target_col] == ss_id]
    for dist in res_retained["dist"].unique():
        for norm_scaling in res_retained["norm"].unique():
            if bae_type != "ae":
                # plot
                max_valid_prob, _ = plot_retained_perf(
                    filter_res_retained,
                    bae_type,
                    norm_scaling,
                    dist,
                    unc_method=unc_method,
                    perf_key=perf_key,
                    ax=ax,
                    pickle_folder=base_folder + "pickles/",
                    max_xlim=0.5,
                    dotted=True,
                )
                global_max.append(max_valid_prob)

                # update label
                dist_label = dist_map[dist]
                if norm_scaling:
                    dist_label = dist_label + "+Norm"
                legends.append(dist_label)
    # plot
    max_valid_prob, mean_res = plot_retained_perf(
        filter_res_retained,
        bae_type,
        norm_scaling,
        dist,
        unc_method="varnll",
        perf_key=perf_key,
        ax=ax,
        pickle_folder=base_folder + "pickles/",
        max_xlim=0.5,
    )
    global_max.append(max_valid_prob)
    legends.append("Var(NLL)")

    # plot baseline
    # ax.set_xlim(0, np.min(global_max))
    # ax.set_xlim(0, max_rej_perc)
    ax.axhline(y=mean_res[0], xmin=0, xmax=10, color="black")
    legends.append("Baseline")
    ax.set_title(subtitle, fontsize="small")
    ax.set_xlabel(arc_xlabel)

axes[1].set_xlabel(arc_xlabel)
axes[0].set_ylabel(arc_ylabel[perf_key])

# set legend box position
if show_legend:
    box = axes[-1].get_position()
    axes[-1].set_position([box.x0, box.y0, box.width * 0.98, box.height])
    axes[-1].legend(
        legends, fontsize="x-small", loc="center left", bbox_to_anchor=(1, 0.5)
    )

# for ax in axes:
#     ax.set_aspect('equal', 'box')

fig.tight_layout()

if savefig:
    fig_suffix = perf_key if not show_legend else perf_key + "_LG_"
    fig.savefig("arc_" + dataset + "_dist_" + fig_suffix + ".png", dpi=500)

# === TABLE OF RESULTS ALL ===
res = pd.read_csv(base_folder + base_filename + "_retained_perf_AMENDED.csv")
table_res = []
selected_cols = [
    "bae_type",
    "unc_method",
    "weighted-auroc",
    "weighted-gss",
    "weighted-f1_score",
    "max-auroc",
    "max-gss",
    "max-f1_score",
    "baseline-auroc",
    "baseline-gss",
    "baseline-f1_score",
]

for uniq_target in np.unique(res[target_col]):
    filtered_targetcol = res[res[target_col] == uniq_target]
    filtered_targetcol = filtered_targetcol.groupby(
        ["unc_method", "dist", "norm", "bae_type"]
    )
    filtered_targetcol = filtered_targetcol.mean().reset_index()

    # select from best proba distribution
    for bae_type in filtered_targetcol["bae_type"].unique():
        for unc_method in ["proba-epi", "proba-alea", "proba-total", "varnll"]:
            if (
                not (bae_type == "ae" and unc_method == "proba-epi")
                and not (bae_type == "ae" and unc_method == "proba-total")
                and not (bae_type == "ae" and unc_method == "varnll")
            ):
                temp_rows = filtered_targetcol[
                    (filtered_targetcol["bae_type"] == bae_type)
                    & (filtered_targetcol["unc_method"] == unc_method)
                ].reset_index(drop=True)

                # select row with max weighted-gss
                max_row = np.argmax(temp_rows["weighted-gss"])

                temp_rows = temp_rows.iloc[max_row]
                new_temp = {col: temp_rows[col] for col in selected_cols}
                new_temp.update({"uniq_target": uniq_target})
                table_res.append(new_temp)


def get_best_proba_res_(res, uniq_target, target_col="ss_id", metric="gss"):
    table_res = []
    selected_cols = [
        "bae_type",
        "unc_method",
    ] + [prefix + metric for prefix in ["weighted-", "max-", "baseline-"]]

    filtered_targetcol = res[res[target_col] == uniq_target]
    filtered_targetcol = filtered_targetcol.groupby(
        ["unc_method", "dist", "norm", "bae_type"]
    )
    filtered_targetcol = filtered_targetcol.mean().reset_index()

    # select from best proba distribution
    for bae_type in filtered_targetcol["bae_type"].unique():
        for unc_method in [
            "proba-epi",
            "proba-alea",
            "proba-total",
            "varnll",
            "exceed",
        ]:
            if (
                not (bae_type == "ae" and unc_method == "proba-epi")
                and not (bae_type == "ae" and unc_method == "proba-total")
                and not (bae_type == "ae" and unc_method == "varnll")
            ):
                temp_rows = filtered_targetcol[
                    (filtered_targetcol["bae_type"] == bae_type)
                    & (filtered_targetcol["unc_method"] == unc_method)
                ].reset_index(drop=True)

                # select row with max weighted-gss
                max_row = np.argmax(temp_rows["weighted-" + metric])

                temp_rows = temp_rows.iloc[max_row]
                new_temp = {col: temp_rows[col] for col in selected_cols}
                new_temp.update({"uniq_target": uniq_target})
                table_res.append(new_temp)

    return pd.DataFrame(table_res)


def get_best_proba_res(res, uniq_target, target_col="ss_id", metrics=["gss"]):
    table_res_list = [
        get_best_proba_res_(res, uniq_target, target_col=target_col, metric=metric)
        for metric in metrics
    ]

    new_table = table_res_list[0].copy()

    for metric, table_res in zip(metrics, table_res_list):
        perf_cols = [prefix + metric for prefix in ["weighted-", "max-", "baseline-"]]
        for col in perf_cols:
            new_table[col] = table_res[col]

    return new_table


table_res_all = get_best_proba_res(
    res, uniq_target, target_col=target_col, metrics=["gss", "f1_score"]
)

# Convert to LATEX table row
main_metric = "weighted"
metric_1 = "gss"
metric_2 = "f1_score"


def reorder_metric_mean(table_res, uniq_target, metrics=["gss", "f1_score"]):
    selected_cols = ["bae_type", "unc_method",] + [
        prefix + metric
        for prefix in ["weighted-", "max-", "baseline-"]
        for metric in metrics
    ]

    auroc_sensor_mean = table_res[table_res["uniq_target"] == uniq_target][
        selected_cols
    ]
    # rearrange rows
    key_orders = [
        ["ae", "vae", "mcd", "vi", "ens", "sghmc"],
        ["proba-epi", "proba-alea", "proba-total", "varnll", "exceed"],
    ]

    reorder_auroc_mean = []
    for bae_type in key_orders[0]:
        for unc_method in key_orders[1]:
            reorder_auroc_mean.append(
                auroc_sensor_mean[
                    (auroc_sensor_mean["bae_type"] == bae_type)
                    & (auroc_sensor_mean["unc_method"] == unc_method)
                ]
            )
    reorder_auroc_mean = pd.concat(reorder_auroc_mean).reset_index()
    return reorder_auroc_mean


def latex_bold_val(x, apply_bold=False):
    formatted = "{:.1f}".format(x * 100)
    if apply_bold:
        formatted = "\\textbf{" + formatted + "}"
    return formatted


metrics = [metric_1, metric_2]
selected_cols = ["bae_type", "unc_method",] + [
    prefix + metric
    for prefix in ["weighted-", "max-", "baseline-"]
    for metric in metrics
]

for uniq_target in res[target_col].unique():
    table_res = get_best_proba_res(
        res, uniq_target, target_col=target_col, metrics=metrics
    )
    auroc_sensor_mean = table_res[table_res["uniq_target"] == uniq_target][
        selected_cols
    ]

    # rearrange rows
    key_orders = [
        ["ae", "vae", "mcd", "vi", "ens", "sghmc"],
        ["proba-epi", "proba-alea", "proba-total", "varnll", "exceed"],
    ]

    reorder_metrics_mean = []
    for bae_type in key_orders[0]:
        for unc_method in key_orders[1]:
            reorder_metrics_mean.append(
                auroc_sensor_mean[
                    (auroc_sensor_mean["bae_type"] == bae_type)
                    & (auroc_sensor_mean["unc_method"] == unc_method)
                ]
            )
    reorder_metrics_mean = pd.concat(reorder_metrics_mean).reset_index()

    # write to file
    latex_tb_f = open(
        "latex-" + latex_name + "-uncood-" + str(uniq_target) + ".txt", "w"
    )
    all_lines = ""
    for i, row in reorder_metrics_mean.round(3).iterrows():
        max_metric1_score = np.round(
            reorder_metrics_mean[reorder_metrics_mean["bae_type"] == row["bae_type"]][
                main_metric + "-" + metric_1
            ].max(),
            3,
        )
        max_metric2_score = np.round(
            reorder_metrics_mean[reorder_metrics_mean["bae_type"] == row["bae_type"]][
                main_metric + "-" + metric_2
            ].max(),
            3,
        )
        if row["bae_type"] != "ae":
            bold_metric1 = (
                True
                if max_metric1_score == row[main_metric + "-" + metric_1]
                else False
            )
            bold_metric2 = (
                True
                if max_metric2_score == row[main_metric + "-" + metric_2]
                else False
            )
        else:
            bold_metric1 = False
            bold_metric2 = False

        weighted_metric1 = latex_bold_val(
            row[main_metric + "-" + metric_1], apply_bold=bold_metric1
        )
        weighted_metric2 = latex_bold_val(
            row[main_metric + "-" + metric_2], apply_bold=bold_metric2
        )
        diff_metric1 = latex_bold_val(
            row[main_metric + "-" + metric_1] - row["baseline-" + metric_1],
            apply_bold=False,
        )
        diff_metric2 = latex_bold_val(
            row[main_metric + "-" + metric_2] - row["baseline-" + metric_2],
            apply_bold=False,
        )
        diff_metric1_symb = (
            (
                "(+"
                if float(
                    row[main_metric + "-" + metric_1] - row["baseline-" + metric_1]
                )
                >= 0
                else "("
            )
            + diff_metric1
            + ")"
        )
        diff_metric2_symb = (
            (
                "(+"
                if float(
                    row[main_metric + "-" + metric_2] - row["baseline-" + metric_2]
                )
                >= 0
                else "("
            )
            + diff_metric2
            + ")"
        )

        # check if bold is needed
        if bold_metric1:
            diff_metric1_symb = "\\textbf{" + diff_metric1_symb + "}"
        if bold_metric2:
            diff_metric2_symb = "\\textbf{" + diff_metric2_symb + "}"

        newline = (
            " & ".join(
                (
                    bae_type_map[row["bae_type"]],
                    unc_map[row["unc_method"]],
                    weighted_metric1 + diff_metric1_symb,
                    weighted_metric2 + diff_metric2_symb,
                )
            )
            + " \\\\"
        )
        # add a mid rule after each ae model rows
        # if (i + 1) % len(key_orders[1]) == 0:
        if i != (len(reorder_metrics_mean) - 1) and (
            reorder_metrics_mean.iloc[i]["bae_type"]
            != reorder_metrics_mean.iloc[i + 1]["bae_type"]
        ):
            newline += " \\midrule"
        elif i == (len(reorder_metrics_mean) - 1):
            newline += " \\bottomrule"
        newline += " \n"
        all_lines += newline
    latex_tb_f.write(all_lines)
    latex_tb_f.close()
