import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# res_folder = "results/latent"

ll_res_folder = "results/likelihood"
ll_group_keys = ["ss_id", "bae_type", "full_likelihood"]

# === BASELINE AUROC ===
# auroc results
auroc_res = pd.read_csv(os.path.join(ll_res_folder, "STRATH_FORGE_AUROC.csv"))

auroc_res_mean = auroc_res.groupby(ll_group_keys).mean()
auroc_res_std = auroc_res.groupby(ll_group_keys).std()

# bce vs se results
bce_v_se_res = pd.read_csv(os.path.join(ll_res_folder, "STRATH_FORGE_BCE_VS_SE.csv"))

bce_v_se_res_mean = bce_v_se_res.groupby(ll_group_keys).mean()
bce_v_se_res_std = bce_v_se_res.groupby(ll_group_keys).std()


# === Retained Perf. ===
# fixed_keys = {"full_likelihood": "mse", "ss_id": "13"}
# fixed_keys = {"full_likelihood": "mse", "ss_id": "13", "norm": False, "dist": "ecdf"}
# fixed_keys = {"full_likelihood": "mse", "ss_id": "71", "norm": False, "dist": "ecdf"}
fixed_keys = {
    "full_likelihood": "mse",
    # "full_likelihood": "hetero-gauss",
    "ss_id": "[13, 71]",
    # "ss_id": "71",
}

drop_keys = [
    {"bae_type": "ae", "unc_method": "varnll"},
    {"bae_type": "ae", "unc_method": "proba-epi"},
    {"bae_type": "ae", "unc_method": "proba-alea"},
    {"unc_method": "random"},
    # {"unc_method": "varnll"},
]
# drop_keys

retained_group_keys = ["bae_type", "dist", "norm", "unc_method"]
retained_res = pd.read_csv(
    os.path.join(ll_res_folder, "STRATH_FORGE_retained_perf.csv")
)

# filter to select the fixed keys
for key, val in fixed_keys.items():
    retained_res = retained_res[retained_res[key] == val]

# drop certain indices
def drop_selected_rows(df, drop_dict):
    final_indices = []
    for drop_ in drop_dict:
        new_df = df.copy()
        for key, val in drop_.items():
            new_df = new_df[new_df[key] == val]
        sel_rows = new_df.index
        final_indices.append(sel_rows)
    final_indices = [item for sublist in final_indices for item in sublist]
    new_df = df.drop(final_indices, axis=0)
    return new_df


retained_res = drop_selected_rows(retained_res, drop_keys)

retained_res_mean = retained_res.groupby(retained_group_keys).mean()
retained_res_std = retained_res.groupby(retained_group_keys).std()

# retained_res["diff-auroc"] = retained_res["baseline-auroc"]- retained_res["weighted-auroc"]
perf_key = "auroc"
# perf_key = "gss"
# perf_key = "f1_score"
base_key = "weighted-"
# base_key = "max-"
y_key = base_key + perf_key

filtered_res = retained_res_mean[[y_key, "baseline-" + perf_key]]
filtered_res = filtered_res.reset_index()

final_df = []
for i, row in enumerate(filtered_res.iterrows()):
    new_row = row[1].copy()
    new_col = new_row["unc_method"] + "-" + y_key

    new_row[new_col] = new_row[y_key]
    new_row = new_row.drop(["unc_method", y_key, "baseline-" + perf_key])
    if i == 0:
        final_df = new_row.copy()
    else:
        final_df = pd.concat((final_df, new_row), axis=1)
final_df = final_df.T

# === ROWS
# filtered_res["model_name"] = (
#     filtered_res["bae_type"]
#     + "-"
#     + filtered_res["dist"]
#     + "-"
#     + filtered_res["norm"].astype(str)
# )
# filtered_res["prob_dist"] = (
#     +filtered_res["dist"] + "-" + filtered_res["norm"].astype(str)
# )
#
# filtered_model_name = filtered_res["model_name"].unique()
#
# new_res = {}
# all_res = []
# unc_method_norm_pairs = {
#     "exceed": False,
#     "proba-alea": True,
#     "proba-epi": True,
#     "proba-total": True,
# }
#
# for base_model in filtered_model_name:
#     temp_ = base_model.split("-")
#     bae_type = temp_[0]
#     prob_dist = "-".join(temp_[1:])
#     new_res.update({"bae_type": bae_type})
#     new_res.update({"dist": temp_[1]})
#     new_res.update({"norm": temp_[2]})
#
#     new_res.update({"prob_dist": prob_dist})
#
#     filtered_pd = filtered_res[filtered_res["model_name"] == base_model]
#
#     baseline_perf = filtered_pd["baseline-" + perf_key].iloc[0]
#     for unc_method in filtered_pd["unc_method"].unique():
#         if unc_method in unc_method_norm_pairs.keys():
#             filtered_pd_ = filtered_pd[
#                 filtered_pd["norm"] == unc_method_norm_pairs[unc_method]
#             ]
#             unc_method_perf = filtered_pd_[filtered_pd_["unc_method"] == unc_method][
#                 y_key
#             ].item()
#
#             new_res.update({unc_method: unc_method_perf})
#             new_res.update({unc_method + "_diff": unc_method_perf - baseline_perf})
#
#     all_res.append(new_res.copy())
# all_res = pd.DataFrame(all_res)
#
# final_table_res = all_res[all_res["prob_dist"].isin(["ecdf-False", ""])]

# rr = retained_res_std[[y_key + perf_key, "baseline-" + perf_key]]

# ========
perf_key = "auroc"
# perf_key = "gss"
base_key = "weighted-"

filtered_pd = retained_res_mean.copy().reset_index()
filtered_pd["diff-" + perf_key] = (
    filtered_pd[y_key] - filtered_pd["baseline-" + perf_key]
)
best_perf = filtered_pd.groupby(["bae_type", "unc_method"]).max().reset_index()


y_key = base_key + perf_key

# filtered_res = best_perf[[y_key, "baseline-" + perf_key]]
# filtered_res = filtered_res.reset_index()
# filtered_res = best_perf
final_best_df = []
new_res = {}

for bae_type in best_perf["bae_type"].unique():
    new_res = {}
    new_res.update({"bae_type": bae_type})
    filtered_res = best_perf[best_perf["bae_type"] == bae_type].copy()
    for i, (_, row) in enumerate(filtered_res.iterrows()):
        new_res.update({row["unc_method"] + "-" + y_key: row[y_key]})
        new_res.update(
            {row["unc_method"] + "-" + "diff-" + y_key: row["diff-" + perf_key]}
        )
    final_best_df.append(new_res.copy())
final_best_df = pd.DataFrame(final_best_df)


#     new_row = row[1].copy()
#     new_col = new_row["unc_method"] + "-" + y_key
#
#     new_row[new_col] = new_row[y_key]
#     new_row = new_row.drop(["unc_method", y_key, "baseline-" + perf_key])
#     if i == 0:
#         final_df = new_row.copy()
#     else:
#         final_df = pd.concat((final_df, new_row), axis=1)
# final_df = final_df.T


# for unc_method in filtered_pd["unc_method"].unique():
#     for bae_type in filtered_pd["bae_type"].unique():
#         new_res = {}
#         if unc_method in unc_method_norm_pairs.keys():
#             print(unc_method)
#             print(bae_type)
#             new_res.update({"bae_type": bae_type})
#
#             filtered_pd_ = filtered_pd[
#                 filtered_pd["norm"] == unc_method_norm_pairs[unc_method]
#             ]
#             filtered_pd_ = filtered_pd_[filtered_pd_["bae_type"] == bae_type]
#             unc_method_perf = filtered_pd_[filtered_pd_["unc_method"] == unc_method][
#                 y_key
#             ].max()
#
#             new_res.update({unc_method: unc_method_perf})
#             new_res.update({unc_method + "_diff": unc_method_perf - baseline_perf})
#         all_res.append(new_res.copy())
# all_res = pd.DataFrame(all_res)

# .drop_duplicates()

# === =Actual Plotting===

retained_res_mean_reset = retained_res_mean.reset_index()
retained_res_mean_reset = retained_res_mean_reset[
    retained_res_mean_reset["unc_method"] != "varnll"
]

fig, axes = plt.subplots(1, 3)

for ax, dist in zip(axes, ["ecdf", "norm", "uniform"]):
    retained_dist = retained_res_mean_reset[retained_res_mean_reset["dist"] == dist]
    sns.boxplot(
        x="unc_method", y=y_key, hue="norm", data=retained_dist, palette="Set3", ax=ax
    )
    # sns.boxplot(
    #     x="norm", y=y_key, hue="unc_method", data=retained_dist, palette="Set3", ax=ax
    # )
    # Removed 'ax' from T.W.'s answer here aswell:
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
    #
    # # Put a legend to the right side
    # ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5), ncol=1)
    #
    ax.set_ylim(0.77, 0.91)
    ax.set_title(dist)

axes[0].get_legend().remove()
axes[1].get_legend().remove()

box = axes[-1].get_position()
axes[-1].set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position

# Put a legend to the right side
axes[-1].legend(loc="center right", bbox_to_anchor=(1.25, 0.5), ncol=1)

# axes[1].set_title("Effect of Outlier probability normalisation")
