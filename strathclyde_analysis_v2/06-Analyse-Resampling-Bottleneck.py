import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

res_maps = {
    "resample": {
        "res_folder": "results/resampling",
        "x_key": "resample_factor",
        "csv_name": "STRATH_FORGE_RESAMPLING_AUROC.csv",
        "x_label": "Downsampling factor",
        "x_ticks": [
            r"x$1$",
            r"x$2$",
            r"x$5$",
            r"x$10$",
        ],
    },
    "latent": {
        "res_folder": "results/latent",
        "x_key": "latent_factor",
        "csv_name": "STRATH_FORGE_AUROC.csv",
        "x_label": "Latent dimensions",
        "x_ticks": [
            r"x$\frac{1}{4}$",
            r"x$\frac{1}{2}$",
            r"x$1$",
            r"x$2$",
        ],
    },
}

# Configure selection of results to be plotted
# res_mode = "latent"
res_mode = "resample"
savefig = True
# savefig = False

res_folder = res_maps[res_mode]["res_folder"]
x_key = res_maps[res_mode]["x_key"]
groupby_keys = ["ss_id", "skip", x_key, "bae_type"]


# === BASELINE AUROC ===
# auroc results
auroc_res = pd.read_csv(os.path.join(res_folder, res_maps[res_mode]["csv_name"]))

print(auroc_res.columns)

# ========================
legend_map = {"ae": "Det. AE", "ens": "BAE", "vae": "VAE"}
auroc_res_mean = auroc_res.groupby(groupby_keys).mean()
auroc_res_sem = auroc_res.groupby(groupby_keys).sem()

auroc_res_mean = auroc_res_mean.reset_index()
perf_label = "E_AUROC"

figsize = (9, 3.5)
fig, axes = plt.subplots(1, 3, sharey=True, figsize=figsize)

for ss_id, ax in zip(auroc_res_mean["ss_id"].unique(), axes):
    subset_ssid = auroc_res_mean[auroc_res_mean["ss_id"] == ss_id]
    labels = []

    for skip in [True, False]:
        for bae_type in auroc_res_mean["bae_type"].unique():
            subset = subset_ssid[
                (subset_ssid["skip"] == skip) & (subset_ssid["bae_type"] == bae_type)
            ]
            x_labels = np.arange(len(subset[x_key]))
            labels.append(
                legend_map[bae_type] + "+Skip" if skip else legend_map[bae_type]
            )
            ax.errorbar(
                x_labels,
                subset[perf_label],
                yerr=auroc_res_sem.iloc[subset.index][perf_label],
                fmt="o--" if skip else "o-",
                capsize=3,
                elinewidth=1,
                markeredgewidth=1,
            )
            ax.set_xticks(
                x_labels,
            )
            ax.set_xticklabels(res_maps[res_mode]["x_ticks"])
axes[0].set_ylabel("AUROC")
axes[1].set_xlabel(res_maps[res_mode]["x_label"])

axes[0].set_title("(a)")
axes[1].set_title("(b)")
axes[2].set_title("(c)")

fig.tight_layout()
plt.legend(labels, fontsize="small")

if savefig:
    fig.savefig("strath-bottleneck-" + res_mode + ".png", dpi=500)
