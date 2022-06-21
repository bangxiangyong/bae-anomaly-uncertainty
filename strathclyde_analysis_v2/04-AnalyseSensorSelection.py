import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle


def sanitize_sensor_names(column_names):
    sensors = [
        sensor.split("[")[0].strip().replace("_", "-").replace(" ", "-")
        for sensor in column_names
    ]
    return sensors


ll_res_folder = "results/sensors"
ll_group_keys = ["ss_id", "skip"]

pickle_path = "pickles"
column_names = sanitize_sensor_names(
    pickle.load(open(pickle_path + "/" + "column_names.p", "rb"))
)


# === AUROC USING ONE SENSOR AT TIME===
figsize = (7, 3.1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
for bae_type, ax in zip(["AE", "BAE"], (ax1, ax2)):
    auroc_res = pd.read_csv(
        os.path.join(ll_res_folder, "STRATH_FORGE_SENSORS_" + bae_type + "_AUROC.csv")
    )
    auroc_res = auroc_res[auroc_res["ss_id"] != "[2, 25, 11, 13, 54, 71, 82]"]
    auroc_res["ss_name"] = [
        column_names[int(ss_id)] for ss_id in auroc_res["ss_id"].values
    ]

    bplot = sns.barplot(
        x="ss_name",
        hue="skip",
        y="E_AUROC",
        data=auroc_res,
        capsize=0.2,
        ax=ax,
        errwidth=1.5,
        ci=95,
    )
    ax.axhline(y=0.8, color="black", linestyle="--")
    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        horizontalalignment="right",
        fontsize="x-small",
    )

plt.ylim(0.5, 1.0)
plt.yticks(np.arange(0.5, 1.0, 0.05).round(2))

ax1.set_title("(a) Deterministic AE")
ax2.set_title("(b) BAE")

# set legend
ax1.legend([], [], frameon=False)
h, l = ax2.get_legend_handles_labels()
ax2.legend(
    h,
    ["Without Skip", "With Skip"],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize="small",
)

# set x & y label
ax1.set_ylabel("AUROC")
ax2.set_ylabel("")

ax1.set_xlabel("")
ax2.set_xlabel("")

fig.tight_layout()

# plt.savefig("sensor-selection.png", dpi=500)
