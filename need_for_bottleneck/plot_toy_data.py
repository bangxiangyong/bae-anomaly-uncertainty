import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# load csv
from util.helper import plot_decision_boundary

experiment_folder = "experiments/"
model_info = pd.read_csv(experiment_folder + "toy_data_bottleneck.csv")

model_info["model_name"] = (
    model_info["bae_type"]
    + "-"
    + model_info["skip"].astype(str)
    + "-"
    + model_info["overcomplete"].astype(str)
)

label_map = {
    "ae-True-True": "Overcomplete AE+Skip",
    "ae-True-False": "Undercomplete AE+Skip",
    "ae-False-True": "Overcomplete AE",
    "ae-False-False": "Undercomplete AE",
    "ens-True-True": "Overcomplete BAE+Skip",
    "vae-True-True": "Overcomplete VAE+Skip",
}

model_plot_order = [
    "ae-False-False",
    "ae-True-False",
    "ae-False-True",
    "ae-True-True",
    "ens-True-True",
    "vae-True-True",
]
data_plot_order = [
    "blob",
    "circle",
    "moon",
]

# create subplots
# plotsize_factor = 3
figsize = (12, 6)
fig, axes = plt.subplots(len(data_plot_order), len(model_plot_order), figsize=figsize)
text_size = "small"
for col_i, model in enumerate(model_plot_order):
    for row_i, dataset in enumerate(data_plot_order):
        row = model_info[
            (model_info["model_name"] == model) & (model_info["dataset"] == dataset)
        ]

        if len(row) > 0:
            # load output dict
            output_dict = pickle.load(
                open(experiment_folder + "pickles/" + row["pickle"].item(), "rb")
            )

            if row_i == 0 and col_i == 0:
                plt_legend = True
            else:
                plt_legend = False

            # plot  decision boundary
            plot_decision_boundary(
                x_inliers_train=output_dict["x_inliers_train"],
                x_inliers_test=output_dict["x_inliers_test"],
                x_outliers_test=output_dict["x_outliers_test"],
                grid_2d=output_dict["grid_2d"],
                Z=np.log(output_dict["nll_pred_grid_mean"]),
                ax=axes[row_i][col_i],
                legend=plt_legend,
                legend_params={"fontsize": "x-small", "loc": "lower right"},
            )
            axes[row_i][col_i].set_xticks(())
            axes[row_i][col_i].set_yticks(())

            if row_i == 0:
                axes[row_i][col_i].set_title(
                    label_map[row["model_name"].item()], fontsize=text_size
                )
fig.tight_layout()
fig.savefig("2d-toy-bottleneck.png", dpi=500)
