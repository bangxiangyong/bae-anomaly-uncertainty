import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


def generate_grid2d(X_train, span=1):
    grid = np.mgrid[
        X_train[:, 0].min() - span : X_train[:, 0].max() + span : 100j,
        X_train[:, 1].min() - span : X_train[:, 1].max() + span : 100j,
    ]
    grid_2d = grid.reshape(2, -1).T
    return grid_2d, grid


# plot figure
def plot_decision_boundary(
    x_inliers_train,
    grid_2d,
    Z,
    x_outliers_train=None,
    x_inliers_test=None,
    x_outliers_test=None,
    anomaly_threshold=None,
    fig=None,
    ax=None,
    figsize=(6, 4),
    cmap="Greys",
    colorbar=True,
    legend=True,
    legend_params={},
):
    grid = grid_2d.T.reshape(2, 100, 100)
    reshaped_Z = Z.reshape(100, 100)

    plt_leg = {"traces": [], "labels": []}

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    contour = ax.contourf(grid[0], grid[1], reshaped_Z, levels=35, cmap=cmap)

    if fig is not None and colorbar:
        fig.colorbar(contour)

    # plot decision boundary
    if anomaly_threshold is not None:
        a = ax.contour(
            grid[0],
            grid[1],
            reshaped_Z,
            levels=[anomaly_threshold],
            linewidths=1.5,
            colors="red",
        )
        ax.contourf(
            grid[0],
            grid[1],
            reshaped_Z,
            levels=[Z.min(), anomaly_threshold],
            colors="tab:blue",
            alpha=0.5,
        )

    inlier_train = ax.scatter(
        x_inliers_train[:, 0], x_inliers_train[:, 1], c="tab:green", s=20, edgecolor="k"
    )
    plt_leg["traces"].append(inlier_train)
    plt_leg["labels"].append("Inliers (Train)")

    if x_outliers_train is not None:
        outlier_train = ax.scatter(
            x_outliers_train[:, 0],
            x_outliers_train[:, 1],
            c="tab:orange",
            s=20,
            # edgecolor="k",
        )
        plt_leg["traces"].append(outlier_train)
        plt_leg["labels"].append("Anomalies (Train)")

    if x_inliers_test is not None:
        inlier_test = ax.scatter(
            x_inliers_test[:, 0],
            x_inliers_test[:, 1],
            c="tab:green",
            marker="x",
            s=20,
            # edgecolor="k",
        )
        plt_leg["traces"].append(inlier_test)
        plt_leg["labels"].append("Inliers (Test)")

    if x_outliers_test is not None:
        outlier_test = ax.scatter(
            x_outliers_test[:, 0],
            x_outliers_test[:, 1],
            c="tab:orange",
            marker="x",
            s=20,
            # edgecolor="k",
        )
        plt_leg["traces"].append(outlier_test)
        plt_leg["labels"].append("Anomalies (Test)")

    if anomaly_threshold is not None:
        plt_leg["traces"] += [a.collections[0]]
        plt_leg["labels"] += ["Decision Boundary"]

    if legend:
        ax.legend(plt_leg["traces"], plt_leg["labels"], **legend_params)


def get_anomaly_threshold(raw_train_scores, percentile=95):
    # higher means more anomalous
    anomaly_threshold = stats.scoreatpercentile(raw_train_scores, percentile)
    return anomaly_threshold


def get_hard_predictions(raw_train_scores, anomaly_threshold):
    hard_pred = np.zeros(len(raw_train_scores))
    hard_pred[np.argwhere(raw_train_scores >= anomaly_threshold)] = 1
    return hard_pred


# convert to percentiles
def convert_percentile(raw_train_scores, new_scores):
    pct = np.vectorize(lambda x: stats.percentileofscore(raw_train_scores, x))(
        new_scores
    )
    return pct


# def concat_params_res(dict_params, pd_res):
#     exp_row = pd.DataFrame([dict_params])
#     if not isinstance(pd_res, pd.DataFrame):
#         pd_res = pd.DataFrame([pd_res])
#
#     exp_row = pd.concat((exp_row, pd_res), axis=1)
#
#     return exp_row


def concat_params_res(*params):
    res = pd.DataFrame([{}])
    for i, param in enumerate(params):
        # convert to pandas Df
        if not isinstance(param, pd.DataFrame):
            param = pd.DataFrame([param])

        # append
        if i == 0:
            res = param
        else:
            res = pd.concat((res, param), axis=1)
    return res
