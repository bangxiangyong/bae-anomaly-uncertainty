import itertools

import torch
from pyod.utils.data import get_outliers_inliers, generate_data

# generate random data with two features
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4

from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from util.exp_manager import ExperimentManager
from util.generate_data import (
    generate_moons,
    generate_circles,
    generate_blobs,
    generate_aniso,
)
from util.helper import generate_grid2d, plot_decision_boundary
from sklearn.datasets import make_blobs


bae_set_seed(1)

train_samples = 100
X_train, Y_train, X_test, Y_test = generate_circles(
    train_only=False, n_samples=train_samples, test_size=0.5, outlier_class=1
)

# X_train, Y_train, X_test, Y_test = generate_aniso(train_only=False,
#                                                   n_samples=500,
#                                                   test_size=0.5,
#                                                   outlier_class = 1
#                                                   )

X_train, Y_train, X_test, Y_test = generate_blobs(
    train_only=False, n_samples=train_samples, test_size=0.5, outlier_class=1
)

offset = 5
X_train = X_train + offset
X_test = X_test + offset
# X_train, Y_train, X_test, Y_test = generate_moons(train_only=False,
#                                                   n_samples=500,
#                                                   test_size=0.5,
#                                                   outlier_class = 1
#                                                   )

# by default the outlier fraction is 0.1 in generate data function
outlier_fraction = 0.01

# store outliers and inliers in different numpy arrays
x_outliers_train, x_inliers_train = get_outliers_inliers(X_train, Y_train)
x_outliers_test, x_inliers_test = get_outliers_inliers(X_test, Y_test)

# separate the two features and use it to plot the data
# F1 = X_train[:,[0]].reshape(-1,1)
# F2 = X_train[:,[1]].reshape(-1,1)

# create a meshgrid
# xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
#
# # scatter plot
# plt.scatter(F1,F2)
# plt.xlabel('F1')
# plt.ylabel('F2')

# ====================BAE===========================


local_exp_path = "experiments/"
exp_man = ExperimentManager(folder_name=local_exp_path)

# grid = {
#     "skip": [True, False],
#     "overcomplete": [True, False],
#     "bae_type": ["ae", "ens", "vae"],
#     "dataset": ["blob", "circle", "moon"],
# }

# AE
# grid = {
#     "skip": [True, False],
#     "overcomplete": [True, False],
#     "bae_type": ["ae"],
#     "dataset": ["blob", "circle", "moon"],
# }

# BAE & VAE
# grid = {
#     "skip": [True],
#     "overcomplete": [True],
#     "bae_type": ["ens", "vae"],
#     "dataset": ["blob", "circle", "moon"],
# }

grid = {
    "skip": [True],
    "overcomplete": [True],
    "bae_type": ["ae"],
    "dataset": ["circle"],
}

dataset_map = {
    "blob": generate_blobs,
    "circle": generate_circles,
    "moon": generate_moons,
}
bae_type_classes = {
    "ens": BAE_Ensemble,
    "vae": VAE,
    "ae": BAE_Ensemble,
}
n_bae_samples_map = {
    "ens": 5,
    "mcd": 100,
    "sghmc": 100,
    "vi": 100,
    "vae": 100,
    "ae": 1,
}

# Loop over all grid search combinations
for rep, values in enumerate(tqdm(itertools.product(*grid.values()))):
    bae_set_seed(1)
    # setup the grid
    exp_params = dict(zip(grid.keys(), values))

    # unpack exp params
    skip = exp_params["skip"]
    overcomplete = exp_params["overcomplete"]
    bae_type = exp_params["bae_type"]
    dataset = exp_params["dataset"]
    generate_dataset = dataset_map[dataset]

    # prepare data
    input_dim = x_inliers_train.shape[1]
    X_train, Y_train, X_test, Y_test = generate_dataset(
        train_only=False,
        n_samples=train_samples,
        test_size=0.35,
        outlier_class=2 if dataset == "blob" else 1,
    )
    x_outliers_train, x_inliers_train = get_outliers_inliers(X_train, Y_train)
    x_outliers_test, x_inliers_test = get_outliers_inliers(X_test, Y_test)

    # prepare model
    chain_params = [
        {
            "base": "linear",
            # "architecture": [input_dim, input_dim * 2, input_dim * 4],
            "architecture": [input_dim, 50, 50, 50, 100 if overcomplete else 1],
            "activation": "selu",
            "norm": "none",
            "bias": False,
        }
    ]

    lin_autoencoder = bae_type_classes[bae_type](
        chain_params=chain_params,
        last_activation="sigmoid",
        last_norm="none",
        # twin_output=True,
        # twin_params={"activation": "none", "norm": False},
        # skip=False,
        skip=skip,
        # use_cuda=True,
        # scaler_enabled=True,
        learning_rate=0.0001,
        num_samples=n_bae_samples_map[bae_type],
        likelihood="gaussian",
        homoscedestic_mode="none",
        # weight_decay=0.0000000001,
        use_cuda=True,
    )

    scaler = MinMaxScaler(clip=False)
    x_train_scaled = scaler.fit_transform(x_inliers_train)

    # === FIT AE ===
    # lin_autoencoder.fit(x_train_scaled, num_epochs=3500)

    # run lr_range_finder
    train_dataloader = convert_dataloader(
        x_train_scaled, batch_size=len(x_train_scaled) // 5, drop_last=False
    )

    min_lr, max_lr, half_iter = run_auto_lr_range_v4(
        train_dataloader,
        lin_autoencoder,
        window_size=3,
        num_epochs=10,
        run_full=False,
        save_mecha="copy",
    )
    lin_autoencoder.fit(train_dataloader, num_epochs=1000)

    # predict model and visualise grid
    nll_pred_train = lin_autoencoder.predict(x_train_scaled, select_keys=["nll"])["nll"]
    nll_pred_train_mean = nll_pred_train.mean(-1).mean(0)
    nll_pred_train_var = nll_pred_train.mean(-1).var(0)

    # predict grid2d
    grid_2d, grid_map = generate_grid2d(np.concatenate((X_train, X_test)), span=0.5)

    nll_pred_grid = lin_autoencoder.predict(
        scaler.transform(grid_2d), select_keys=["nll"]
    )["nll"]
    nll_pred_grid_mean = nll_pred_grid.mean(-1).mean(0)
    nll_pred_grid_var = nll_pred_grid.mean(-1).var(0)

    # save outputs
    output_dict = {
        "x_inliers_train": x_inliers_train,
        "x_inliers_test": x_inliers_test,
        "x_outliers_test": x_outliers_test,
        "grid_2d": grid_2d,
        "nll_pred_grid_mean": nll_pred_grid_mean,
        "nll_pred_grid_var": nll_pred_grid_var,
        "nll_pred_train_mean": nll_pred_train_mean,
        "nll_pred_train_var": nll_pred_train_var,
    }

    exp_man.encode_pickle(exp_params, output_dict)
    exp_man.update_csv(
        exp_params, insert_pickle=True, csv_name="toy_data_bottleneck.csv"
    )

    plot_decision_boundary(
        x_inliers_train=x_inliers_train,
        x_inliers_test=x_inliers_test,
        x_outliers_test=x_outliers_test,
        grid_2d=grid_2d,
        Z=np.log(nll_pred_grid_mean),
        # anomaly_threshold=np.percentile(np.log(nll_pred_train_mean), 90),
    )

# plot_decision_boundary(x_inliers_train=x_inliers_train,
#                        x_outliers_train=x_outliers_train,
#                        x_inliers_test=x_inliers_test,
#                        x_outliers_test=x_outliers_test,
#                        grid_2d=grid_2d,
#                        Z=nll_pred_grid_var)

#
# # train mu network
# train_loader = convert_dataloader(X_train_scaled, batch_size=150, shuffle=True)
# bae_ensemble.fit(train_loader,num_epochs=500)
#
#
# train_nll = bae_ensemble.predict_samples(X_train_scaled, select_keys=["se"])
#
# # get raw train scores
# raw_train_scores = train_nll.mean(0)[0].mean(-1)
# raw_train_scores = np.exp(raw_train_scores)
# raw_train_scores_perc = convert_percentile(raw_train_scores, raw_train_scores) # convert to pct
#
# # get threshold
# anomaly_threshold = stats.scoreatpercentile(raw_train_scores_perc,100-(100*outlier_fraction))
#
# # apply threshold to get hard predictions
# hard_pred = get_hard_predictions(raw_train_scores_perc, anomaly_threshold)
#
# # visualise grid
# grid_2d, grid = generate_grid2d(full_X, span=1)
# y_pred_grid = bae_ensemble.predict_samples(scaler.transform(grid_2d), select_keys=["se"]).mean(0)[0].mean(-1)
# y_pred_grid = np.exp(y_pred_grid)
# y_pred_grid = convert_percentile(raw_train_scores, y_pred_grid) # convert to pct
#
# plot_decision_boundary(x_inliers_train=x_inliers_train,
#                        x_outliers_train=x_outliers_train,
#                        x_inliers_test=x_inliers_test,
#                        x_outliers_test=x_outliers_test,
#                        grid_2d=grid_2d,
#                        Z=y_pred_grid,
#                        anomaly_threshold=anomaly_threshold)
#
# plot_decision_boundary(x_inliers_train=x_inliers_train,
#                        x_outliers_train=x_outliers_train,
#                        x_inliers_test=x_inliers_test,
#                        x_outliers_test=x_outliers_test,
#                        grid_2d=grid_2d,
#                        Z=y_pred_grid)
#
#

# plot percentile conversion (ECDF)
# plt.figure()
# plt.scatter(raw_train_scores, convert_percentile(raw_train_scores, raw_train_scores))
# plt.xlabel("Raw scores")
# plt.ylabel("ECDF")
#
# # evaluation of performance
# y_nll_test = bae_ensemble.predict_samples(scaler.transform(X_test), select_keys=["se"]).mean(0).mean(-1)[0]
# y_nll_test = np.exp(y_nll_test)
# y_nll_test_perc = convert_percentile(raw_train_scores, y_nll_test) # convert to pct
#
# hard_pred_test = get_hard_predictions(y_nll_test_perc, anomaly_threshold)
#
# # AUROC & MCC
# fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_nll_test, pos_label=1)
# auroc_test = metrics.auc(fpr, tpr)
# mcc_test = matthews_corrcoef(Y_test, hard_pred_test)
# prec_n_test = precision_n_scores(Y_test, y_nll_test)
# print("AUROC: {:.2f}".format(auroc_test))
# print("MCC: {:.2f}".format(mcc_test))
# print("PREC@N: {:.2f}".format(prec_n_test))
#
#
# def hard_predict(raw_train_scores, y_nll_test, outlier_fraction=0.1):
#     y_nll_test_perc = convert_percentile(raw_train_scores, y_nll_test)  # convert to pct
#     raw_train_scores_perc = convert_percentile(raw_train_scores, raw_train_scores)  # convert to pct
#     anomaly_threshold = stats.scoreatpercentile(raw_train_scores_perc, 100 - (100 * outlier_fraction))
#
#     hard_pred_test = get_hard_predictions(y_nll_test_perc, anomaly_threshold)
#
#     return hard_pred_test
#
#
# y_nll_grid2d = np.exp(bae_ensemble.predict_samples(scaler.transform(grid_2d), select_keys=["se"]).mean(-1)[:,0])
# raw_train_scores = np.exp(train_nll.mean(-1)[:,0])
#
# hard_pred_grid2d_samples = np.array([hard_predict(raw_train_scores_i, y_nll_test_i,
#                                                 outlier_fraction=outlier_fraction)
#                                    for y_nll_test_i,raw_train_scores_i in zip(y_nll_grid2d,raw_train_scores)])
# # hard_pred_grid2d_samples = np.array([convert_percentile(raw_train_scores_i, y_nll_test_i)
# #                                      for raw_train_scores_i,y_nll_test_i in zip(raw_train_scores,y_nll_grid2d)])
#
# uncertainty_grid2d = hard_pred_grid2d_samples.std(0)
# # uncertainty_grid2d = np.percentile(hard_pred_grid2d_samples, 50,axis=0)*(1-np.percentile(hard_pred_grid2d_samples, 50,axis=0))
# # uncertainty_grid2d = np.abs(np.percentile(hard_pred_grid2d_samples, 75,axis=0)-np.percentile(hard_pred_grid2d_samples, 25,axis=0))
# # uncertainty_grid2d = np.percentile(hard_pred_grid2d_samples, 75,axis=0)
#
# # plot uncertainty
# plot_decision_boundary(x_inliers_train=x_inliers_train,
#                        x_outliers_train=x_outliers_train,
#                        x_inliers_test=x_inliers_test,
#                        x_outliers_test=x_outliers_test,
#                        grid_2d=grid_2d,
#                        Z=uncertainty_grid2d,
#                        cmap="Greys"
#                        )
#
# # filter on uncertainty
# y_nll_test = np.exp(bae_ensemble.predict_samples(scaler.transform(X_test), select_keys=["se"]).mean(-1)[:,0])
# raw_train_scores = np.exp(train_nll.mean(-1)[:,0])
#
# hard_pred_test_samples = np.array([hard_predict(raw_train_scores_i, y_nll_test_i,
#                                                 outlier_fraction=outlier_fraction)
#                                    for y_nll_test_i,raw_train_scores_i in zip(y_nll_test,raw_train_scores)])
# uncertainty_test = hard_pred_test_samples.std(0)
#
# prec_n_test_unc = []
# ulim = uncertainty_test.max()-0.01
# unc_range = np.linspace(uncertainty_test.min(),ulim,100)
# for unc_lim in np.linspace(uncertainty_test.min(),ulim,100):
#     unc_arg = np.argwhere(uncertainty_test>=unc_lim).flatten()
#     prec_n_test_unc += [precision_n_scores(Y_test[unc_arg], y_nll_test.mean(0)[unc_arg])]
#
# plt.figure()
# plt.plot(unc_range,prec_n_test_unc)
#
#
#
