from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
import os
import numpy as np
from scipy.stats import spearmanr

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import (
    Encoder,
    infer_decoder,
    Autoencoder,
)
from baetorch.baetorch.models.base_layer import DenseLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood.calc_uncertainty_ood import (
    calc_ood_threshold,
    convert_hard_predictions,
)
from util.convergence import (
    bae_fit_convergence,
    plot_convergence,
    bae_semi_fit_convergence,
)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.metrics import matthews_corrcoef
from util.evaluate_ood import plot_histogram_ood, plot_kde_ood
from statsmodels.stats.stattools import medcouple


def calc_auroc(nll_inliers_train_mean, nll_inliers_valid_mean):
    y_true = np.concatenate(
        (
            np.zeros(nll_inliers_train_mean.shape[0]),
            np.ones(nll_inliers_valid_mean.shape[0]),
        )
    )
    y_scores = np.concatenate((nll_inliers_train_mean, nll_inliers_valid_mean))

    auroc = roc_auc_score(y_true, y_scores)
    return auroc


def calc_auroc_valid(bae_ensemble, x_inliers_train, x_inliers_valid):

    nll_inliers_train = bae_ensemble.predict_samples(
        x_inliers_train, select_keys=["se"]
    )
    nll_inliers_valid = bae_ensemble.predict_samples(
        x_inliers_valid, select_keys=["se"]
    )

    nll_inliers_train_mean = nll_inliers_train.mean(0)[0].mean(-1)
    nll_inliers_valid_mean = nll_inliers_valid.mean(0)[0].mean(-1)

    y_true = np.concatenate(
        (
            np.zeros(nll_inliers_train_mean.shape[0]),
            np.ones(nll_inliers_valid_mean.shape[0]),
        )
    )
    y_scores = np.concatenate((nll_inliers_train_mean, nll_inliers_valid_mean))

    auroc = roc_auc_score(y_true, y_scores)

    return auroc


# random_seed = 987
random_seed = 1233333

bae_set_seed(random_seed)

use_cuda = True
clip_data_01 = True
activation = "leakyrelu"
last_activation = "sigmoid"  # tanh
likelihood = "gaussian"  # gaussian
train_size = 0.80
num_samples = 10
# multi_perc_thresholds = np.arange(85,100)
# multi_perc_thresholds = np.arange(75,100)
# multi_perc_thresholds = np.arange(90,100)
# multi_perc_thresholds = np.arange(70,80)
multi_perc_thresholds = np.arange(90, 100, 0.5)
# multi_perc_thresholds = np.arange(90,100,0.05)
# multi_perc_thresholds = np.arange(95,96)
# multi_perc_thresholds = np.arange(90,100)
perc_threshold = 95
# semi_supervised = True
semi_supervised = False
num_train_outliers = 5

# ==============PREPARE DATA==========================
base_folder = "od_benchmark"
mat_file_list = os.listdir(base_folder)
mat_file = mat_file_list[5]
mat = loadmat(os.path.join(base_folder, mat_file))

X = mat["X"]
y = mat["y"].ravel()

x_outliers, x_inliers = get_outliers_inliers(X, y)

if semi_supervised:
    x_outliers_train, x_outliers_test = train_test_split(
        x_outliers,
        train_size=np.round(num_train_outliers / len(x_outliers), 2),
        shuffle=True,
        random_state=random_seed,
    )
else:
    x_outliers_train = x_outliers.copy()
    x_outliers_test = x_outliers.copy()

# x_outliers_test = x_outliers[4:]
# x_outliers_train = x_outliers[:4]

x_inliers_train, x_inliers_test = train_test_split(
    x_inliers, train_size=train_size, shuffle=True, random_state=random_seed
)
x_inliers_train, x_inliers_valid = train_test_split(
    x_inliers_train, train_size=train_size, shuffle=True, random_state=random_seed
)


# =================SCALER=========================
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()

scaler = scaler.fit(x_inliers_train)
x_inliers_train = scaler.transform(x_inliers_train)
x_inliers_valid = scaler.transform(x_inliers_valid)
x_inliers_test = scaler.transform(x_inliers_test)
x_outliers_test = scaler.transform(x_outliers_test)
x_outliers_train = scaler.transform(x_outliers_train)

if clip_data_01:
    x_inliers_train = np.clip(x_inliers_train, 0, 1)
    x_inliers_test = np.clip(x_inliers_test, 0, 1)
    x_outliers_test = np.clip(x_outliers_test, 0, 1)

# =================DEFINE BAE========================

input_dim = x_inliers_train.shape[-1]
# activation = "sigmoid"
weight_decay = 0.0001
lr = 0.025
anchored = False

# encoder_nodes = [input_dim*2,20]
# encoder_nodes = [input_dim*4,int(input_dim/3)]
# encoder_nodes = [input_dim*4,100]
# encoder_nodes = [input_dim*4,80]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*8,input_dim]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*6]
# encoder_nodes = [input_dim*2,input_dim*4,2]
# encoder_nodes = [input_dim*8,input_dim]
# encoder_nodes = [input_dim*6,input_dim*4,input_dim*8]
# encoder_nodes = [input_dim*6,input_dim*4]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*6]
# encoder_nodes = [input_dim*4,input_dim*4,int(input_dim/3)]
# encoder_nodes = [input_dim*8,input_dim*4,int(input_dim*5)]
# encoder_nodes = [input_dim*8,int(input_dim/3)]
# encoder_nodes = [input_dim*8,input_dim*4,int(input_dim/2)]
encoder_nodes = [input_dim * 2, input_dim * 4, input_dim * 8]

encoder = Encoder(
    [
        DenseLayers(
            input_size=input_dim,
            architecture=encoder_nodes[:-1],
            output_size=encoder_nodes[-1],
            activation=activation,
            last_activation=activation,
        )
    ]
)

# specify decoder-mu
decoder_mu = infer_decoder(
    encoder, activation=activation, last_activation=last_activation
)  # symmetrical to encoder

# combine them into autoencoder
autoencoder = Autoencoder(encoder, decoder_mu)

# convert into BAE-Ensemble
bae_ensemble = BAE_Ensemble(
    autoencoder=autoencoder,
    anchored=anchored,
    weight_decay=weight_decay,
    num_samples=num_samples,
    likelihood=likelihood,
    learning_rate=lr,
    verbose=False,
    use_cuda=use_cuda,
)

# ===============FIT BAE CONVERGENCE===========================
train_loader = convert_dataloader(x_inliers_train, batch_size=750, shuffle=True)
run_auto_lr_range_v2(
    train_loader, bae_ensemble, run_full=True, window_size=1, num_epochs=15
)
# bae_ensemble.fit(train_loader,num_epochs=800)

num_epochs_per_cycle = 50
fast_window = num_epochs_per_cycle
slow_window = fast_window * 15
n_stop_points = 10
cvg = 0

auroc_valids = []
auroc_threshold = 0.60

while cvg == 0:
    # while(len(bae_ensemble.losses)<=num_epochs_per_cycle*n_stop_points):
    #     bae_ensemble.fit(train_loader, num_epochs=num_epochs_per_cycle)
    #     if not semi_supervised:
    #         _, cvg = bae_fit_convergence(bae_ensemble=bae_ensemble, x=train_loader,
    #                                      num_epoch=num_epochs_per_cycle,
    #                                      fast_window=fast_window,
    #                                      slow_window=slow_window,
    #                                      n_stop_points=n_stop_points
    #                                      )

    # else:
    #     _, cvg = bae_fit_convergence(bae_ensemble=bae_ensemble, x=train_loader,
    #                                  num_epoch=num_epochs_per_cycle,
    #                                  fast_window=fast_window,
    #                                  slow_window=slow_window,
    #                                  n_stop_points=n_stop_points
    #                                  )
    # bae_ensemble.fit(x_outliers_train, num_epochs=1, inverse=True)

    # bae_ensemble.semisupervised_fit(x_inliers=train_loader,
    #                                 x_outliers=x_outliers_train,
    #                                 num_epochs=5)

    # _, cvg = bae_semi_fit_convergence(bae_ensemble=bae_ensemble,
    #                                   x=train_loader,
    #                                   x_outliers=x_outliers_train,
    #                              num_epoch=num_epochs_per_cycle,
    #                              fast_window=fast_window,
    #                              slow_window=slow_window,
    #                              n_stop_points=n_stop_points
    #                              )

    _, cvg = bae_fit_convergence(
        bae_ensemble=bae_ensemble,
        x=train_loader,
        num_epoch=num_epochs_per_cycle,
        fast_window=fast_window,
        slow_window=slow_window,
        n_stop_points=n_stop_points,
    )

    if semi_supervised:
        bae_ensemble.semisupervised_fit(
            x_inliers=train_loader,
            x_outliers=x_outliers_train,
            num_epochs=int(num_epochs_per_cycle / 2),
        )

    auroc_valid = calc_auroc_valid(bae_ensemble, x_inliers_train, x_inliers_valid)
    auroc_valids.append(auroc_valid)
    print("AUROC-VALID: {:.3f}".format(auroc_valid))
    print("AUROC-VALID: {:.3f}".format(auroc_threshold))

    auroc_ood = calc_auroc_valid(bae_ensemble, x_inliers_test, x_outliers_test)
    print("AUROC-OOD: {:.3f}".format(auroc_ood))
    if auroc_valid >= auroc_threshold:
        break


fig, ax = plt.subplots(1, 1)
plot_convergence(
    losses=bae_ensemble.losses,
    fast_window=fast_window,
    slow_window=slow_window,
    n_stop_points=n_stop_points,
    ax=ax,
)

# ===============PREDICT BAE==========================

nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=["se"])
nll_inliers_test = bae_ensemble.predict_samples(x_inliers_test, select_keys=["se"])
nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["se"])
nll_outliers_test = bae_ensemble.predict_samples(x_outliers_test, select_keys=["se"])
nll_outliers_train = bae_ensemble.predict_samples(x_outliers_train, select_keys=["se"])

nll_inliers_train_mean = nll_inliers_train.mean(0)[0].mean(-1)
nll_inliers_test_mean = nll_inliers_test.mean(0)[0].mean(-1)
nll_inliers_valid_mean = nll_inliers_valid.mean(0)[0].mean(-1)
nll_outliers_test_mean = nll_outliers_test.mean(0)[0].mean(-1)
nll_outliers_train_mean = nll_outliers_train.mean(0)[0].mean(-1)

nll_inliers_train_var = nll_inliers_train.var(0)[0].mean(-1)
nll_inliers_test_var = nll_inliers_test.var(0)[0].mean(-1)
nll_outliers_test_var = nll_outliers_test.var(0)[0].mean(-1)
nll_outliers_train_var = nll_outliers_train.var(0)[0].mean(-1)

# nll_inliers_train_var = nll_inliers_train.mean(-1).var(0)[0]
# nll_inliers_test_var = nll_inliers_test.mean(-1).var(0)[0]
# nll_outliers_test_var = nll_outliers_test.mean(-1).var(0)[0]
# nll_outliers_train_var = nll_outliers_train.mean(-1).var(0)[0]
#


def plot_kde_auroc(
    nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean, mode="kde"
):
    y_true = np.concatenate(
        (
            np.zeros(nll_inliers_test_mean.shape[0]),
            np.ones(nll_outliers_test_mean.shape[0]),
        )
    )
    y_scores = np.concatenate((nll_inliers_test_mean, nll_outliers_test_mean))

    auroc = roc_auc_score(y_true, y_scores)
    auroc_text = "AUROC: {:.3f}".format(auroc, 2)

    fig, ax = plt.subplots(1, 1)

    if mode == "kde":
        plot_kde_ood(
            nll_inliers_train_mean,
            nll_inliers_test_mean,
            nll_outliers_test_mean,
            fig=fig,
            ax=ax,
        )

    if mode == "hist":
        plot_histogram_ood(
            nll_inliers_train_mean,
            nll_inliers_test_mean,
            nll_outliers_test_mean,
            fig=fig,
            ax=ax,
        )

    ax.set_title(auroc_text)


plot_kde_auroc(
    nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean, mode="kde"
)
plot_kde_auroc(
    nll_inliers_train_var, nll_inliers_test_var, nll_outliers_test_var, mode="kde"
)


def bae_pred_all(bae_ensemble, x, return_mean=False):
    y_pred_samples = bae_ensemble.predict_samples(x, select_keys=["se", "y_mu"])
    y_latent_samples = bae_ensemble.predict_latent_samples(x)

    y_nll_mean = y_pred_samples.mean(0)[0]
    y_nll_var = y_pred_samples.var(0)[0]
    y_recon_var = y_pred_samples.var(0)[1]
    y_latent_mean = y_latent_samples.mean(0)
    y_latent_var = y_latent_samples.var(0)
    # y_latent_weighted = y_latent_mean/(y_latent_var**0.5)

    if return_mean:
        y_nll_mean = y_nll_mean.mean(-1)
        y_nll_var = y_nll_var.mean(-1)
        y_recon_var = y_recon_var.mean(-1)

    return {
        "nll_mean": y_nll_mean,
        "nll_var": y_nll_var,
        "recon_var": y_recon_var,
        "latent_mean": y_latent_mean,
        "latent_var": y_latent_var,
    }


def bae_predict_ood_v1(
    bae_ensemble, x_train, x_test, keys=["nll_mean", "nll_var"], perc_threshold=99
):
    """
    Combine nll_mean and nll_var method
    """
    preds_train = bae_pred_all(bae_ensemble, x_train, return_mean=True)
    preds_test = bae_pred_all(bae_ensemble, x_test, return_mean=True)
    thresholds = {
        key: calc_ood_threshold(
            training_scores=preds_train[key], perc_threshold=perc_threshold
        )
        for key in keys
    }
    hard_preds = np.array(
        [
            convert_hard_predictions(
                test_scores=preds_test[key], ood_threshold=thresholds[key]
            )
            for key in keys
        ]
    )

    return hard_preds


def bae_predict_ood_v2(bae_ensemble, x_train, x_test, perc_threshold=99):
    """
    Apply Threshold on each samples. Threshold obtained from each BAE sample's training scores.
    """

    preds_train = bae_ensemble.predict_samples(x_train, select_keys=["se"]).mean(-1)[
        :, 0
    ]
    preds_test = bae_ensemble.predict_samples(x_test, select_keys=["se"]).mean(-1)[:, 0]

    thresholds = [
        calc_ood_threshold(training_scores=preds_train_i, perc_threshold=perc_threshold)
        for preds_train_i in preds_train
    ]
    hard_preds = np.array(
        [
            convert_hard_predictions(
                test_scores=preds_test_i, ood_threshold=thresholds_i
            )
            for preds_test_i, thresholds_i in zip(preds_test, thresholds)
        ]
    )

    return hard_preds


hard_preds_inliers_train_v1 = bae_predict_ood_v1(
    bae_ensemble,
    x_inliers_train,
    x_inliers_train,
    keys=["nll_mean", "nll_var"],
    perc_threshold=perc_threshold,
)
hard_preds_inliers_test_v1 = bae_predict_ood_v1(
    bae_ensemble,
    x_inliers_train,
    x_inliers_test,
    keys=["nll_mean", "nll_var"],
    perc_threshold=perc_threshold,
)
hard_preds_outlier_test_v1 = bae_predict_ood_v1(
    bae_ensemble,
    x_inliers_train,
    x_outliers_test,
    keys=["nll_mean", "nll_var"],
    perc_threshold=perc_threshold,
)

plot_kde_auroc(
    hard_preds_inliers_train_v1.mean(0),
    hard_preds_inliers_test_v1.mean(0),
    hard_preds_outlier_test_v1.mean(0),
    mode="hist",
)

hard_preds_inliers_train_v2 = bae_predict_ood_v2(
    bae_ensemble, x_inliers_train, x_inliers_train, perc_threshold=perc_threshold
)
hard_preds_inliers_test_v2 = bae_predict_ood_v2(
    bae_ensemble, x_inliers_train, x_inliers_test, perc_threshold=perc_threshold
)
hard_preds_outlier_test_v2 = bae_predict_ood_v2(
    bae_ensemble, x_inliers_train, x_outliers_test, perc_threshold=perc_threshold
)

plot_kde_auroc(
    hard_preds_inliers_train_v2.mean(0),
    hard_preds_inliers_test_v2.mean(0),
    hard_preds_outlier_test_v2.mean(0),
    mode="hist",
)


def evaluate_f1_score(hard_preds_inliers_test, hard_preds_outlier_test):
    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )
    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    threshold = 0.5
    y_scores[np.argwhere(y_scores >= threshold)] = 1
    y_scores[np.argwhere(y_scores < threshold)] = 0

    f1_score_ = f1_score(y_true, y_scores)
    print("F1-Score: {:.2f}".format(f1_score_))

    return f1_score_


print("F1-SCORE SINGLE THRESHOLD")
f1_score_v1 = evaluate_f1_score(hard_preds_inliers_test_v1, hard_preds_outlier_test_v1)
f1_score_v2 = evaluate_f1_score(hard_preds_inliers_test_v2, hard_preds_outlier_test_v2)

# hard_preds_inliers_test_v1.std(0)
# hard_preds_inliers_test_v2.std(0)
#
# hard_preds_outlier_test_v2.std(0)
#


def evaluate_avgprc_misclass(hard_preds_inliers_test, hard_preds_outlier_test):
    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )

    y_scores_unc = np.concatenate(
        (hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2)
    )

    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    threshold = 0.5
    y_scores[np.argwhere(y_scores >= threshold)] = 1
    y_scores[np.argwhere(y_scores < threshold)] = 0

    error = np.abs(y_scores - y_true)

    avgprc = average_precision_score(error, y_scores_unc)
    print(
        "AVG-PRC: {:.2f} , BASELINE: {:.2f}, AVG-PRC-RATIO: {:.2f}".format(
            avgprc, error.mean(), avgprc / error.mean()
        )
    )

    return avgprc


def evaluate_auprc_misclass(hard_preds_inliers_test, hard_preds_outlier_test):
    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )

    y_scores_unc = np.concatenate(
        (hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2)
    )

    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    threshold = 0.5
    y_scores[np.argwhere(y_scores >= threshold)] = 1
    y_scores[np.argwhere(y_scores < threshold)] = 0

    error = np.abs(y_scores - y_true)

    precision, recall, thresholds = precision_recall_curve(error, y_scores_unc)
    auprc = auc(recall, precision)

    print(
        "AUPRC: {:.2f} , BASELINE: {:.2f}, AUPRC-RATIO: {:.2f}".format(
            auprc, error.mean(), auprc / error.mean()
        )
    )

    return auprc


# precision, recall, thresholds = precision_recall_curve(testy, probs)
# auc = auc(recall, precision)


avgprc_v1 = evaluate_avgprc_misclass(
    hard_preds_inliers_test_v1, hard_preds_outlier_test_v1
)
avgprc_v2 = evaluate_avgprc_misclass(
    hard_preds_inliers_test_v2, hard_preds_outlier_test_v2
)

auprc_v1 = evaluate_auprc_misclass(
    hard_preds_inliers_test_v1, hard_preds_outlier_test_v1
)
auprc_v2 = evaluate_auprc_misclass(
    hard_preds_inliers_test_v2, hard_preds_outlier_test_v2
)


# plt.figure()
# plt.scatter(y_scores_unc,error)
#
# threshold_uncs = np.linspace(0,1,100)
# errors_given_unc = []
# for threshold_unc in threshold_uncs:
#     errors_given_unc.append(error[np.argwhere(y_scores_unc>=threshold_unc)[:,0]].mean())
# errors_given_unc = np.array(errors_given_unc)
#
# plt.figure()
# plt.plot(threshold_uncs, errors_given_unc)


# auprc = average_precision_score(error, y_scores_unc)

# hard_preds_inliers_test = hard_preds_inliers_test_v1
# hard_preds_outlier_test = hard_preds_outlier_test_v1
#
# y_true = np.concatenate((np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
#                          np.ones(hard_preds_outlier_test.mean(0).shape[0])))
#
# y_scores_unc = np.concatenate((hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2))
#
# y_scores = np.concatenate((hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0)))
#
# threshold = 0.5
# y_scores[np.argwhere(y_scores >= threshold)] = 1
# y_scores[np.argwhere(y_scores < threshold)] = 0
# y_scores = y_scores.astype(int)
#
# f1_score_high_unc = f1_score(y_true[np.argwhere(y_scores_unc>=0.25)[:,0]],
#                      y_scores[np.argwhere(y_scores_unc>=0.25)[:,0]])
# f1_score_low_unc = f1_score(y_true[np.argwhere(y_scores_unc<0.25)[:,0]],
#                      y_scores[np.argwhere(y_scores_unc<0.25)[:,0]])
# f1_score_all = f1_score(y_true, y_scores)
#
# x_labels = ['Low Unc', 'High Unc', 'All']
# scores = [f1_score_low_unc,f1_score_high_unc,f1_score_all]
#
# x_pos = [i for i, _ in enumerate(x_labels)]
#
# plt.figure()
# plt.bar(x_pos, scores, color='tab:blue')
# plt.xticks(x_pos, x_labels)
#
# print(f1_score_high_unc)
# print(f1_score_low_unc)
# print(f1_score_all)
# print(f1_score_low_unc-f1_score_high_unc)
# print(len(np.argwhere(y_scores_unc>=0.25)[:,0])/y_scores_unc.shape[0])
# print(len(np.argwhere(y_scores_unc<0.25)[:,0])/y_scores_unc.shape[0])

# =============MULTIPLE THRESHOLDS==============


def multi_hard_predict_v1(
    bae_ensemble,
    x_inliers_train,
    x_test,
    keys=["nll_mean", "nll_var"],
    thresholds=np.arange(80, 100),
):
    outputs = [
        bae_predict_ood_v1(
            bae_ensemble,
            x_inliers_train,
            x_test,
            keys=keys,
            perc_threshold=perc_threshold_,
        )
        for perc_threshold_ in thresholds
    ]
    outputs = np.concatenate(outputs, axis=0)
    return outputs


def multi_hard_predict_v2(
    bae_ensemble, x_inliers_train, x_test, thresholds=np.arange(80, 100)
):
    outputs = [
        bae_predict_ood_v2(
            bae_ensemble, x_inliers_train, x_test, perc_threshold=perc_threshold_
        )
        for perc_threshold_ in thresholds
    ]
    outputs = np.concatenate(outputs, axis=0)
    return outputs


def evaluate_unc(
    hard_preds_inliers_test,
    hard_preds_outlier_test,
    uncertainty_threshold=0.95,
    decision_threshold=0.5,
):
    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )

    y_scores_unc = np.concatenate(
        (hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2)
    )

    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    decision_threshold = 0.5
    y_scores[np.argwhere(y_scores >= decision_threshold)] = 1
    y_scores[np.argwhere(y_scores < decision_threshold)] = 0
    y_scores = y_scores.astype(int)

    f1_score_high_unc = f1_score(
        y_true[np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0]],
        y_scores[np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0]],
    )
    f1_score_low_unc = f1_score(
        y_true[np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0]],
        y_scores[np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0]],
    )
    f1_score_all = f1_score(y_true, y_scores)

    perc_uncertain = (
        len(np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0])
        / y_scores_unc.shape[0]
    )
    perc_certain = (
        len(np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0])
        / y_scores_unc.shape[0]
    )

    print("HIGH UNC: {:.2f}".format((f1_score_high_unc)))
    print("LOW UNC: {:.2f}".format((f1_score_low_unc)))
    print("W/O UNC: {:.2f}".format((f1_score_all)))
    print("% UNC: {:.2f}".format((perc_uncertain)))
    print("% CER: {:.2f}".format((perc_certain)))

    return f1_score_high_unc, f1_score_low_unc, perc_uncertain, perc_certain


print("F1-SCORE MULTI THRESHOLD")

multihard_preds_inliers_train_v1 = multi_hard_predict_v1(
    bae_ensemble,
    x_inliers_train,
    x_inliers_train,
    keys=["nll_mean", "nll_var"],
    thresholds=multi_perc_thresholds,
)
multihard_preds_inliers_test_v1 = multi_hard_predict_v1(
    bae_ensemble,
    x_inliers_train,
    x_inliers_test,
    keys=["nll_mean", "nll_var"],
    thresholds=multi_perc_thresholds,
)
multihard_preds_outlier_test_v1 = multi_hard_predict_v1(
    bae_ensemble,
    x_inliers_train,
    x_outliers_test,
    keys=["nll_mean", "nll_var"],
    thresholds=multi_perc_thresholds,
)

multihard_preds_inliers_train_v2 = multi_hard_predict_v2(
    bae_ensemble, x_inliers_train, x_inliers_train, thresholds=multi_perc_thresholds
)
multihard_preds_inliers_test_v2 = multi_hard_predict_v2(
    bae_ensemble, x_inliers_train, x_inliers_test, thresholds=multi_perc_thresholds
)
multihard_preds_outlier_test_v2 = multi_hard_predict_v2(
    bae_ensemble, x_inliers_train, x_outliers_test, thresholds=multi_perc_thresholds
)

plot_kde_auroc(
    multihard_preds_inliers_train_v1.mean(0),
    multihard_preds_inliers_test_v1.mean(0),
    multihard_preds_outlier_test_v1.mean(0),
    mode="hist",
)

plot_kde_auroc(
    multihard_preds_inliers_train_v2.mean(0),
    multihard_preds_inliers_test_v2.mean(0),
    multihard_preds_outlier_test_v2.mean(0),
    mode="hist",
)

f1_score_v1 = evaluate_f1_score(
    multihard_preds_inliers_test_v1, multihard_preds_outlier_test_v1
)
f1_score_v2 = evaluate_f1_score(
    multihard_preds_inliers_test_v2, multihard_preds_outlier_test_v2
)

avgprc_v1 = evaluate_avgprc_misclass(
    multihard_preds_inliers_test_v1, multihard_preds_outlier_test_v1
)
avgprc_v2 = evaluate_avgprc_misclass(
    multihard_preds_inliers_test_v2, multihard_preds_outlier_test_v2
)

auprc_v1 = evaluate_auprc_misclass(
    multihard_preds_inliers_test_v1, multihard_preds_outlier_test_v1
)
auprc_v2 = evaluate_auprc_misclass(
    multihard_preds_inliers_test_v2, multihard_preds_outlier_test_v2
)


# ==========================F1 SCORE | UNC ==============================

uncertainty_threshold = 0.5
evaluate_unc(
    multihard_preds_inliers_test_v1,
    multihard_preds_outlier_test_v1,
    uncertainty_threshold=uncertainty_threshold,
)
print("----------------")
evaluate_unc(
    multihard_preds_inliers_test_v2,
    multihard_preds_outlier_test_v2,
    uncertainty_threshold=uncertainty_threshold,
)


# =========================DEBUG: PLOT LATENT============================
def predict_latent_ex(bae_ensemble, x, n_ae=1):
    y_latent_samples = bae_ensemble.predict_latent_samples(x)
    # y_latent_mean = y_latent_samples[n_ae]
    y_latent_mean = y_latent_samples.std(0)
    return y_latent_mean


def predict_latent_pca(bae_ensemble, x):
    y_latent_mean, _ = bae_ensemble.predict_latent(x, transform_pca=True)
    return y_latent_mean


latent_inliers_train_mean = bae_pred_all(
    bae_ensemble, x_inliers_train, return_mean=False
)["latent_mean"]
latent_inliers_test_mean = bae_pred_all(
    bae_ensemble, x_inliers_test, return_mean=False
)["latent_mean"]
latent_outliers_test_mean = bae_pred_all(
    bae_ensemble, x_outliers_test, return_mean=False
)["latent_mean"]

# latent_inliers_train_mean = bae_pred_all(bae_ensemble, x_inliers_train, return_mean=False)["latent_var"]
# latent_inliers_test_mean = bae_pred_all(bae_ensemble, x_inliers_test, return_mean=False)["latent_var"]
# latent_outliers_test_mean = bae_pred_all(bae_ensemble, x_outliers, return_mean=False)["latent_var"]

n_ae = 1
latent_inliers_train_mean = predict_latent_ex(bae_ensemble, x_inliers_train, n_ae=n_ae)
latent_inliers_test_mean = predict_latent_ex(bae_ensemble, x_inliers_test, n_ae=n_ae)
latent_outliers_test_mean = predict_latent_ex(bae_ensemble, x_outliers_test, n_ae=n_ae)

latent_inliers_train_pca = predict_latent_pca(bae_ensemble, x_inliers_train)
latent_inliers_test_pca = predict_latent_pca(bae_ensemble, x_inliers_test)
latent_outliers_test_pca = predict_latent_pca(bae_ensemble, x_outliers_test)

plt.figure()
for sample in latent_inliers_train_mean:
    plt.plot(sample, alpha=0.5, color="tab:blue")

for sample in latent_inliers_test_mean:
    plt.plot(sample, alpha=0.5, color="tab:green")

for sample in latent_outliers_test_mean:
    plt.plot(sample, alpha=0.5, color="tab:orange")

plt.figure()
plt.scatter(
    latent_inliers_train_pca[:, 0],
    latent_inliers_train_pca[:, 1],
    alpha=0.5,
    color="tab:blue",
)
plt.scatter(
    latent_inliers_test_pca[:, 0],
    latent_inliers_test_pca[:, 1],
    alpha=0.5,
    color="tab:green",
)
plt.scatter(
    latent_outliers_test_pca[:, 0],
    latent_outliers_test_pca[:, 1],
    alpha=0.5,
    color="tab:orange",
)

# ========================================

from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.models.iforest import IForest


# train kNN detector

# clf = KNN()
# clf = AutoEncoder(epochs=500)
clf = OCSVM()
# clf = IForest()


def evaluate_clf(
    clf, x_inliers_train, x_inliers_test, x_outliers, scaler=None, roundup=3
):

    if scaler is not None:
        scaler = scaler.fit(x_inliers_train)
        x_inliers_train = scaler.transform(x_inliers_train)
        x_inliers_test = scaler.transform(x_inliers_test)
        x_outliers = scaler.transform(x_outliers)

    clf.fit(x_inliers_train)
    y_inliers_pred = clf.predict(x_inliers_test)
    y_outliers_pred = clf.predict(x_outliers)

    y_inliers_pred_raw = clf.decision_function(x_inliers_test)
    y_outliers_pred_raw = clf.decision_function(x_outliers)

    pyod_auroc = calc_auroc(y_inliers_pred_raw, y_outliers_pred_raw)

    y_true = np.concatenate(
        (np.zeros(y_inliers_pred.shape[0]), np.ones(y_outliers_pred.shape[0]))
    )

    f1_score_ = f1_score(
        y_true, np.concatenate((y_inliers_pred, y_outliers_pred), axis=0)
    )
    mcc_score_ = matthews_corrcoef(
        y_true, np.concatenate((y_inliers_pred, y_outliers_pred), axis=0)
    )

    res = {
        "f1": np.round(f1_score_, roundup),
        "mcc": np.round(mcc_score_, roundup),
        "auroc": np.round(pyod_auroc, roundup),
    }
    print(res)
    return res


# latent_scaler = MinMaxScaler()
latent_scaler = StandardScaler()
# latent_scaler.fit(x_inliers_train)

print("PYOD-RAW-DATA")
pyod_ori = evaluate_clf(
    clf, x_inliers_train, x_inliers_test, x_outliers_test, scaler=latent_scaler
)
# pyod_ori = evaluate_clf(clf, latent_scaler.transform(x_inliers_train), latent_scaler.transform(x_inliers_test), latent_scaler.transform(x_outliers))
#
# pyod_latent = evaluate_clf(clf,
#                            latent_scaler.fit_transform(latent_inliers_train_mean),
#                            latent_scaler.transform(latent_inliers_test_mean),
#                            latent_scaler.transform(latent_outliers_test_mean))

print("PYOD-LATENT-RAW")
pyod_latent = evaluate_clf(
    clf,
    latent_inliers_train_mean,
    latent_inliers_test_mean,
    latent_outliers_test_mean,
    scaler=latent_scaler,
)

print("PYOD-LATENT-PCA")
pyod_latent_pca = evaluate_clf(
    clf,
    latent_inliers_train_pca,
    latent_inliers_test_pca,
    latent_outliers_test_pca,
    scaler=latent_scaler,
)


# ======================OC-NLL================
nll_inliers_train_concat = np.concatenate(
    (
        np.expand_dims(nll_inliers_train_mean, 1),
        np.expand_dims(nll_inliers_train_var, 1),
    ),
    axis=1,
)

nll_inliers_test_concat = np.concatenate(
    (np.expand_dims(nll_inliers_test_mean, 1), np.expand_dims(nll_inliers_test_var, 1)),
    axis=1,
)

nll_outliers_test_concat = np.concatenate(
    (
        np.expand_dims(nll_outliers_test_mean, 1),
        np.expand_dims(nll_outliers_test_var, 1),
    ),
    axis=1,
)
nll_outliers_train_concat = np.concatenate(
    (
        np.expand_dims(nll_outliers_train_mean, 1),
        np.expand_dims(nll_outliers_train_var, 1),
    ),
    axis=1,
)

# nll_inliers_train_mean = nll_inliers_train_concat.mean(0)[0].mean(-1)
# nll_inliers_test_mean = nll_inliers_test.mean(0)[0].mean(-1)
# nll_outliers_test_mean = nll_outliers_test.mean(0)[0].mean(-1)
# nll_outliers_train_mean = nll_outliers_train.mean(0)[0].mean(-1)

plt.figure()
plt.scatter(nll_inliers_train_concat[:, 0], nll_inliers_train_concat[:, 1])
plt.scatter(nll_inliers_test_concat[:, 0], nll_inliers_test_concat[:, 1])
plt.scatter(nll_outliers_test_concat[:, 0], nll_outliers_test_concat[:, 1])
plt.scatter(nll_outliers_train_concat[:, 0], nll_outliers_train_concat[:, 1])

print("PYOD-MEAN-VAR-CONCAT")
pyod_mean_var_nll = evaluate_clf(
    clf,
    nll_inliers_train_concat,
    nll_inliers_test_concat,
    nll_outliers_test_concat,
    scaler=latent_scaler,
)

nll_inliers_train_raw_mean = nll_inliers_train.mean(0)[0]
nll_inliers_test_raw_mean = nll_inliers_test.mean(0)[0]
nll_outliers_test_raw_mean = nll_outliers_test.mean(0)[0]
nll_outliers_train_raw_mean = nll_outliers_train.mean(0)[0]

nll_inliers_train_raw_var = nll_inliers_train.var(0)[0]
nll_inliers_test_raw_var = nll_inliers_test.var(0)[0]
nll_outliers_test_raw_var = nll_outliers_test.var(0)[0]
nll_outliers_train_raw_var = nll_outliers_train.var(0)[0]

nll_inliers_train_raw_combined = np.concatenate(
    (nll_inliers_train_raw_mean, nll_inliers_train_raw_var), axis=1
)
nll_inliers_test_raw_combined = np.concatenate(
    (nll_inliers_test_raw_mean, nll_inliers_test_raw_var), axis=1
)
nll_outliers_test_raw_combined = np.concatenate(
    (nll_outliers_test_raw_mean, nll_outliers_test_raw_var), axis=1
)
nll_outliers_train_raw_combined = np.concatenate(
    (nll_outliers_train_raw_mean, nll_outliers_train_raw_var), axis=1
)


print("PYOD-MEAN-RAW")
pyod_mean_raw_nll = evaluate_clf(
    clf,
    nll_inliers_train_raw_mean,
    nll_inliers_test_raw_mean,
    nll_outliers_test_raw_mean,
    scaler=latent_scaler,
)

print("PYOD-VAR-RAW")
pyod_var_raw_nll = evaluate_clf(
    clf,
    nll_inliers_train_raw_var,
    nll_inliers_test_raw_var,
    nll_outliers_test_raw_var,
    scaler=latent_scaler,
)

print("PYOD-COMBINED-RAW")
pyod_var_raw_nll = evaluate_clf(
    clf,
    nll_inliers_train_raw_combined,
    nll_inliers_test_raw_combined,
    nll_outliers_test_raw_combined,
    scaler=latent_scaler,
)


# pyod_latent = evaluate_clf(clf, latent_inliers_train_mean, latent_inliers_test_mean, latent_outliers_test_mean)

# clf = clf.fit(latent_scaler.fit_transform(nll_inliers_train_raw_combined))
# pyod_id_score = clf.decision_function(latent_scaler.transform(nll_inliers_test_raw_combined))
# pyod_ood_score = clf.decision_function(latent_scaler.transform(nll_outliers_test_raw_combined))
# pyod_auroc = calc_auroc(pyod_id_score, pyod_ood_score)
# print("PYOD NLL-RAW AUROC: {:.3f}".format(pyod_auroc))

# pyod_latent = evaluate_clf(clf, latent_inliers_train_mean, latent_inliers_test_mean, latent_outliers_test_mean)


# =============================================
from sklearn.metrics import accuracy_score

# accuracy_score(y_true, y_pred)


def evaluate_unc_(
    hard_preds_inliers_test,
    hard_preds_outlier_test,
    uncertainty_threshold=0.95,
    decision_threshold=0.5,
):
    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )

    y_scores_unc = np.concatenate(
        (hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2)
    )

    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    y_scores[np.argwhere(y_scores >= decision_threshold)] = 1
    y_scores[np.argwhere(y_scores < decision_threshold)] = 0
    y_hard_pred = y_scores.astype(int)

    error = y_hard_pred - y_true

    # f1_score_high_unc = f1_score(y_true[np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0]],
    #                              y_scores[np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0]])
    # f1_score_low_unc = f1_score(y_true[np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0]],
    #                             y_scores[np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0]])
    # f1_score_all = f1_score(y_true, y_scores)

    # perc_uncertain = len(np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0]) / y_scores_unc.shape[0]
    # perc_certain = len(np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0]) / y_scores_unc.shape[0]
    #
    # print("HIGH UNC: {:.2f}".format((f1_score_high_unc)))
    # print("LOW UNC: {:.2f}".format((f1_score_low_unc)))
    # print("W/O UNC: {:.2f}".format((f1_score_all)))
    # print("% UNC: {:.2f}".format((perc_uncertain)))
    # print("% CER: {:.2f}".format((perc_certain)))

    # return f1_score_high_unc, f1_score_low_unc, perc_uncertain, perc_certain

    # return f1_score_low_unc
    return error, y_scores_unc, y_hard_pred


#
# evaluate_unc(multihard_preds_inliers_test_v2, multihard_preds_outlier_test_v2, uncertainty_threshold=uncertainty_threshold)
#
# unc_thresholds = np.linspace(0,100,2)
# unc_thresholds = np.linspace(0,50,10)
# unc_f1_scores = [evaluate_unc(multihard_preds_inliers_test_v2,
#                         multihard_preds_outlier_test_v2,
#                         uncertainty_threshold=unc,
#                         decision_threshold=0.5)[0] for unc in unc_thresholds]
#
# plt.figure()
# plt.plot(unc_thresholds,unc_f1_scores)
#
#
# #======================================
# pred_error, y_scores_unc,y_hard_pred = evaluate_unc_(multihard_preds_inliers_test_v2,
#                         multihard_preds_outlier_test_v2)
#
# tn_indices = np.argwhere((pred_error == 0) & (y_hard_pred == 0))[:,0]
#
# def inverse_select(data, indices):
#     return data[np.setdiff1d(np.arange(data.shape[0]), indices)]
#
# pred_error = np.abs(inverse_select(pred_error, tn_indices))
# y_scores_unc = inverse_select(y_scores_unc, tn_indices)
#
# # type1_error = # false positives
#
# plt.figure()
# plt.hist(y_scores_unc)
#
# unc_thresholds = np.linspace(0,1.,6)
# bins = [pred_error[np.argwhere((y_scores_unc<unc_thresholds[i+1]) &
#                               (y_scores_unc>=unc_thresholds[i]))[:,0]]
#         for i in range(len(unc_thresholds)-1)]
#
# bins = [np.mean(bin) for bin in bins]
#
# print(bins)
#
#


# =================FPR AND FNR ==============
from sklearn.metrics import confusion_matrix


def get_y_unc(
    hard_preds_inliers_test,
    hard_preds_outlier_test,
    uncertainty_threshold=0.95,
    decision_threshold=0.5,
):
    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )

    y_scores_unc = np.concatenate(
        (hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2)
    )

    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    y_scores[np.argwhere(y_scores >= decision_threshold)] = 1
    y_scores[np.argwhere(y_scores < decision_threshold)] = 0
    y_hard_pred = y_scores.astype(int)

    # error = (y_hard_pred-y_true)

    return y_scores_unc, y_hard_pred, y_true


def calc_error_unc(y_scores_unc, y_true, y_hard_pred, unc_threshold=0):
    indices = np.argwhere(y_scores_unc <= unc_threshold)[:, 0]
    tn, fp, fn, tp = confusion_matrix(y_true[indices], y_hard_pred[indices]).ravel()
    fpr = fp / (fp + tn)
    fdr = fp / (fp + tp)
    fnr = fn / (fn + tp)
    forate = fn / (fn + tn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    mcc = ((tp * tn) - (fp * fn)) / (
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    if np.isnan(mcc):
        mcc = 0
    f1 = tp / (tp + 0.5 * (fp + fn))
    perc = len(indices) / len(y_scores_unc)
    ba = (tpr + tnr) / 2

    # return fpr, fdr, fnr, forate, tpr, tnr, mcc, f1, perc
    return fpr, fdr, fnr, forate, tpr, tnr, mcc, f1, perc, ba
    # return fpr, fdr, fnr, tpr, tnr, mcc, f1, perc


# unc_thresholds = [0.1,0.5,0.8,1.]
# y_scores_unc, y_hard_pred, y_true = get_y_unc(multihard_preds_inliers_test_v1,
#                         multihard_preds_outlier_test_v1)
#

y_scores_unc, y_hard_pred, y_true = get_y_unc(
    multihard_preds_inliers_test_v2, multihard_preds_outlier_test_v2
)

unc_thresholds = np.unique(np.round(y_scores_unc, 3))
error_uncs = np.array(
    [
        calc_error_unc(y_scores_unc, y_true, y_hard_pred, unc_threshold=unc_)
        for unc_ in unc_thresholds
    ]
)

# error_uncs[:,0] /= error_uncs[0,0]
# error_uncs[:,1] /= error_uncs[0,1]
# error_uncs[:,2] /= error_uncs[0,2]
# error_uncs[:,3] /= error_uncs[0,3]
# error_uncs[:,3] /= error_uncs[0,3]
# error_uncs[:,3] /= error_uncs[0,3]

# error_uncs /= error_uncs[0,:]

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1)
# ax1.plot(unc_thresholds, error_uncs[:,0], '-o')
# ax2.plot(unc_thresholds, error_uncs[:,1], '-o')
# ax3.plot(unc_thresholds, error_uncs[:,2], '-o')
# ax4.plot(unc_thresholds, error_uncs[:,3], '-o')
ax1.plot(unc_thresholds, error_uncs[:, 0], "-o")
ax2.plot(unc_thresholds, error_uncs[:, 1], "-o")
ax3.plot(unc_thresholds, error_uncs[:, 3], "-o")
ax4.plot(unc_thresholds, error_uncs[:, 5], "-o")
ax5.plot(unc_thresholds, error_uncs[:, 6], "-o")
ax6.plot(unc_thresholds, error_uncs[:, 7], "-o")
ax7.plot(unc_thresholds, error_uncs[:, 8], "-o")

# ax8.plot(unc_thresholds, error_uncs[:,8]*error_uncs[:,6], '-o')

spman_mcc = spearmanr(unc_thresholds, error_uncs[:, 6])[0]
spman_f1 = spearmanr(unc_thresholds, error_uncs[:, 7])[0]

mcc_diff = error_uncs[0, 6] - error_uncs[-1, 6]
f1_diff = error_uncs[0, 7] - error_uncs[-1, 7]

print("SPMAN MCC : {:.3f}".format(spman_mcc))
print("SPMAN F1 : {:.3f}".format(spman_f1))
print("DIFF MCC : {:.3f}".format(mcc_diff))
print("DIFF F1 : {:.3f}".format(f1_diff))
print("PERC HIGH : {:.3f}".format(error_uncs[0, 8]))
print("MCC HIGH : {:.3f}".format(error_uncs[0, 6]))
print("MCC LOW : {:.3f}".format(error_uncs[-1, 6]))
print("F1 HIGH : {:.3f}".format(error_uncs[0, 7]))
print("F1 LOW : {:.3f}".format(error_uncs[-1, 7]))
print("MCC MEAN : {:.3f}".format(error_uncs[:, 6].mean()))
print("F1 MEAN : {:.3f}".format(error_uncs[:, 7].mean()))
# print("PERC MEAN : {:.3f}".format(error_uncs[:,8].mean()))
# print("WEIGHTED MCC : {:.3f}".format((error_uncs[:,6]*error_uncs[:,8]).mean()))
# print("WEIGHTED F1 : {:.3f}".format((error_uncs[:,7]*error_uncs[:,8]).mean()))

# print("WEIGHTED MCC : {:.3f}".format((error_uncs[0,6]*error_uncs[0,8]+error_uncs[-1,6]*error_uncs[-1,8])/2))

# print("WEIGHTED F1 : {:.3f}".format((error_uncs[:,7]*error_uncs[:,8]).mean()))


# ================UNCERTAINTY WRT FPFNTNTP===================
y_scores_unc, y_hard_pred, y_true = get_y_unc(
    multihard_preds_inliers_test_v1, multihard_preds_outlier_test_v1
)

# y_scores_unc, y_hard_pred, y_true = get_y_unc(multihard_preds_inliers_test_v2,
#                         multihard_preds_outlier_test_v2)
y_unc = y_scores_unc

indices_tp = np.argwhere((y_true == 1) & (y_hard_pred == 1))[:, 0]
indices_tn = np.argwhere((y_true == 0) & (y_hard_pred == 0))[:, 0]
indices_fp = np.argwhere((y_true == 0) & (y_hard_pred == 1))[:, 0]
indices_fn = np.argwhere((y_true == 1) & (y_hard_pred == 0))[:, 0]
indices_0_error = np.concatenate((indices_tp, indices_tn))
indices_all_error = np.concatenate((indices_fp, indices_fn))

error_type1 = np.concatenate(
    (np.ones(len(indices_fp)), np.zeros(len(indices_tp)))
).astype(int)
error_type2 = np.concatenate(
    (np.ones(len(indices_fn)), np.zeros(len(indices_tn)))
).astype(int)
error_all = np.abs((y_true - y_hard_pred))

y_unc_type1 = np.concatenate((y_unc[indices_fp], y_unc[indices_tp]))
y_unc_type2 = np.concatenate((y_unc[indices_fn], y_unc[indices_tn]))
y_unc_all = y_unc

precision_type1, recall_type1, thresholds = precision_recall_curve(
    error_type1, y_unc_type1
)
precision_type2, recall_type2, thresholds = precision_recall_curve(
    error_type2, y_unc_type2
)
precision_type_all, recall_type_all, thresholds = precision_recall_curve(
    error_all, y_unc
)

auprc_type1 = auc(recall_type1, precision_type1)
auprc_type2 = auc(recall_type2, precision_type2)
auprc_type_all = auc(recall_type_all, precision_type_all)

baseline_type1 = error_type1.mean()
baseline_type2 = error_type2.mean()
baseline_all = error_all.mean()

lift_type1 = auprc_type1 / baseline_type1
lift_type2 = auprc_type2 / baseline_type2
lift_all = auprc_type_all / baseline_all

labels = ["TP", "TN", "TP+TN", "Type 1 (FP)", "Type 2 (FN)", "FP+FN"]
fig, ax1 = plt.subplots(1, 1)
ax1.boxplot(
    [
        y_unc[indices_tp],
        y_unc[indices_tn],
        y_unc[indices_0_error],
        y_unc[indices_fp],
        y_unc[indices_fn],
        y_unc[indices_all_error],
    ],
    notch=False,
)

# ax1.violinplot([y_unc[indices_tp],
#              y_unc[indices_tn],
#              y_unc[indices_0_error],
#              y_unc[indices_fp],
#              y_unc[indices_fn],
#              y_unc[indices_all_error],
#              ], showmedians=True, showextrema=True
#             )
ax1.set_xticks(np.arange(1, len(labels) + 1))
ax1.set_xticklabels(labels)
ax1.set_ylabel("Uncertainty")

print(
    "AUPRC-TYPE1 (TP VS FP): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(
        auprc_type1, baseline_type1, lift_type1
    )
)
print(
    "AUPRC-TYPE2 (TN VS FN): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(
        auprc_type2, baseline_type2, lift_type2
    )
)
print(
    "AUPRC-ALL   (TP+TN VS FP+FN): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(
        auprc_type_all, baseline_all, lift_all
    )
)

auroc_type1 = (
    roc_auc_score(error_type1, y_unc_type1)
    if (baseline_type1 > 0 or baseline_type1 == 1)
    else np.nan
)
auroc_type2 = (
    roc_auc_score(error_type2, y_unc_type2)
    if (baseline_type2 > 0 or baseline_type2 == 1)
    else np.nan
)
auroc_type_all = (
    roc_auc_score(error_all, y_unc)
    if (baseline_all > 0 or baseline_all == 1)
    else np.nan
)

print("AUROC-TYPE1 (TP VS FP): {:.2f} ".format(auprc_type1))
print("AUROC-TYPE2 (TN VS FN): {:.2f} ".format(auroc_type2))
print("AUROC-ALL   (TP+TN VS FP+FN): {:.2f} ".format(auroc_type_all))


# =====Adjusted box plot============
# plot_kde_auroc(nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean, mode="kde")
# plot_kde_auroc(nll_inliers_train_var, nll_inliers_test_var, nll_outliers_test_var, mode="kde")

# mc_1 = medcouple(nll_inliers_train_mean)
mc_1 = medcouple(nll_outliers_test_mean)

# [L, R] = [Q1 - 1.5 * exp(-3.5MC) *IQR, Q3 + 1.5 * exp(4
# MC) *IQR] if MC ≥ 0
# = [Q1 - 1.5 * exp(-4MC) *IQR, Q3 + 1.5 * exp(3.5
# MC) *IQR] if MC ≤ 0

multiplier = 1.5 * np.exp(3 * mc_1) if mc_1 >= 0 else 1.5 * np.exp(4 * mc_1)

fig, ax1 = plt.subplots(1, 1)
# ax1.boxplot([nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean])

# ax1.boxplot([nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean], whis=multiplier)
# ax1.boxplot([nll_inliers_train_mean,nll_inliers_valid_mean, nll_inliers_test_mean, nll_outliers_test_mean])

# ax1.boxplot([nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean], whis=(0,95))
# ax1.boxplot([nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean], whis=(0,95))
# ax1.violinplot([nll_inliers_train_mean,nll_inliers_valid_mean, nll_inliers_test_mean, nll_outliers_test_mean])
# ax1.violinplot([nll_inliers_train_mean,np.concatenate((nll_inliers_train_mean,nll_inliers_valid_mean)), nll_inliers_test_mean, nll_outliers_test_mean])
# ax1.boxplot([nll_inliers_train_mean,np.concatenate((nll_inliers_train_mean,nll_inliers_valid_mean)), nll_inliers_test_mean, nll_outliers_test_mean])
ax1.boxplot(
    [
        nll_inliers_train_mean,
        nll_inliers_valid_mean,
        nll_inliers_test_mean,
        nll_outliers_test_mean,
    ]
)

ax1.set_xticks(np.arange(1, 5))
# ax1.set_xticklabels(["Train (Inlier)", "Test (Inlier)", "Test (Outlier)"])
ax1.set_xticklabels(
    ["Train (Inlier)", "Valid (Inlier)", "Test (Inlier)", "Test (Outlier)"]
)
