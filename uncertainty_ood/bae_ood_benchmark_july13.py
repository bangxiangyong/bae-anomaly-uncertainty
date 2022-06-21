from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
import os
import numpy as np
from scipy.special import erf
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from statsmodels.distributions import ECDF

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
from sklearn.metrics import confusion_matrix
from scipy.stats import beta, gamma, lognorm, norm, uniform, expon
from util.uncertainty import (
    convert_prob,
    convert_cdf,
    get_pred_unc,
    get_y_results,
    evaluate_mcc_f1_unc,
    evaluate_unc_perf,
    plot_unc_tptnfpfn,
    plot_kde_auroc,
    calc_error_unc,
)


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
clip_data_01 = False
activation = "leakyrelu"
last_activation = "tanh"  # tanh
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
num_train_outliers = 2

# ==============PREPARE DATA==========================
base_folder = "od_benchmark"
mat_file_list = os.listdir(base_folder)
mat_file = mat_file_list[-1]
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

x_inliers_train, x_inliers_test = train_test_split(
    x_inliers, train_size=train_size, shuffle=True, random_state=random_seed
)
x_inliers_train, x_inliers_valid = train_test_split(
    x_inliers_train, train_size=train_size, shuffle=True, random_state=random_seed
)


# =================SCALER=========================
scaler = MinMaxScaler()
# scaler = StandardScaler()

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
    x_inliers_valid = np.clip(x_inliers_valid, 0, 1)
    x_outliers_train = np.clip(x_outliers_train, 0, 1)

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
    print("AUROC-THRESHOLD: {:.3f}".format(auroc_threshold))

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


# =======================LAST FEATURE======================

thresholds = (None, None)
p_threshold = 0.5


nll_inliers_train = bae_ensemble.predict_samples(
    x_inliers_train, select_keys=["se"]
).mean(-1)[:, 0]
nll_inliers_test = bae_ensemble.predict_samples(
    x_inliers_test, select_keys=["se"]
).mean(-1)[:, 0]
nll_inliers_valid = bae_ensemble.predict_samples(
    x_inliers_valid, select_keys=["se"]
).mean(-1)[:, 0]
nll_outliers_test = bae_ensemble.predict_samples(
    x_outliers_test, select_keys=["se"]
).mean(-1)[:, 0]
nll_outliers_train = bae_ensemble.predict_samples(
    x_outliers_train, select_keys=["se"]
).mean(-1)[:, 0]


def predict_ood_unc(
    bae_ensemble,
    x_ref,
    x_inliers,
    x_outliers,
    p_threshold=0.5,
    dist_cdf="ecdf",
    scaling=True,
):
    # get the NLL (BAE samples)
    nll_ref = bae_ensemble.predict_samples(x_ref, select_keys=["se"]).mean(-1)[:, 0]
    nll_inliers = bae_ensemble.predict_samples(x_inliers, select_keys=["se"]).mean(-1)[
        :, 0
    ]
    nll_outliers = bae_ensemble.predict_samples(x_outliers, select_keys=["se"]).mean(
        -1
    )[:, 0]

    # convert to outlier probability (BAE samples)
    prob_inliers_test, unc_inliers_test = convert_prob(
        convert_cdf(
            nll_inliers_valid, nll_inliers_valid, dist=dist_cdf, scaling=scaling
        ),
        *thresholds
    )
    prob_outliers_test, unc_outliers_test = convert_prob(
        convert_cdf(
            nll_inliers_valid, nll_outliers_test, dist=dist_cdf, scaling=scaling
        ),
        *thresholds
    )

    # compute mean and var over probabilistic BAE samples
    prob_inliers_test_mean = prob_inliers_test.mean(0)
    prob_outliers_test_mean = prob_outliers_test.mean(0)

    total_unc_inliers_test = get_pred_unc(prob_inliers_test, unc_inliers_test)
    total_unc_outliers_test = get_pred_unc(prob_outliers_test, unc_outliers_test)

    # compress into y_true, y_hard, and y_unc for results evaluation
    y_true, y_hard_pred, y_unc = get_y_results(
        prob_inliers_test_mean,
        prob_outliers_test_mean,
        total_unc_inliers_test,
        total_unc_outliers_test,
        p_threshold=p_threshold,
    )

    return y_true, y_hard_pred, y_unc


y_true, y_hard_pred, y_unc = predict_ood_unc(
    bae_ensemble,
    x_inliers_valid,
    x_inliers_test,
    x_outliers_test,
    p_threshold=0.5,
)

# USE METHOD 1
# prob_inliers_train, unc_inliers_train = convert_prob(convert_ecdf_output(nll_inliers_train,
#                                                      nll_inliers_train),
#                                  *thresholds)
# prob_inliers_test, unc_inliers_test = convert_prob(convert_ecdf_output(nll_inliers_train,
#                                                      nll_inliers_test),
#                                  *thresholds)
# prob_outliers_test, unc_outliers_test = convert_prob(convert_ecdf_output(nll_inliers_train,
#                                                      nll_outliers_test),
#                                  *thresholds)

# prob_inliers_train, unc_inliers_train = convert_prob(convert_ecdf_output(nll_inliers_train,
#                                                      nll_inliers_train),
#                                  *thresholds)
# prob_inliers_test, unc_inliers_test = convert_prob(convert_ecdf_output(nll_inliers_valid,
#                                                      nll_inliers_valid),
#                                  *thresholds)
# prob_outliers_test, unc_outliers_test = convert_prob(convert_ecdf_output(nll_inliers_valid,
#                                                      nll_outliers_test),
#                                  *thresholds)
#

#
# prob_inliers_train, unc_inliers_train = convert_prob(convert_erf(nll_inliers_train,
#                                                      nll_inliers_train),
#                                  *thresholds)
# prob_inliers_test, unc_inliers_test = convert_prob(convert_erf(nll_inliers_valid,
#                                                      nll_inliers_valid),
#                                  *thresholds)
# prob_outliers_test, unc_outliers_test = convert_prob(convert_erf(nll_inliers_valid,
#                                                      nll_outliers_test),
#                                  *thresholds)


# prob_inliers_train, unc_inliers_train = convert_prob(convert_minmax(nll_inliers_train,
#                                                      nll_inliers_train),
#                                  *thresholds)
# prob_inliers_test, unc_inliers_test = convert_prob(convert_minmax(nll_inliers_valid,
#                                                      nll_inliers_valid),
#                                  *thresholds)
# prob_outliers_test, unc_outliers_test = convert_prob(convert_minmax(nll_inliers_valid,
#                                                      nll_outliers_test),
#                                  *thresholds)

# unc_type = "epistemic"
# unc_type = "aleatoric"
unc_type = "total"


dist_cdf = "expon"  # gamma, lognorm , norm , uniform, expon, "ecdf"
scaling = True
p_threshold = 0.5
prob_inliers_train, unc_inliers_train = convert_prob(
    convert_cdf(nll_inliers_train, nll_inliers_train, dist=dist_cdf, scaling=scaling),
    *thresholds
)
prob_inliers_test, unc_inliers_test = convert_prob(
    convert_cdf(nll_inliers_valid, nll_inliers_valid, dist=dist_cdf, scaling=scaling),
    *thresholds
)
prob_outliers_test, unc_outliers_test = convert_prob(
    convert_cdf(nll_inliers_valid, nll_outliers_test, dist=dist_cdf, scaling=scaling),
    *thresholds
)


fig, (ax1, ax2) = plt.subplots(2, 1)
for nll_, prob_ in zip(nll_inliers_train, prob_inliers_train):
    indices = np.argsort(nll_)
    ax1.plot(nll_[indices], prob_[indices], color="tab:blue", alpha=0.5)
    ax2.plot(
        nll_[indices],
        prob_[indices] * (1 - prob_[indices]),
        color="tab:blue",
        alpha=0.5,
    )

prob_inliers_test_mean = prob_inliers_test.mean(0)
prob_outliers_test_mean = prob_outliers_test.mean(0)

total_unc_inliers_test = get_pred_unc(prob_inliers_test, unc_inliers_test)
total_unc_outliers_test = get_pred_unc(prob_outliers_test, unc_outliers_test)

y_true, y_hard_pred, y_unc = get_y_results(
    prob_inliers_test_mean,
    prob_outliers_test_mean,
    total_unc_inliers_test,
    total_unc_outliers_test,
    p_threshold=p_threshold,
)

# evaluation
# need better evaluation
# 1 : f1 | unc
# 2 : auprc misclassification (TYPE 1 , TYPE 2, ALL)
# 3 : AUROC BINARY CLASSIFICATION

evaluate_mcc_f1_unc(y_true, y_hard_pred, y_unc[unc_type])
evaluate_unc_perf(y_true, y_hard_pred, y_unc[unc_type], verbose=True)
plot_unc_tptnfpfn(y_true, y_hard_pred, y_unc[unc_type])
plot_kde_auroc(
    prob_inliers_test_mean, prob_inliers_test_mean, prob_outliers_test_mean, mode="hist"
)

plt.figure()
plt.scatter(prob_inliers_test_mean, total_unc_inliers_test[unc_type])
plt.scatter(prob_outliers_test_mean, total_unc_outliers_test[unc_type])

plt.figure()
plt.scatter(total_unc_inliers_test["epistemic"], total_unc_inliers_test["aleatoric"])

# ==========================REVISE EVALUTION METRIC===============================


def calc_f1_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn)


def calc_mcc_score(tp, fp, fn, tn):
    mcc = ((tp * tn) - (fp * fn)) / (
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return mcc


def calc_perc_indices(indices_len, total_samples):
    perc = indices_len / total_samples
    return perc


def calc_precision(tp, fp):
    return tp / (tp + fp)


def calc_fdrate(tp, fp):
    fdr = fp / (fp + tp)
    return fdr


def calc_forate(fn, tn):
    forate = fn / (fn + tn)
    return forate


def calc_gmean_ss(tp, fp, fn, tn):
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return np.sqrt(tpr * tnr)


def calc_metrics(tp, fp, fn, tn, indices_len, total_samples):
    return {
        "f1": calc_f1_score(tp, fp, fn),
        "mcc": calc_mcc_score(tp, fp, fn, tn),
        "precision": calc_precision(tp, fp),
        "perc": calc_perc_indices(indices_len, total_samples),
        "gmean_ss": calc_gmean_ss(tp, fp, fn, tn),
        "forate": calc_forate(fn, tn),
        "fdr": calc_fdrate(tp, fp),
    }


def calc_performance_unc(y_scores_unc, y_true, y_hard_pred, unc_threshold=0):
    # specify retained and rejected indices
    retained_indices = np.argwhere(y_scores_unc <= unc_threshold)[:, 0]
    rejected_indices = np.argwhere(y_scores_unc > unc_threshold)[:, 0]

    rejected_metrics = {}

    # get their confusion matrix
    retained_matr = confusion_matrix(
        y_true[retained_indices], y_hard_pred[retained_indices]
    )
    rejected_matr = confusion_matrix(
        y_true[rejected_indices], y_hard_pred[rejected_indices]
    )

    # sanity check for existence of confusion matrix
    # if doesn't exist, then exit
    if len(retained_matr) > 1:
        tn, fp, fn, tp = retained_matr.ravel()
        retained_metrics = calc_metrics(
            tp, fp, fn, tn, len(retained_indices), len(y_scores_unc)
        )
        retained_metrics.update({"threshold": unc_threshold})

    else:
        return ()

    if len(rejected_matr) > 1:
        rj_tn, rj_fp, rj_fn, rj_tp = rejected_matr.ravel()
        rejected_metrics = calc_metrics(
            rj_tn, rj_fp, rj_fn, rj_tp, len(rejected_indices), len(y_scores_unc)
        )
        rejected_metrics.update({"threshold": unc_threshold})

    return retained_metrics, rejected_metrics


y_unc_scaled = np.round(y_unc[unc_type] * 4, 2)
unc_thresholds_ = np.unique(y_unc_scaled)
# unc_thresholds_ = np.concatenate((unc_thresholds_, np.array([1.01])))

# unc_thresholds_ = np.unique(np.round(y_unc[unc_type], 3))
unc_thresholds = []
error_uncs = []
retained_metrics = []
rejected_metrics = []

len(np.argwhere(y_unc_scaled <= 1.0)[:, 0]) / len(y_unc_scaled)

for unc_ in unc_thresholds_:
    # error_unc = calc_error_unc(y_unc[unc_type], y_true, y_hard_pred, unc_threshold=unc_)
    metrics = calc_performance_unc(
        y_unc_scaled, y_true, y_hard_pred, unc_threshold=unc_
    )
    if len(metrics) > 0:
        # print("reject")
        # print(unc_)
        retained_metrics_, rejected_metrics_ = metrics
        retained_metrics.append(retained_metrics_)
        rejected_metrics.append(rejected_metrics_)
        unc_thresholds.append(unc_)
    # if len(error_unc) > 0 and ~np.isnan(error_unc[0]) and ~np.isnan(error_unc[1]):
    #     error_uncs.append(error_unc)
    #     unc_thresholds.append(unc_)

retained_metrics__ = pd.DataFrame(retained_metrics)
rejected_metrics__ = pd.DataFrame(rejected_metrics)
unc_thresholds = np.array(unc_thresholds)

key = "mcc"

baselines = retained_metrics__.iloc[-1]
wmean_retained = ((retained_metrics__.T * retained_metrics__["perc"]).sum(1)/retained_metrics__["perc"].sum()).round(3)
wmean_rejected = ((rejected_metrics__.dropna().T * rejected_metrics__["perc"].dropna()).sum(1)/rejected_metrics__["perc"].dropna().sum()).round(3)

error_uncs = np.array(error_uncs)

# x_metric = error_uncs[:, 0]
# y_metric = 1 - error_uncs[:, 1]
x_metric = error_uncs[:, -1]
# x_metric = 1 - error_uncs[:, 1]
# y_metric = error_uncs[:, 1]
# y_metric = unc_thresholds
y_metric = error_uncs[:, 0]
# scores = []
# for i in range(1, len(x_metric)):
#     score = (x_metric[i] - x_metric[i - 1]) * y_metric[i]
#     scores.append(score)
# scores = np.array(scores)
# avg_score = scores.mean()

plt.figure()
plt.scatter(x_metric, y_metric)
plt.scatter(x_metric, error_uncs[:, 1])

# plt.scatter(x_metric_, y_metric_)
plt.legend(["RETAINED", "REJECTED"])


plt.figure()
plt.scatter(unc_thresholds, y_metric)

# plt.figure()

# auc_score = auc(x_metric, y_metric)
# auc_score = auc(y_metric, y_metric)

weighted_mean = (x_metric * y_metric).sum() / x_metric.sum()
best_line = y_metric.min()
baseline = y_metric.max()

# factor = weighted_mean / baseline

# print(avg_score)
# print(auc_score)
print((x_metric * y_metric).sum() / x_metric.sum())
print((y_metric.max(), y_metric.min()))


# unc_thresholds_ = np.unique(np.round(y_unc[unc_type], 3))
# unc_thresholds = []
# error_uncs = []
# for unc_ in unc_thresholds_:
#     error_unc = calc_error_unc(y_unc[unc_type], y_true, y_hard_pred, unc_threshold=unc_)
#     if len(error_unc) > 0:
#         error_uncs.append(error_unc)
#         unc_thresholds.append(unc_)
#         indices = np.argwhere(y_unc[unc_type] >= unc_)[:, 0]
#         conf_matr = confusion_matrix(y_true[indices], y_hard_pred[indices])
#         if len(conf_matr) > 1:
#             tn, fp, fn, tp = conf_matr.ravel()
#         with np.errstate(divide='ignore', invalid='ignore'):
#             fpr = fp / (fp + tn)
#             fdr = fp / (fp + tp)
#             fnr = fn / (fn + tp)
#             forate = fn / (fn + tn)
#             precision = tp / (tp + fp)
#             tpr = tp / (tp + fn)
#             tnr = tn / (tn + fp)
#             mcc = ((tp * tn) - (fp * fn)) / (
#                 np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
#             )
#             if np.isnan(mcc):
#                 mcc = 0
#             f1 = (2 * tp) / (2 * tp + fp + fn)
#             perc = len(indices) / len(y_unc[unc_type])
#             ba = (tpr + tnr) / 2
#             if

# (
#     y_metric.max() * x_metric[y_metric.argmax()]
#     + y_metric.min() * x_metric[y_metric.argmin()]
# ) / (x_metric.max() + x_metric.min())

# auc_f1_unc = auc(error_uncs[:, 0], 1 - error_uncs[:, 1])
#
# plt.figure()
# plt.scatter(error_uncs[:, 0], 1 - error_uncs[:, 1])
#
# from sklearn.metrics import roc_curve
#
# fpr, tpr, thresholds = roc_curve(y_true, y_hard_pred)
# auc(x_metric, y_metric)


# ========================FIT BETA DISTRIBUTION===========================


samples = nll_inliers_train[0]
test_samples = nll_inliers_test[0]

# samples = (samples - np.mean(samples)) / (np.std(samples) * np.sqrt(2))
# test_samples = (test_samples - np.mean(samples)) / (np.std(samples) * np.sqrt(2))

gamma_params = gamma.fit(samples)


rv1 = gamma(*gamma.fit(samples))
rv2 = lognorm(*lognorm.fit(samples))
rv3 = norm(*norm.fit(samples))
rv4 = uniform(*uniform.fit(samples))
rv6 = expon(*expon.fit(samples))

pre_erf_score = (test_samples - np.mean(samples)) / (np.std(samples) * np.sqrt(2))
# pre_erf_score = test_samples
rv5 = np.clip(erf(pre_erf_score), 0, 1)
# rv5 = (erf(pre_erf_score)+1)*0.5


plt.figure()
plt.hist(samples, density=True, alpha=0.5)
plt.scatter(test_samples, rv1.pdf(test_samples))
plt.scatter(test_samples, rv2.pdf(test_samples))
plt.scatter(test_samples, rv3.pdf(test_samples))
plt.scatter(test_samples, rv4.pdf(test_samples))
plt.scatter(test_samples, rv6.pdf(test_samples))

plt.figure()
plt.scatter(test_samples, rv1.cdf(test_samples))
plt.scatter(test_samples, rv2.cdf(test_samples))
plt.scatter(test_samples, rv3.cdf(test_samples) + 0.01, color="green")
plt.scatter(test_samples, rv4.cdf(test_samples))
plt.scatter(test_samples, rv6.cdf(test_samples))
plt.scatter(test_samples, rv5, color="black")

plt.scatter(test_samples, ECDF(samples)(test_samples))
plt.legend(["GAMMA", "LOGNORM", "NORM", "UNI", "ERF", "ECDF", "EXPON"])

plt.figure()
for dist in [rv1, rv2, rv3, rv4, rv6]:
    scaled_res = np.clip(
        (dist.cdf(test_samples) - dist.cdf(samples.mean()))
        / (1 - dist.cdf(samples.mean())),
        0,
        1,
    )
    plt.scatter(test_samples, scaled_res)
plt.scatter(test_samples, rv5, color="black")

scaled_ECDF_model = ECDF(samples)
scaled_ECDF = np.clip(
    (scaled_ECDF_model(test_samples) - scaled_ECDF_model(samples.mean()))
    / (1 - scaled_ECDF_model(samples.mean())),
    0,
    1,
)

plt.scatter(test_samples, scaled_ECDF + 0.01)
plt.legend(["GAMMA", "LOGNORM", "NORM", "UNI", "EXPON", "ERF", "ECDF"])

# ========================ESS=============================================

# filter out high unc. samples

unc_thresholds = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
unc_threshold = 1.0
filtered_test = prob_inliers_test_mean[
    np.argwhere(total_unc_inliers_test[unc_type] * 4 <= unc_threshold)[:, 0]
]

filtered_outlier = prob_outliers_test_mean[
    np.argwhere(total_unc_outliers_test[unc_type] * 4 <= unc_threshold)[:, 0]
]


filtereds = []
for unc_threshold in unc_thresholds:
    filtered_test = prob_inliers_test_mean[
        np.argwhere(total_unc_inliers_test[unc_type] * 4 <= unc_threshold)[:, 0]
    ]
    filtered_outlier = prob_outliers_test_mean[
        np.argwhere(total_unc_outliers_test[unc_type] * 4 <= unc_threshold)[:, 0]
    ]

    # filtered_test = prob_inliers_test_mean[np.argwhere(total_unc_inliers_test * 4 >= unc_threshold)[:, 0]]
    # filtered_outlier = prob_outliers_test_mean[np.argwhere(total_unc_outliers_test * 4 >= unc_threshold)[:, 0]]

    if len(filtered_test) > 0 and len(filtered_outlier) > 0:
        auroc_filtered = calc_auroc(filtered_test, filtered_outlier)
        filtereds.append(auroc_filtered)

print(filtereds)
