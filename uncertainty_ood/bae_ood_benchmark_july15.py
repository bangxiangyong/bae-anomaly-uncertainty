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
    roc_curve,
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
from baetorch.baetorch.util.data_model_manager import DataModelManager
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

from util.exp_manager import ExperimentManager
from util.uncertainty import (
    convert_prob,
    convert_cdf,
    get_pred_unc,
    get_y_results,
    evaluate_mcc_f1_unc,
    evaluate_unc_perf,
    plot_unc_tptnfpfn,
    plot_kde_auroc,
    evaluate_error_unc,
    calc_performance_unc,
    get_optimal_threshold,
    convert_hard_pred,
    get_pred_optimal,
    calc_hard_metrics,
    calc_metrics2,
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
random_seed = 22

bae_set_seed(random_seed)

use_cuda = True
clip_data_01 = False
activation = "leakyrelu"
last_activation = "tanh"  # tanh
likelihood = "gaussian"  # gaussian
train_size = 0.80
num_samples = 5
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
mat_file = mat_file_list[1]
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
# encoder_nodes = [input_dim * 8, input_dim * 4, input_dim * 2]
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
    homoscedestic_mode="none",
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
auroc_threshold = 0.85

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


# y_true, y_hard_pred, y_unc = predict_ood_unc(
#     bae_ensemble,
#     x_inliers_valid,
#     x_inliers_test,
#     x_outliers_test,
#     p_threshold=0.5,
# )

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

dist_cdf = "expon"  # gamma, norm , uniform, expon, "ecdf"
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


prob_inliers_test_mean = prob_inliers_test.mean(0)
prob_outliers_test_mean = prob_outliers_test.mean(0)

total_unc_inliers_test = get_pred_unc(prob_inliers_test, unc_inliers_test)
total_unc_outliers_test = get_pred_unc(prob_outliers_test, unc_outliers_test)

y_true, y_hard_pred, y_unc, y_soft_pred = get_y_results(
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

y_unc_scaled = np.round(y_unc[unc_type] * 4, 2)
unc_thresholds_ = np.unique(y_unc_scaled)

unc_thresholds = []
error_uncs = []
retained_metrics = []
with np.errstate(divide="ignore", invalid="ignore"):
    for unc_ in unc_thresholds_:
        metrics = calc_performance_unc(
            y_unc_scaled,
            y_true,
            y_hard_pred,
            y_soft_pred=y_soft_pred,
            unc_threshold=unc_,
        )
        if len(metrics) > 0:
            retained_metrics_ = metrics
            retained_metrics.append(retained_metrics_)

retained_metrics = pd.DataFrame(retained_metrics)
res_wmean = (
    (retained_metrics.T * retained_metrics["perc"]).sum(1)
    / retained_metrics["perc"].sum()
).round(3)

res_spman = {}
perc_retained = retained_metrics["perc"].values
for key in retained_metrics.columns:
    sel_col = retained_metrics[key].values
    sp_ = spearmanr(perc_retained, sel_col)
    res_spman.update({key: -1 * sp_[0] if sp_[1] <= 0.05 else 0})
res_spman = pd.DataFrame([res_spman])

aupr_unc_metrics = pd.DataFrame(
    [evaluate_error_unc(y_true, y_hard_pred, y_unc[unc_type], verbose=True)]
).T


auroc_prob = np.round(calc_auroc(prob_inliers_test_mean, prob_outliers_test_mean), 3)

auroc_base = np.round(
    calc_auroc(nll_inliers_test.mean(0), nll_outliers_test.mean(0)), 3
)

print("AUROC: {:.3f}".format(auroc_prob))
print("AUROC-BASE: {:.3f}".format(auroc_base))

y_true, y_hard_optim = get_pred_optimal(
    prob_inliers_test_mean,
    prob_outliers_test_mean,
)

res_baselines = pd.DataFrame(
    [calc_metrics2(y_true, y_hard_pred, y_soft_pred=y_soft_pred)]
).T
res_optim = pd.DataFrame([calc_metrics2(y_true, y_hard_optim)]).T

print(aupr_unc_metrics)
print("---W-MEAN---")
print(res_wmean)
print("---BASELINE---")
print(res_baselines)
print("---OPTIM---")
print(res_optim)

exp_man = ExperimentManager(lookup_table_file="experiment_run1.csv")


exp_params = {
    "architecture": str([20, 20, 10]),
    "random_seed": random_seed,
    "dataset": mat_file,
}
exp_params.update(calc_metrics2(y_true, y_hard_pred))

exp_man.update_csv(exp_params)

plot_kde_auroc(prob_inliers_test_mean, prob_inliers_test_mean, prob_outliers_test_mean)

plt.scatter(
    prob_inliers_test_mean, prob_inliers_test_mean * (1 - prob_inliers_test_mean)
)

# fig, (ax1, ax2) = plt.subplots(2, 1)
# for nll_, prob_ in zip(nll_inliers_train, prob_inliers_train):
#     indices = np.argsort(nll_)
#     ax1.plot(nll_[indices], prob_[indices], color="tab:blue", alpha=0.5)
#     ax2.plot(
#         nll_[indices],
#         prob_[indices] * (1 - prob_[indices]),
#         color="tab:blue",
#         alpha=0.5,
#     )
