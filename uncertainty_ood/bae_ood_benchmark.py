from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
import os
import numpy as np
from scipy.stats import spearmanr

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import DenseLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood.calc_uncertainty_ood import calc_ood_threshold, convert_hard_predictions
from util.convergence import bae_fit_convergence, plot_convergence
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.metrics import matthews_corrcoef
from util.evaluate_ood import plot_histogram_ood, plot_kde_ood


# random_seed = 987
random_seed = 1233333

bae_set_seed(random_seed)

use_cuda = True
clip_data_01 = False
activation = "leakyrelu"
last_activation = "sigmoid" # tanh
likelihood = "gaussian" # gaussian
train_size = 0.7
num_samples = 5
# multi_perc_thresholds = np.arange(85,100)
# multi_perc_thresholds = np.arange(75,100)
multi_perc_thresholds = np.arange(95,100)
perc_threshold = 97.5


#==============PREPARE DATA==========================
base_folder = "od_benchmark"
mat_file_list = os.listdir(base_folder)

mat_file = mat_file_list[-1]
mat = loadmat(os.path.join(base_folder, mat_file))

X = mat['X']
y = mat['y'].ravel()

x_outliers, x_inliers = get_outliers_inliers(X, y)

x_inliers_train, x_inliers_test = train_test_split(x_inliers, train_size=train_size,shuffle=True, random_state=random_seed)

#=================SCALER=========================
scaler = MinMaxScaler()
scaler = scaler.fit(x_inliers_train)
x_inliers_train = scaler.transform(x_inliers_train)
x_inliers_test = scaler.transform(x_inliers_test)
x_outliers = scaler.transform(x_outliers)

if clip_data_01:
    x_inliers_train = np.clip(x_inliers_train, 0, 1)
    x_inliers_test = np.clip(x_inliers_test, 0, 1)
    x_outliers = np.clip(x_outliers, 0, 1)

#=================DEFINE BAE========================

input_dim = x_inliers_train.shape[-1]
# activation = "sigmoid"
weight_decay = 0.0001
lr = 0.025

# encoder_nodes = [input_dim*2,20]
# encoder_nodes = [input_dim*4,int(input_dim/3)]
# encoder_nodes = [input_dim*4,100]
# encoder_nodes = [input_dim*4,80]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*8,input_dim]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*6]
# encoder_nodes = [input_dim*2,input_dim*4,2]
# encoder_nodes = [input_dim*8,input_dim]
# encoder_nodes = [input_dim*8,input_dim*4,input_dim*2]
# encoder_nodes = [input_dim*8,input_dim*4,input_dim]
encoder_nodes = [input_dim*8,input_dim*4,int(input_dim*5)]
# encoder_nodes = [input_dim*8,int(input_dim/2)]

encoder = Encoder([DenseLayers(input_size=input_dim,
                               architecture=encoder_nodes[:-1],
                               output_size=encoder_nodes[-1], activation=activation,last_activation=activation
                               )])

# specify decoder-mu
decoder_mu = infer_decoder(encoder,activation=activation,last_activation=last_activation) #symmetrical to encoder

# combine them into autoencoder
autoencoder = Autoencoder(encoder, decoder_mu)

# convert into BAE-Ensemble
bae_ensemble = BAE_Ensemble(autoencoder=autoencoder,
                            anchored=True,
                            weight_decay=weight_decay,
                            num_samples=num_samples,
                            likelihood=likelihood,
                            learning_rate=lr,
                            verbose=False,
                            use_cuda = use_cuda
                            )

#===============FIT BAE CONVERGENCE===========================
train_loader = convert_dataloader(x_inliers_train, batch_size=250, shuffle=True)
run_auto_lr_range_v2(train_loader, bae_ensemble, run_full=True, window_size=1, num_epochs=15)
# bae_ensemble.fit(train_loader,num_epochs=150)

num_epochs_per_cycle = 20
fast_window = num_epochs_per_cycle
slow_window = fast_window*10
n_stop_points = 3
cvg = 0


while(cvg == 0):
    _, cvg = bae_fit_convergence(bae_ensemble=bae_ensemble, x=train_loader,
                                 num_epoch=num_epochs_per_cycle,
                                 fast_window=fast_window,
                                 slow_window=slow_window,
                                 n_stop_points=n_stop_points
                                 )

fig, ax = plt.subplots(1,1)
plot_convergence(losses=bae_ensemble.losses,
                 fast_window=fast_window,
                 slow_window=slow_window,
                 n_stop_points= n_stop_points,
                 ax=ax)

#===============PREDICT BAE==========================

nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=["se"])
nll_inliers_test = bae_ensemble.predict_samples(x_inliers_test, select_keys=["se"])
nll_outliers_test = bae_ensemble.predict_samples(x_outliers, select_keys=["se"])

nll_inliers_train_mean = nll_inliers_train.mean(0)[0].mean(-1)
nll_inliers_test_mean = nll_inliers_test.mean(0)[0].mean(-1)
nll_outliers_test_mean = nll_outliers_test.mean(0)[0].mean(-1)

nll_inliers_train_var = nll_inliers_train.var(0)[0].mean(-1)
nll_inliers_test_var = nll_inliers_test.var(0)[0].mean(-1)
nll_outliers_test_var = nll_outliers_test.var(0)[0].mean(-1)

def plot_kde_auroc(nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean, mode="kde"):
    y_true = np.concatenate((np.zeros(nll_inliers_test_mean.shape[0]),
                             np.ones(nll_outliers_test_mean.shape[0])))
    y_scores = np.concatenate((nll_inliers_test_mean, nll_outliers_test_mean))

    auroc = roc_auc_score(y_true, y_scores)
    auroc_text = "AUROC: {:.3f}".format(auroc, 2)

    fig, ax = plt.subplots(1, 1)

    if mode == "kde":
        plot_kde_ood(nll_inliers_train_mean,
                     nll_inliers_test_mean,
                     nll_outliers_test_mean,
                     fig=fig,
                     ax=ax)

    if mode == "hist":
        plot_histogram_ood(nll_inliers_train_mean,
                           nll_inliers_test_mean,
                           nll_outliers_test_mean,
                           fig=fig,
                           ax=ax,
                           )

    ax.set_title(auroc_text)


plot_kde_auroc(nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean)
plot_kde_auroc(nll_inliers_train_var, nll_inliers_test_var, nll_outliers_test_var)


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
        y_nll_mean =y_nll_mean.mean(-1)
        y_nll_var = y_nll_var.mean(-1)
        y_recon_var = y_recon_var.mean(-1)

    return {"nll_mean": y_nll_mean,
            "nll_var":y_nll_var,
            "recon_var":y_recon_var,
            "latent_mean": y_latent_mean,
            "latent_var": y_latent_var,
            # "latent_weighted":y_latent_weighted
            }

def bae_predict_ood_v1(bae_ensemble, x_train, x_test, keys=["nll_mean","nll_var"], perc_threshold=99):
    """
    Combine nll_mean and nll_var method
    """
    preds_train = bae_pred_all(bae_ensemble, x_train, return_mean=True)
    preds_test = bae_pred_all(bae_ensemble, x_test, return_mean=True)
    thresholds = {key:calc_ood_threshold(training_scores=preds_train[key],perc_threshold=perc_threshold) for key in keys}
    hard_preds = np.array([convert_hard_predictions(test_scores=preds_test[key], ood_threshold=thresholds[key]) for key in
                  keys])

    return hard_preds

def bae_predict_ood_v2(bae_ensemble, x_train, x_test, perc_threshold=99):
    """
    Apply Threshold on each samples. Threshold obtained from each BAE sample's training scores.
    """

    preds_train = bae_ensemble.predict_samples(x_train, select_keys=["se"]).mean(-1)[:,0]
    preds_test = bae_ensemble.predict_samples(x_test, select_keys=["se"]).mean(-1)[:,0]

    thresholds = [calc_ood_threshold(training_scores=preds_train_i,perc_threshold=perc_threshold) for preds_train_i in preds_train]
    hard_preds = np.array([convert_hard_predictions(test_scores=preds_test_i, ood_threshold=thresholds_i) for preds_test_i,thresholds_i in
                  zip(preds_test,thresholds)])

    return hard_preds


hard_preds_inliers_train_v1 = bae_predict_ood_v1(bae_ensemble, x_inliers_train, x_inliers_train, keys=["nll_mean","nll_var"], perc_threshold = perc_threshold)
hard_preds_inliers_test_v1 = bae_predict_ood_v1(bae_ensemble, x_inliers_train, x_inliers_test, keys=["nll_mean","nll_var"], perc_threshold = perc_threshold)
hard_preds_outlier_test_v1 = bae_predict_ood_v1(bae_ensemble, x_inliers_train, x_outliers, keys=["nll_mean","nll_var"], perc_threshold = perc_threshold)

plot_kde_auroc(hard_preds_inliers_train_v1.mean(0),
               hard_preds_inliers_test_v1.mean(0),
               hard_preds_outlier_test_v1.mean(0),
               mode="hist"
               )

hard_preds_inliers_train_v2 = bae_predict_ood_v2(bae_ensemble, x_inliers_train, x_inliers_train, perc_threshold = perc_threshold)
hard_preds_inliers_test_v2 = bae_predict_ood_v2(bae_ensemble, x_inliers_train, x_inliers_test, perc_threshold = perc_threshold)
hard_preds_outlier_test_v2 = bae_predict_ood_v2(bae_ensemble, x_inliers_train, x_outliers, perc_threshold = perc_threshold)

plot_kde_auroc(hard_preds_inliers_train_v2.mean(0),
               hard_preds_inliers_test_v2.mean(0),
               hard_preds_outlier_test_v2.mean(0),
               mode="hist")

def evaluate_f1_score(hard_preds_inliers_test, hard_preds_outlier_test):
    y_true = np.concatenate((np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
                             np.ones(hard_preds_outlier_test.mean(0).shape[0])))
    y_scores = np.concatenate((hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0)))

    threshold = 0.5
    y_scores[np.argwhere(y_scores>=threshold)] = 1
    y_scores[np.argwhere(y_scores<threshold)] = 0

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

def evaluate_avgprc_misclass(hard_preds_inliers_test,hard_preds_outlier_test):
    y_true = np.concatenate((np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
                             np.ones(hard_preds_outlier_test.mean(0).shape[0])))

    y_scores_unc = np.concatenate((hard_preds_inliers_test.std(0)*2, hard_preds_outlier_test.std(0)*2))

    y_scores = np.concatenate((hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0)))

    threshold = 0.5
    y_scores[np.argwhere(y_scores >= threshold)] = 1
    y_scores[np.argwhere(y_scores < threshold)] = 0

    error = np.abs(y_scores - y_true)

    avgprc = average_precision_score(error, y_scores_unc)
    print("AVG-PRC: {:.2f} , BASELINE: {:.2f}, AVG-PRC-RATIO: {:.2f}".format(avgprc,error.mean(), avgprc/error.mean()))

    return avgprc

def evaluate_auprc_misclass(hard_preds_inliers_test,hard_preds_outlier_test):
    y_true = np.concatenate((np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
                             np.ones(hard_preds_outlier_test.mean(0).shape[0])))

    y_scores_unc = np.concatenate((hard_preds_inliers_test.std(0)*2, hard_preds_outlier_test.std(0)*2))

    y_scores = np.concatenate((hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0)))

    threshold = 0.5
    y_scores[np.argwhere(y_scores >= threshold)] = 1
    y_scores[np.argwhere(y_scores < threshold)] = 0

    error = np.abs(y_scores - y_true)

    precision, recall, thresholds = precision_recall_curve(error, y_scores_unc)
    auprc = auc(recall, precision)

    print("AUPRC: {:.2f} , BASELINE: {:.2f}, AUPRC-RATIO: {:.2f}".format(auprc,error.mean(), auprc/error.mean()))

    return auprc

# precision, recall, thresholds = precision_recall_curve(testy, probs)
# auc = auc(recall, precision)


avgprc_v1 = evaluate_avgprc_misclass(hard_preds_inliers_test_v1,hard_preds_outlier_test_v1)
avgprc_v2 = evaluate_avgprc_misclass(hard_preds_inliers_test_v2,hard_preds_outlier_test_v2)

auprc_v1 = evaluate_auprc_misclass(hard_preds_inliers_test_v1,hard_preds_outlier_test_v1)
auprc_v2 = evaluate_auprc_misclass(hard_preds_inliers_test_v2,hard_preds_outlier_test_v2)


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

#=============MULTIPLE THRESHOLDS==============


def multi_hard_predict_v1(bae_ensemble, x_inliers_train, x_test, keys=["nll_mean","nll_var"], thresholds=np.arange(80,100)):
    outputs = [bae_predict_ood_v1(bae_ensemble, x_inliers_train, x_test, keys=keys, perc_threshold = perc_threshold_) for perc_threshold_ in thresholds]
    outputs = np.concatenate(outputs, axis=0)
    return outputs

def multi_hard_predict_v2(bae_ensemble, x_inliers_train, x_test, thresholds=np.arange(80,100)):
    outputs = [bae_predict_ood_v2(bae_ensemble, x_inliers_train, x_test, perc_threshold = perc_threshold_) for perc_threshold_ in thresholds]
    outputs = np.concatenate(outputs, axis=0)
    return outputs

def evaluate_unc(hard_preds_inliers_test, hard_preds_outlier_test, uncertainty_threshold=0.95, decision_threshold=0.5):
    y_true = np.concatenate((np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
                             np.ones(hard_preds_outlier_test.mean(0).shape[0])))

    y_scores_unc = np.concatenate((hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2))

    y_scores = np.concatenate((hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0)))

    decision_threshold = 0.5
    y_scores[np.argwhere(y_scores >= decision_threshold)] = 1
    y_scores[np.argwhere(y_scores < decision_threshold)] = 0
    y_scores = y_scores.astype(int)

    f1_score_high_unc = f1_score(y_true[np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0]],
                                 y_scores[np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0]])
    f1_score_low_unc = f1_score(y_true[np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0]],
                                y_scores[np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0]])
    f1_score_all = f1_score(y_true, y_scores)

    perc_uncertain = len(np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0]) / y_scores_unc.shape[0]
    perc_certain = len(np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0]) / y_scores_unc.shape[0]

    print("HIGH UNC: {:.2f}".format((f1_score_high_unc)))
    print("LOW UNC: {:.2f}".format((f1_score_low_unc)))
    print("W/O UNC: {:.2f}".format((f1_score_all)))
    print("% UNC: {:.2f}".format((perc_uncertain)))
    print("% CER: {:.2f}".format((perc_certain)))

    return f1_score_high_unc, f1_score_low_unc, perc_uncertain, perc_certain


print("F1-SCORE MULTI THRESHOLD")

multihard_preds_inliers_train_v1 = multi_hard_predict_v1(bae_ensemble, x_inliers_train, x_inliers_train, keys=["nll_mean","nll_var"], thresholds=multi_perc_thresholds)
multihard_preds_inliers_test_v1 = multi_hard_predict_v1(bae_ensemble, x_inliers_train, x_inliers_test, keys=["nll_mean","nll_var"], thresholds=multi_perc_thresholds)
multihard_preds_outlier_test_v1 = multi_hard_predict_v1(bae_ensemble, x_inliers_train, x_outliers, keys=["nll_mean","nll_var"], thresholds=multi_perc_thresholds)

multihard_preds_inliers_train_v2 = multi_hard_predict_v2(bae_ensemble, x_inliers_train, x_inliers_train, thresholds=multi_perc_thresholds)
multihard_preds_inliers_test_v2 = multi_hard_predict_v2(bae_ensemble, x_inliers_train, x_inliers_test, thresholds=multi_perc_thresholds)
multihard_preds_outlier_test_v2 = multi_hard_predict_v2(bae_ensemble, x_inliers_train, x_outliers, thresholds=multi_perc_thresholds)

plot_kde_auroc(multihard_preds_inliers_train_v1.mean(0),
               multihard_preds_inliers_test_v1.mean(0),
               multihard_preds_outlier_test_v1.mean(0),
               mode="hist"
               )

plot_kde_auroc(multihard_preds_inliers_train_v2.mean(0),
               multihard_preds_inliers_test_v2.mean(0),
               multihard_preds_outlier_test_v2.mean(0),
               mode="hist"
               )

f1_score_v1 = evaluate_f1_score(multihard_preds_inliers_test_v1, multihard_preds_outlier_test_v1)
f1_score_v2 = evaluate_f1_score(multihard_preds_inliers_test_v2, multihard_preds_outlier_test_v2)

avgprc_v1 = evaluate_avgprc_misclass(multihard_preds_inliers_test_v1,multihard_preds_outlier_test_v1)
avgprc_v2 = evaluate_avgprc_misclass(multihard_preds_inliers_test_v2,multihard_preds_outlier_test_v2)

auprc_v1 = evaluate_auprc_misclass(multihard_preds_inliers_test_v1,multihard_preds_outlier_test_v1)
auprc_v2 = evaluate_auprc_misclass(multihard_preds_inliers_test_v2,multihard_preds_outlier_test_v2)


#==========================F1 SCORE | UNC ==============================

uncertainty_threshold = 0.5
evaluate_unc(multihard_preds_inliers_test_v1, multihard_preds_outlier_test_v1, uncertainty_threshold=uncertainty_threshold)
print("----------------")
evaluate_unc(multihard_preds_inliers_test_v2, multihard_preds_outlier_test_v2, uncertainty_threshold=uncertainty_threshold)


#=========================DEBUG: PLOT LATENT============================
def predict_latent_ex(bae_ensemble, x, n_ae=1):
    y_latent_samples = bae_ensemble.predict_latent_samples(x)
    y_latent_mean = y_latent_samples[n_ae]
    return y_latent_mean

def predict_latent_pca(bae_ensemble, x):
    y_latent_mean,_ = bae_ensemble.predict_latent(x, transform_pca=True)
    return y_latent_mean

latent_inliers_train_mean = bae_pred_all(bae_ensemble, x_inliers_train, return_mean=False)["latent_mean"]
latent_inliers_test_mean = bae_pred_all(bae_ensemble, x_inliers_test, return_mean=False)["latent_mean"]
latent_outliers_test_mean = bae_pred_all(bae_ensemble, x_outliers, return_mean=False)["latent_mean"]

# latent_inliers_train_mean = bae_pred_all(bae_ensemble, x_inliers_train, return_mean=False)["latent_var"]
# latent_inliers_test_mean = bae_pred_all(bae_ensemble, x_inliers_test, return_mean=False)["latent_var"]
# latent_outliers_test_mean = bae_pred_all(bae_ensemble, x_outliers, return_mean=False)["latent_var"]

# n_ae = 7
# latent_inliers_train_mean = predict_latent_ex(bae_ensemble, x_inliers_train, n_ae=n_ae)
# latent_inliers_test_mean = predict_latent_ex(bae_ensemble, x_inliers_test, n_ae=n_ae)
# latent_outliers_test_mean = predict_latent_ex(bae_ensemble, x_outliers, n_ae=n_ae)

latent_inliers_train_pca = predict_latent_pca(bae_ensemble, x_inliers_train)
latent_inliers_test_pca = predict_latent_pca(bae_ensemble, x_inliers_test)
latent_outliers_test_pca = predict_latent_pca(bae_ensemble, x_outliers)

plt.figure()
for sample in latent_inliers_train_mean:
    plt.plot(sample, alpha=0.5, color="tab:blue")

for sample in latent_inliers_test_mean:
    plt.plot(sample, alpha=0.5, color="tab:green")

for sample in latent_outliers_test_mean:
    plt.plot(sample, alpha=0.5, color="tab:orange")

plt.figure()
plt.scatter(latent_inliers_train_pca[:,0], latent_inliers_train_pca[:,1], alpha=0.5, color="tab:blue")
plt.scatter(latent_inliers_test_pca[:,0], latent_inliers_test_pca[:,1], alpha=0.5, color="tab:green")
plt.scatter(latent_outliers_test_pca[:,0], latent_outliers_test_pca[:,1], alpha=0.5, color="tab:orange")

#========================================

from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder_torch import AutoEncoder


# train kNN detector

clf = KNN()
# clf = AutoEncoder(epochs=500)
# clf = OCSVM()

def evaluate_clf(clf, x_inliers_train, x_inliers_test, x_outliers):

    clf.fit(x_inliers_train)
    y_inliers_pred = clf.predict(x_inliers_test)
    y_outliers_pred = clf.predict(x_outliers)

    y_true = np.concatenate((np.zeros(y_inliers_pred.shape[0]),
                             np.ones(y_outliers_pred.shape[0])))

    f1_score_ = f1_score(y_true, np.concatenate((y_inliers_pred,y_outliers_pred),axis=0))
    mcc_score_ = matthews_corrcoef(y_true, np.concatenate((y_inliers_pred,y_outliers_pred),axis=0))

    res = {"f1": f1_score_, "mcc":mcc_score_}
    print(res)
    return res

# latent_scaler = MinMaxScaler()
# latent_scaler = StandardScaler()
# latent_scaler.fit(x_inliers_train)
pyod_ori = evaluate_clf(clf, x_inliers_train, x_inliers_test, x_outliers)
# pyod_ori = evaluate_clf(clf, latent_scaler.transform(x_inliers_train), latent_scaler.transform(x_inliers_test), latent_scaler.transform(x_outliers))
#
# pyod_latent = evaluate_clf(clf,
#                            latent_scaler.fit_transform(latent_inliers_train_mean),
#                            latent_scaler.transform(latent_inliers_test_mean),
#                            latent_scaler.transform(latent_outliers_test_mean))

pyod_latent = evaluate_clf(clf, latent_inliers_train_mean, latent_inliers_test_mean, latent_outliers_test_mean)
pyod_latent_pca = evaluate_clf(clf, latent_inliers_train_pca, latent_inliers_test_pca, latent_outliers_test_pca)

#=============================================
from sklearn.metrics import accuracy_score
# accuracy_score(y_true, y_pred)

def evaluate_unc_(hard_preds_inliers_test, hard_preds_outlier_test, uncertainty_threshold=0.95, decision_threshold=0.5):
    y_true = np.concatenate((np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
                             np.ones(hard_preds_outlier_test.mean(0).shape[0])))

    y_scores_unc = np.concatenate((hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2))

    y_scores = np.concatenate((hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0)))

    y_scores[np.argwhere(y_scores >= decision_threshold)] = 1
    y_scores[np.argwhere(y_scores < decision_threshold)] = 0
    y_hard_pred = y_scores.astype(int)

    error = (y_hard_pred-y_true)

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


#=================FPR AND FNR ==============
from sklearn.metrics import confusion_matrix

def get_y_unc(hard_preds_inliers_test, hard_preds_outlier_test, uncertainty_threshold=0.95, decision_threshold=0.5):
    y_true = np.concatenate((np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
                             np.ones(hard_preds_outlier_test.mean(0).shape[0])))

    y_scores_unc = np.concatenate((hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2))

    y_scores = np.concatenate((hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0)))

    y_scores[np.argwhere(y_scores >= decision_threshold)] = 1
    y_scores[np.argwhere(y_scores < decision_threshold)] = 0
    y_hard_pred = y_scores.astype(int)

    # error = (y_hard_pred-y_true)

    return y_scores_unc, y_hard_pred, y_true

def calc_error_unc(y_scores_unc, y_true, y_hard_pred, unc_threshold=0):
    indices = np.argwhere(y_scores_unc <= unc_threshold)[:,0]
    tn, fp, fn, tp = confusion_matrix(y_true[indices], y_hard_pred[indices]).ravel()
    fpr = fp/(fp+tn)
    fdr = fp / (fp + tp)
    fnr = fn/(fn+tp)
    forate = fn/(fn+tn)
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    mcc = ((tp*tn)-(fp*fn))/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    if np.isnan(mcc):
        mcc = 0
    f1 = tp/(tp+0.5*(fp+fn))
    perc = len(indices)/len(y_scores_unc)
    ba = (tpr+tnr)/2

    # return fpr, fdr, fnr, forate, tpr, tnr, mcc, f1, perc
    return fpr, fdr, fnr, forate, tpr, tnr, mcc, f1, perc, ba
    # return fpr, fdr, fnr, tpr, tnr, mcc, f1, perc

# unc_thresholds = [0.1,0.5,0.8,1.]
y_scores_unc, y_hard_pred, y_true = get_y_unc(multihard_preds_inliers_test_v1,
                        multihard_preds_outlier_test_v1)

# y_scores_unc, y_hard_pred, y_true = get_y_unc(multihard_preds_inliers_test_v2,
#                         multihard_preds_outlier_test_v2)

unc_thresholds = np.unique(np.round(y_scores_unc,3))
error_uncs = np.array([calc_error_unc(y_scores_unc, y_true, y_hard_pred, unc_threshold=unc_) for unc_ in unc_thresholds])
# error_uncs[:,0] /= error_uncs[0,0]
# error_uncs[:,1] /= error_uncs[0,1]
# error_uncs[:,2] /= error_uncs[0,2]
# error_uncs[:,3] /= error_uncs[0,3]
# error_uncs[:,3] /= error_uncs[0,3]
# error_uncs[:,3] /= error_uncs[0,3]

# error_uncs /= error_uncs[0,:]

fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1)
# ax1.plot(unc_thresholds, error_uncs[:,0], '-o')
# ax2.plot(unc_thresholds, error_uncs[:,1], '-o')
# ax3.plot(unc_thresholds, error_uncs[:,2], '-o')
# ax4.plot(unc_thresholds, error_uncs[:,3], '-o')
ax1.plot(unc_thresholds, error_uncs[:,0], '-o')
ax2.plot(unc_thresholds, error_uncs[:,1], '-o')
ax3.plot(unc_thresholds, error_uncs[:,3], '-o')
ax4.plot(unc_thresholds, error_uncs[:,5], '-o')
ax5.plot(unc_thresholds, error_uncs[:,6], '-o')
ax6.plot(unc_thresholds, error_uncs[:,7], '-o')
ax7.plot(unc_thresholds, error_uncs[:,8], '-o')

# ax8.plot(unc_thresholds, error_uncs[:,8]*error_uncs[:,6], '-o')

spman_mcc = spearmanr(unc_thresholds, error_uncs[:,6])[0]
spman_f1 = spearmanr(unc_thresholds, error_uncs[:,7])[0]

mcc_diff =  error_uncs[0,6]-error_uncs[-1,6]
f1_diff =  error_uncs[0,7]-error_uncs[-1,7]

print("SPMAN MCC : {:.3f}".format(spman_mcc))
print("SPMAN F1 : {:.3f}".format(spman_f1))
print("DIFF MCC : {:.3f}".format(mcc_diff))
print("DIFF F1 : {:.3f}".format(f1_diff))
print("PERC HIGH : {:.3f}".format(error_uncs[0,8]))
print("MCC HIGH : {:.3f}".format(error_uncs[0,6]))
print("MCC LOW : {:.3f}".format(error_uncs[-1,6]))
print("F1 HIGH : {:.3f}".format(error_uncs[0,7]))
print("F1 LOW : {:.3f}".format(error_uncs[-1,7]))
print("MCC MEAN : {:.3f}".format(error_uncs[:,6].mean()))
print("F1 MEAN : {:.3f}".format(error_uncs[:,7].mean()))
print("PERC MEAN : {:.3f}".format(error_uncs[:,8].mean()))

