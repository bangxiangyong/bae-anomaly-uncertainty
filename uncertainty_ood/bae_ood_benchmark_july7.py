from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
import os
import numpy as np
from scipy.special import erf
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from statsmodels.distributions import ECDF

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import DenseLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood.calc_uncertainty_ood import calc_ood_threshold, convert_hard_predictions
from util.convergence import bae_fit_convergence, plot_convergence, bae_semi_fit_convergence
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.metrics import matthews_corrcoef
from util.evaluate_ood import plot_histogram_ood, plot_kde_ood
from statsmodels.stats.stattools import medcouple
from sklearn.metrics import confusion_matrix
from scipy.stats import beta, gamma, lognorm, norm, uniform, expon


def calc_auroc(nll_inliers_train_mean, nll_inliers_valid_mean):
    y_true = np.concatenate((np.zeros(nll_inliers_train_mean.shape[0]),
                             np.ones(nll_inliers_valid_mean.shape[0])))
    y_scores = np.concatenate((nll_inliers_train_mean, nll_inliers_valid_mean))

    auroc = roc_auc_score(y_true, y_scores)
    return auroc

def calc_auroc_valid(bae_ensemble, x_inliers_train, x_inliers_valid):

    nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=["se"])
    nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["se"])

    nll_inliers_train_mean = nll_inliers_train.mean(0)[0].mean(-1)
    nll_inliers_valid_mean = nll_inliers_valid.mean(0)[0].mean(-1)


    y_true = np.concatenate((np.zeros(nll_inliers_train_mean.shape[0]),
                             np.ones(nll_inliers_valid_mean.shape[0])))
    y_scores = np.concatenate((nll_inliers_train_mean, nll_inliers_valid_mean))

    auroc = roc_auc_score(y_true, y_scores)

    return auroc

# random_seed = 987
random_seed = 1233333

bae_set_seed(random_seed)

use_cuda = True
clip_data_01 = False
activation = "leakyrelu"
last_activation = "tanh" # tanh
likelihood = "gaussian" # gaussian
train_size = 0.80
num_samples = 10
# multi_perc_thresholds = np.arange(85,100)
# multi_perc_thresholds = np.arange(75,100)
# multi_perc_thresholds = np.arange(90,100)
# multi_perc_thresholds = np.arange(70,80)
multi_perc_thresholds = np.arange(90,100,0.5)
# multi_perc_thresholds = np.arange(90,100,0.05)
# multi_perc_thresholds = np.arange(95,96)
# multi_perc_thresholds = np.arange(90,100)
perc_threshold = 95
# semi_supervised = True
semi_supervised = False
num_train_outliers = 2

#==============PREPARE DATA==========================
base_folder = "od_benchmark"
mat_file_list = os.listdir(base_folder)
mat_file = mat_file_list[-1]
mat = loadmat(os.path.join(base_folder, mat_file))

X = mat['X']
y = mat['y'].ravel()

x_outliers, x_inliers = get_outliers_inliers(X, y)

if semi_supervised:
    x_outliers_train, x_outliers_test = train_test_split(x_outliers, train_size = np.round(num_train_outliers/len(x_outliers),2),
                                                         shuffle=True,
                                                         random_state=random_seed)
else:
    x_outliers_train = x_outliers.copy()
    x_outliers_test = x_outliers.copy()

x_inliers_train, x_inliers_test = train_test_split(x_inliers, train_size=train_size,shuffle=True, random_state=random_seed)
x_inliers_train, x_inliers_valid = train_test_split(x_inliers_train, train_size=train_size,shuffle=True, random_state=random_seed)


#=================SCALER=========================
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

#=================DEFINE BAE========================

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
encoder_nodes = [input_dim*2,input_dim*4,input_dim*8]

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
                            anchored=anchored,
                            weight_decay=weight_decay,
                            num_samples=num_samples,
                            likelihood=likelihood,
                            learning_rate=lr,
                            verbose=False,
                            use_cuda = use_cuda
                            )

#===============FIT BAE CONVERGENCE===========================
train_loader = convert_dataloader(x_inliers_train, batch_size=750, shuffle=True)
run_auto_lr_range_v2(train_loader, bae_ensemble, run_full=True, window_size=1, num_epochs=15)
# bae_ensemble.fit(train_loader,num_epochs=800)

num_epochs_per_cycle = 50
fast_window = num_epochs_per_cycle
slow_window = fast_window*15
n_stop_points = 10
cvg = 0

auroc_valids = []
auroc_threshold = 0.60

while(cvg == 0):
    _, cvg = bae_fit_convergence(bae_ensemble=bae_ensemble, x=train_loader,
                                 num_epoch=num_epochs_per_cycle,
                                 fast_window=fast_window,
                                 slow_window=slow_window,
                                 n_stop_points=n_stop_points
                                 )

    if semi_supervised:
        bae_ensemble.semisupervised_fit(x_inliers=train_loader,
                                        x_outliers=x_outliers_train,
                                        num_epochs=int(num_epochs_per_cycle/2))

    auroc_valid = calc_auroc_valid(bae_ensemble, x_inliers_train, x_inliers_valid)
    auroc_valids.append(auroc_valid)
    print("AUROC-VALID: {:.3f}".format(auroc_valid))
    print("AUROC-THRESHOLD: {:.3f}".format(auroc_threshold))

    auroc_ood = calc_auroc_valid(bae_ensemble, x_inliers_test, x_outliers_test)
    print("AUROC-OOD: {:.3f}".format(auroc_ood))
    if auroc_valid >= auroc_threshold:
        break



fig, ax = plt.subplots(1,1)
plot_convergence(losses=bae_ensemble.losses,
                 fast_window=fast_window,
                 slow_window=slow_window,
                 n_stop_points= n_stop_points,
                 ax=ax)

#===============PREDICT BAE==========================

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

hard_preds_inliers_train_v2 = bae_predict_ood_v2(bae_ensemble, x_inliers_train, x_inliers_train, perc_threshold = perc_threshold)
hard_preds_inliers_test_v2 = bae_predict_ood_v2(bae_ensemble, x_inliers_train, x_inliers_test, perc_threshold = perc_threshold)
hard_preds_outlier_test_v2 = bae_predict_ood_v2(bae_ensemble, x_inliers_train, x_outliers_test, perc_threshold = perc_threshold)

plot_kde_auroc(nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean, mode="kde")


#============= MULTIPLE THRESHOLDS==============

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
multihard_preds_outlier_test_v1 = multi_hard_predict_v1(bae_ensemble, x_inliers_train, x_outliers_test, keys=["nll_mean", "nll_var"], thresholds=multi_perc_thresholds)

multihard_preds_inliers_train_v2 = multi_hard_predict_v2(bae_ensemble, x_inliers_train, x_inliers_train, thresholds=multi_perc_thresholds)
multihard_preds_inliers_test_v2 = multi_hard_predict_v2(bae_ensemble, x_inliers_train, x_inliers_test, thresholds=multi_perc_thresholds)
multihard_preds_outlier_test_v2 = multi_hard_predict_v2(bae_ensemble, x_inliers_train, x_outliers_test, thresholds=multi_perc_thresholds)

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

#================= F1-SCORE & MCC | UNC ==============
def get_y_unc(hard_preds_inliers_test,
              hard_preds_outlier_test,
              uncertainty_threshold=0.95,
              decision_threshold=0.5):
    y_true = np.concatenate((np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
                             np.ones(hard_preds_outlier_test.mean(0).shape[0])))

    y_scores_unc = np.concatenate((hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2))

    y_scores = np.concatenate((hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0)))

    y_scores[np.argwhere(y_scores >= decision_threshold)] = 1
    y_scores[np.argwhere(y_scores < decision_threshold)] = 0
    y_hard_pred = y_scores.astype(int)

    return y_scores_unc, y_hard_pred, y_true

def calc_error_unc(y_scores_unc, y_true, y_hard_pred, unc_threshold=0):
    indices = np.argwhere(y_scores_unc <= unc_threshold)[:,0]
    conf_matr = confusion_matrix(y_true[indices], y_hard_pred[indices])
    if len(conf_matr) > 1:
        tn, fp, fn, tp = conf_matr.ravel()
    else:
        return ()
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

    return fpr, fdr, fnr, forate, tpr, tnr, mcc, f1, perc, ba


#=======================LAST FEATURE======================
# scaled_ECDF_model = ECDF(samples)
# scaled_ECDF = np.clip((scaled_ECDF_model(test_samples)-scaled_ECDF_model(samples.mean()))/(1-scaled_ECDF_model(samples.mean())),0,1)

def convert_ecdf_output(bae_nll_train,bae_nll_test, scaling=True):
    ecdf_outputs = []
    for bae_sample_train_i, bae_sample_test_i in zip(bae_nll_train, bae_nll_test):
        ecdf_model = ECDF(bae_sample_train_i)
        ecdf_output = ecdf_model(bae_sample_test_i)
        if scaling:
            ecdf_output = np.clip((ecdf_output - ecdf_model(samples.mean())) / (
                        1 - ecdf_model(samples.mean())), 0, 1)

        ecdf_outputs.append(ecdf_output)
    return np.array(ecdf_outputs)

def convert_single_ecdf(bae_sample_train_i, bae_sample_test_i, scaling=True):
    ecdf_model = ECDF(bae_sample_train_i)
    ecdf_output = ecdf_model(bae_sample_test_i)
    if scaling:
        ecdf_output = np.clip((ecdf_output - ecdf_model(bae_sample_train_i.mean())) / (
                1 - ecdf_model(bae_sample_train_i.mean())), 0, 1)
    return ecdf_output

def convert_erf(bae_nll_train,bae_nll_test):
    ecdf_outputs = []
    for bae_sample_train_i, bae_sample_test_i in zip(bae_nll_train, bae_nll_test):
        pre_erf_score = (bae_sample_test_i - np.mean(bae_sample_train_i)) / (
                np.std(bae_sample_train_i) * np.sqrt(2))
        erf_score = erf(pre_erf_score)
        erf_score = erf_score.clip(0, 1).ravel()

        ecdf_outputs.append(erf_score)
    return np.array(ecdf_outputs)

def convert_minmax(bae_nll_train,bae_nll_test):
    ecdf_outputs = []
    for bae_sample_train_i, bae_sample_test_i in zip(bae_nll_train, bae_nll_test):

        scaler_ = MinMaxScaler().fit(bae_sample_train_i.reshape(-1, 1))
        score = scaler_.transform(bae_sample_test_i.reshape(-1, 1)).ravel().clip(0, 1)

        ecdf_outputs.append(score)
    return np.array(ecdf_outputs)

def convert_cdf(bae_nll_train,bae_nll_test, dist=gamma, scaling=True):
    cdf_outputs = []
    for bae_sample_train_i, bae_sample_test_i in zip(bae_nll_train, bae_nll_test):
        if isinstance(dist, str) and dist == "ecdf":
            cdf_score = convert_single_ecdf(bae_sample_train_i, bae_sample_test_i, scaling=scaling)
        else:
            prob_args = dist.fit(bae_sample_train_i)
            dist_ = dist(*prob_args)
            cdf_score = dist_.cdf(bae_sample_test_i)

            if scaling:
                cdf_score = np.clip((cdf_score - dist_.cdf(bae_sample_train_i.mean())) /
                                    (1 - dist_.cdf(bae_sample_train_i.mean())),0, 1)

        cdf_outputs.append(cdf_score)
    return np.array(cdf_outputs)


def convert_prob(x, threshold_lb=90, threshold_ub=100):
    prob_y = x.copy()
    if threshold_lb is not None and threshold_ub is not None:
        m = 1/(threshold_ub-threshold_lb)
        c = -threshold_lb/(threshold_ub-threshold_lb)

        prob_y= np.clip(m*prob_y+c,0,1)

    elif (threshold_ub is None) and (threshold_lb is not None):
        np.place(prob_y, prob_y < threshold_lb, 0)

    elif threshold_lb is None and threshold_ub is not None:
        np.place(prob_y, prob_y >= threshold_ub, 0)

    unc_y = prob_y*(1-prob_y)

    return prob_y, unc_y


# def convert_prob(x, threshold_lb=90, threshold_ub=100):
#     # m = 1/(threshold_ub-threshold_lb)
#     # c = -threshold_lb/(threshold_ub-threshold_lb)
#
#     prob_y= x
#     unc_y = prob_y*(1-prob_y)
#
#     return prob_y, unc_y


def convert_hard_pred(prob, p_threshold=0.5):
    hard_inliers_test = np.piecewise(prob, [prob < p_threshold, prob >= p_threshold], [0, 1]).astype(int)
    return hard_inliers_test

def get_pred_unc(prob, unc, type=["epistemic","aleatoric"]):

    if "epistemic" in type:
        epi = prob.var(0)
    else:
        epi = np.zeros(unc.shape[1:])
    if "aleatoric" in type:
        alea = unc.mean(0)
    else:
        alea = np.zeros(unc.shape[1:])
    return epi+alea

def get_y_results(prob_inliers_test_mean, prob_outliers_test_mean, total_unc_inliers_test, total_unc_outliers_test ):

    hard_inliers_test = convert_hard_pred(prob_inliers_test_mean, p_threshold=p_threshold)
    hard_outliers_test = convert_hard_pred(prob_outliers_test_mean, p_threshold=p_threshold)

    y_unc = np.concatenate((total_unc_inliers_test, total_unc_outliers_test))
    y_hard_pred = np.concatenate((hard_inliers_test, hard_outliers_test))
    y_true = np.concatenate((np.zeros_like(hard_inliers_test), np.ones_like(hard_outliers_test)))

    return y_true, y_hard_pred, y_unc

def evaluate_mcc_f1_unc(y_true, y_hard_pred, y_unc):
    unc_thresholds_ = np.unique(np.round(y_unc, 3))
    unc_thresholds = []
    error_uncs = []
    for unc_ in unc_thresholds_:
        error_unc = calc_error_unc(y_unc, y_true, y_hard_pred, unc_threshold=unc_)
        if len(error_unc) > 0:
            error_uncs.append(error_unc)
            unc_thresholds.append(unc_)
    unc_thresholds = np.array(unc_thresholds)
    error_uncs = np.array(error_uncs)

    fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1)
    ax1.plot(unc_thresholds, error_uncs[:,0], '-o')
    ax2.plot(unc_thresholds, error_uncs[:,1], '-o')
    ax3.plot(unc_thresholds, error_uncs[:,3], '-o')
    ax4.plot(unc_thresholds, error_uncs[:,5], '-o')
    ax5.plot(unc_thresholds, error_uncs[:,6], '-o')
    ax6.plot(unc_thresholds, error_uncs[:,7], '-o')
    ax7.plot(unc_thresholds, error_uncs[:,8], '-o')

    spman_mcc = spearmanr(unc_thresholds, error_uncs[:,6])[0]
    spman_f1 = spearmanr(unc_thresholds, error_uncs[:,7])[0]

    mcc_diff = error_uncs[0,6]-error_uncs[-1,6]
    f1_diff = error_uncs[0,7]-error_uncs[-1,7]

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

def get_indices_error(y_true, y_hard_pred, y_unc):

    indices_tp =  np.argwhere((y_true ==1) & (y_hard_pred ==1))[:,0]
    indices_tn =  np.argwhere((y_true ==0) & (y_hard_pred ==0))[:,0]
    indices_fp =  np.argwhere((y_true ==0) & (y_hard_pred ==1))[:,0]
    indices_fn =  np.argwhere((y_true ==1) & (y_hard_pred ==0))[:,0]
    indices_0_error = np.concatenate((indices_tp,indices_tn))
    indices_all_error = np.concatenate((indices_fp,indices_fn))

    error_type1 = np.concatenate((np.ones(len(indices_fp)), np.zeros(len(indices_tp)))).astype(int)
    error_type2 = np.concatenate((np.ones(len(indices_fn)), np.zeros(len(indices_tn)))).astype(int)
    error_all = np.abs((y_true - y_hard_pred))

    y_unc_type1 = np.concatenate((y_unc[indices_fp],y_unc[indices_tp]))
    y_unc_type2 = np.concatenate((y_unc[indices_fn],y_unc[indices_tn]))
    y_unc_all = y_unc.copy()

    return indices_tp, indices_tn, indices_fp, indices_fn, \
           indices_0_error, \
           indices_all_error, error_type1, error_type2, error_all, y_unc_type1, y_unc_type2, y_unc_all

def evaluate_unc_perf(y_true, y_hard_pred, y_unc, verbose=True):
    indices_tp, indices_tn, indices_fp, indices_fn, \
    indices_0_error, \
    indices_all_error, error_type1, error_type2, error_all, y_unc_type1, y_unc_type2, y_unc_all = get_indices_error(y_true, y_hard_pred, y_unc)


    precision_type1, recall_type1, thresholds = precision_recall_curve(error_type1, y_unc_type1)
    precision_type2, recall_type2, thresholds = precision_recall_curve(error_type2, y_unc_type2)
    precision_type_all, recall_type_all, thresholds = precision_recall_curve(error_all, y_unc)

    auprc_type1 = auc(recall_type1, precision_type1)
    auprc_type2 = auc(recall_type2, precision_type2)
    auprc_type_all = auc(recall_type_all, precision_type_all)

    baseline_type1 = error_type1.mean()
    baseline_type2 = error_type2.mean()
    baseline_all = error_all.mean()

    lift_type1 = auprc_type1/ baseline_type1
    lift_type2 = auprc_type2/ baseline_type2
    lift_all = auprc_type_all/ baseline_all

    auroc_type1 = roc_auc_score(error_type1, y_unc_type1) if (baseline_type1 > 0 or baseline_type1==1) else np.nan
    auroc_type2 = roc_auc_score(error_type2, y_unc_type2) if (baseline_type2 > 0 or baseline_type2==1) else np.nan
    auroc_type_all = roc_auc_score(error_all, y_unc) if (baseline_all > 0 or baseline_all==1) else np.nan

    if verbose:
        print("AUPRC-TYPE1 (TP VS FP): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(auprc_type1, baseline_type1,
                                                                                              lift_type1))
        print("AUPRC-TYPE2 (TN VS FN): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(auprc_type2, baseline_type2,
                                                                                              lift_type2))
        print("AUPRC-ALL   (TP+TN VS FP+FN): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(auprc_type_all,
                                                                                                    baseline_all, lift_all))

        print("AUROC-TYPE1 (TP VS FP): {:.2f} ".format(auroc_type1))
        print("AUROC-TYPE2 (TN VS FN): {:.2f} ".format(auroc_type2))
        print("AUROC-ALL   (TP+TN VS FP+FN): {:.2f} ".format(auroc_type_all))

    return auprc_type1, auprc_type2, auprc_type_all, \
           baseline_type1, baseline_type2, baseline_all, \
           lift_type1, lift_type2, lift_all ,\
           auroc_type1, auroc_type2, auroc_type_all


def plot_unc_tptnfpfn(y_true, y_hard_pred, y_unc):
    indices_tp, indices_tn, indices_fp, indices_fn, \
    indices_0_error, \
    indices_all_error, error_type1, error_type2, error_all, y_unc_type1, y_unc_type2, y_unc_all = get_indices_error(y_true, y_hard_pred, y_unc)


    labels = ["TP","TN","TP+TN","Type 1 (FP)","Type 2 (FN)", "FP+FN"]
    fig, ax1= plt.subplots(1,1)
    ax1.boxplot([y_unc[indices_tp],
                 y_unc[indices_tn],
                 y_unc[indices_0_error],
                 y_unc[indices_fp],
                 y_unc[indices_fn],
                 y_unc[indices_all_error],
                 ]
                ,
                notch=False
                )

    ax1.set_xticks(np.arange(1, len(labels) + 1))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Uncertainty")

def convert_logistic_prob(nll_train, nll_test, scaler_type=MinMaxScaler, threshold=0.95):
    test_probas = []
    for nll_train_i, nll_test_i  in zip(nll_train, nll_test):
        if scaler_type is not None:
            temp_scaler = scaler_type()
        nll_threshold = np.percentile(nll_train_i, threshold*100)
        indices_p = np.argwhere(nll_train_i >= nll_threshold)[:,0]
        indices_n = np.argwhere(nll_train_i < nll_threshold)[:, 0]

        y_target = np.zeros(len(nll_train_i))
        y_target[indices_p] = 1
        y_target[indices_n] = 0
        y_target = y_target.astype(int)

        clf = LogisticRegression()
        if scaler_type is not None:
            clf = clf.fit(temp_scaler.fit_transform(nll_train_i.reshape(-1,1)), y_target)
        else:
            clf = clf.fit(nll_train_i.reshape(-1,1), y_target)

        test_proba_i = clf.predict_proba(temp_scaler.transform(nll_test_i.reshape(-1,1)))[:,1]
        test_probas.append(test_proba_i)
    test_probas= np.array(test_probas)
    return test_probas, test_probas*(1-test_probas)



# thresholds = (0.25,0.5)
# thresholds = (0.5,1.0)
# thresholds = (0.75,None)
# thresholds = (0.75,1.0)
# thresholds = (0.5,None)
thresholds = (None,None)
# p_threshold = (thresholds[0]+1)/2
p_threshold = 0.5
# unc_types = ["aleatoric"]
# unc_types = ["epistemic"]

# logistic_scaler = RobustScaler
logistic_scaler = StandardScaler
logistic_threshold = 0.9

nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=["se"]).mean(-1)[:,0]
nll_inliers_test = bae_ensemble.predict_samples(x_inliers_test, select_keys=["se"]).mean(-1)[:,0]
nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["se"]).mean(-1)[:,0]
nll_outliers_test = bae_ensemble.predict_samples(x_outliers_test, select_keys=["se"]).mean(-1)[:,0]
nll_outliers_train = bae_ensemble.predict_samples(x_outliers_train, select_keys=["se"]).mean(-1)[:,0]

# nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=["bce"]).mean(-1)[:,0]
# nll_inliers_test = bae_ensemble.predict_samples(x_inliers_test, select_keys=["bce"]).mean(-1)[:,0]
# nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["bce"]).mean(-1)[:,0]
# nll_outliers_test = bae_ensemble.predict_samples(x_outliers_test, select_keys=["bce"]).mean(-1)[:,0]
# nll_outliers_train = bae_ensemble.predict_samples(x_outliers_train, select_keys=["bce"]).mean(-1)[:,0]


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

unc_types = ["epistemic","aleatoric"]
# unc_types = ["epistemic"]
# unc_types = ["aleatoric"]


dist_cdf = uniform #gamma, lognorm , norm , uniform, expon, "ecdf"
scaling = False
p_threshold = 0.5
prob_inliers_train, unc_inliers_train = convert_prob(convert_cdf(nll_inliers_train,
                                                     nll_inliers_train, dist = dist_cdf, scaling=scaling),
                                 *thresholds)
prob_inliers_test, unc_inliers_test = convert_prob(convert_cdf(nll_inliers_valid,
                                                     nll_inliers_valid, dist= dist_cdf, scaling=scaling),
                                 *thresholds)
prob_outliers_test, unc_outliers_test = convert_prob(convert_cdf(nll_inliers_valid,
                                                     nll_outliers_test, dist=dist_cdf, scaling=scaling),
                                 *thresholds)


fig, (ax1,ax2) = plt.subplots(2,1)
for nll_, prob_ in zip(nll_inliers_train, prob_inliers_train):
    indices = np.argsort(nll_)
    ax1.plot(nll_[indices], prob_[indices], color="tab:blue", alpha=0.5)
    ax2.plot(nll_[indices], prob_[indices]*(1-prob_[indices]), color="tab:blue", alpha=0.5)

prob_inliers_test_mean = prob_inliers_test.mean(0)
prob_outliers_test_mean = prob_outliers_test.mean(0)

total_unc_inliers_test = get_pred_unc(prob_inliers_test, unc_inliers_test, type=unc_types)
total_unc_outliers_test = get_pred_unc(prob_outliers_test, unc_outliers_test, type=unc_types)

y_true, y_hard_pred, y_unc = get_y_results(prob_inliers_test_mean, prob_outliers_test_mean,
                                           total_unc_inliers_test, total_unc_outliers_test)

evaluate_mcc_f1_unc(y_true, y_hard_pred, y_unc)
evaluate_unc_perf(y_true, y_hard_pred, y_unc, verbose=True)
plot_unc_tptnfpfn(y_true, y_hard_pred, y_unc)
plot_kde_auroc(prob_inliers_test_mean,prob_inliers_test_mean, prob_outliers_test_mean,mode="hist")

#========================FIT BETA DISTRIBUTION===========================
from scipy.stats import beta, gamma, lognorm, norm, uniform


samples = nll_inliers_train[0]
test_samples = nll_inliers_test[0]

# samples = (samples - np.mean(samples)) / (np.std(samples) * np.sqrt(2))
# test_samples = (test_samples - np.mean(samples)) / (np.std(samples) * np.sqrt(2))

# a1, loc1, scale1 = gamma.fit(samples)
gamma_params = gamma.fit(samples)


rv1 = gamma(*gamma.fit(samples))
rv2 = lognorm(*lognorm.fit(samples))
rv3 = norm(*norm.fit(samples))
rv4 = uniform(*uniform.fit(samples))
rv6 = expon(*expon.fit(samples))

pre_erf_score = (test_samples - np.mean(samples)) / (np.std(samples) * np.sqrt(2))
# pre_erf_score = test_samples
rv5 = np.clip(erf(pre_erf_score),0,1)
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
plt.scatter(test_samples, rv3.cdf(test_samples)+0.01,color="green")
plt.scatter(test_samples, rv4.cdf(test_samples))
plt.scatter(test_samples, rv6.cdf(test_samples))
plt.scatter(test_samples, rv5, color="black")

plt.scatter(test_samples, ECDF(samples)(test_samples))
plt.legend(["GAMMA", "LOGNORM","NORM","UNI","ERF","ECDF","EXPON"])

plt.figure()
for dist in [rv1, rv2, rv3, rv4,rv6]:
    scaled_res = np.clip((dist.cdf(test_samples)-dist.cdf(samples.mean()))/(1-dist.cdf(samples.mean())),0,1)
    plt.scatter(test_samples, scaled_res)
plt.scatter(test_samples, rv5, color="black")

scaled_ECDF_model = ECDF(samples)
scaled_ECDF = np.clip((scaled_ECDF_model(test_samples)-scaled_ECDF_model(samples.mean()))/(1-scaled_ECDF_model(samples.mean())),0,1)

plt.scatter(test_samples, scaled_ECDF+0.01)
plt.legend(["GAMMA", "LOGNORM","NORM","UNI","EXPON","ERF","ECDF"])

#========================ESS=============================================

# filter out high unc. samples

unc_thresholds = [1.0,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.]
unc_threshold = 1.0
filtered_test = prob_inliers_test_mean[np.argwhere(total_unc_inliers_test*4<=unc_threshold)[:,0]]
filtered_outlier = prob_outliers_test_mean[np.argwhere(total_unc_outliers_test*4<=unc_threshold)[:,0]]


filtereds = []
for unc_threshold in unc_thresholds:
    filtered_test = prob_inliers_test_mean[np.argwhere(total_unc_inliers_test * 4 <= unc_threshold)[:, 0]]
    filtered_outlier = prob_outliers_test_mean[np.argwhere(total_unc_outliers_test * 4 <= unc_threshold)[:, 0]]

    # filtered_test = prob_inliers_test_mean[np.argwhere(total_unc_inliers_test * 4 >= unc_threshold)[:, 0]]
    # filtered_outlier = prob_outliers_test_mean[np.argwhere(total_unc_outliers_test * 4 >= unc_threshold)[:, 0]]

    if len(filtered_test) > 0 and len(filtered_outlier) > 0:
        auroc_filtered = calc_auroc(filtered_test, filtered_outlier)
        filtereds.append(auroc_filtered)

print(filtereds)







#
# USE METHOD 2 : LOGISTIC REG
# prob_inliers_test, unc_inliers_test = convert_logistic_prob(nll_inliers_train, nll_inliers_test,
#                                                             scaler_type=logistic_scaler,
#                                                             threshold=logistic_threshold)
# prob_outliers_test, unc_outliers_test = convert_logistic_prob(nll_inliers_train, nll_outliers_test,
#                                                               scaler_type=logistic_scaler,
#                                                               threshold=logistic_threshold)
#
# prob_inliers_test, unc_inliers_test = convert_logistic_prob(nll_inliers_valid, nll_inliers_test,
#                                                             scaler_type=logistic_scaler,
#                                                             threshold=logistic_threshold)
# prob_outliers_test, unc_outliers_test = convert_logistic_prob(nll_inliers_valid, nll_outliers_test,
#                                                               scaler_type=logistic_scaler,
#                                                               threshold=logistic_threshold)


# rr = nll_inliers_train/nll_inliers_train.sum(0)
# rr = nll_inliers_test/nll_inliers_test.sum(0)
# rr_o = nll_outliers_test/nll_outliers_test.sum(0)
#
#
# rr_i = 1/((rr**2).sum(0))
# rr_j = 1/((rr_o**2).sum(0))
#
# # rr_ess = (nll_inliers_train.sum(0)**2)/(nll_inliers_train**2).sum(0)
# # rr_ess2 = (nll_outliers_test.sum(0)**2)/(nll_outliers_test**2).sum(0)
# #
# # plt.figure()
# # plt.hist(rr_ess,density=True)
# # plt.hist(rr_ess2,density=True)
# #
# # plt.figure()
# # plt.hist(nll_inliers_train.mean(0),density=True)
# # plt.hist(nll_outliers_test.mean(0),density=True)
#
# plt.figure()
# plt.hist(rr_i,density=True, alpha=0.5)
# plt.hist(rr_j,density=True, alpha=0.5)
#
# plot_kde_auroc(rr_i, rr_i, rr_j)

# plt.figure()
# plt.hist(rr.var(0),density=True)
# plt.hist(rr_o.var(0),density=True)

#============================================================

# def convert_logistic_prob(nll_train, nll_test, scaler_type=MinMaxScaler, threshold=0.95):
#     test_probas = []
#     for nll_train_i, nll_test_i  in zip(nll_train, nll_test):
#         if scaler_type is not None:
#             temp_scaler = scaler_type()
#         nll_threshold = np.percentile(nll_train_i, threshold*100)
#         indices_p = np.argwhere(nll_train_i >= nll_threshold)[:,0]
#         indices_n = np.argwhere(nll_train_i < nll_threshold)[:, 0]
#
#         y_target = np.zeros(len(nll_train_i))
#         y_target[indices_p] = 1
#         y_target[indices_n] = 0
#         y_target = y_target.astype(int)
#
#         clf = LogisticRegression()
#         if scaler_type is not None:
#             clf = clf.fit(temp_scaler.fit_transform(nll_train_i.reshape(-1,1)), y_target)
#         else:
#             clf = clf.fit(nll_train_i.reshape(-1,1), y_target)
#
#         test_proba_i = clf.predict_proba(temp_scaler.transform(nll_test_i.reshape(-1,1)))[:,1]
#         test_probas.append(test_proba_i)
#     test_probas= np.array(test_probas)
#     return test_probas, test_probas*(1-test_probas)

# p_threshold = 0.95
#
# ecdf_train = convert_ecdf_output(nll_inliers_train, nll_inliers_train)
# ecdf_ood = convert_ecdf_output(nll_inliers_train, nll_outliers_test)
#
# ecdf_train = convert_ecdf_output(nll_inliers_train, nll_inliers_train)
# ecdf_ood = convert_ecdf_output(nll_inliers_train, nll_outliers_test)
#
# fig, (ax1,ax2) = plt.subplots(2,1)
# for i in range(num_samples):
#     ax1.scatter(nll_inliers_train[i], ecdf_train[i], color="tab:blue",alpha=0.5)
#     ax1.scatter(nll_outliers_test[i], ecdf_ood[i], color="tab:orange",alpha=0.5)
# for i in range(num_samples):
#     ax2.scatter(nll_inliers_train[i], ecdf_train[i]*(1-ecdf_train[i]), color="tab:blue",alpha=0.5)
#     ax2.scatter(nll_outliers_test[i], ecdf_ood[i]*(1-ecdf_ood[i]), color="tab:orange",alpha=0.5)
#
# x_ood = nll_inliers_train[i][np.argwhere(ecdf_train[i] >= p_threshold)[:,0]]
# y_ood = np.ones_like(x_ood)
#
# x_train = nll_inliers_train[i][np.argwhere(ecdf_train[i] < p_threshold)[:,0]]
# y_train = np.zeros_like(x_train)
#
# x_plot = np.concatenate((x_train.reshape(-1, 1),x_ood.reshape(-1, 1)))
# y_target = np.concatenate((y_train,y_ood))
#
# # scaler = MinMaxScaler()
# scaler = StandardScaler()
# # scaler = RobustScaler()
#
# # clf = RidgeClassifier()
# clf = LogisticRegression()
# clf = clf.fit(scaler.fit_transform(x_plot),y_target)
#
# # clf = LogisticRegression(random_state=0).fit(
# #     np.concatenate((x_train.reshape(-1, 1),x_ood.reshape(-1, 1))),
# #     np.concatenate((y_train,y_ood)))
#
# x_plot = nll_inliers_train[i].reshape(-1,1)
# # proba = clf.decision_function(scaler.transform(x_plot))
# # proba = clf.predict(scaler.transform(x_plot))
# proba = clf.predict_proba(scaler.transform(x_plot))[:,1]
# hard_pred = clf.predict(scaler.transform(x_plot))
#
# fig, (ax1,ax2) = plt.subplots(2,1)
#
# ax1.scatter(x_plot, y_target)
# ax1.scatter(x_plot, proba)
# ax1.scatter(x_plot, hard_pred+0.01)
# ax1.scatter(nll_inliers_train[i], ecdf_train[i], color="tab:red",alpha=0.5)
#
# ax2.scatter(x_plot, proba*(1-proba)*4)
#
# plt.figure()
# plt.scatter(ecdf_train[i], proba)
# plt.scatter(ecdf_train[i], proba*(1-proba)*4)
#
