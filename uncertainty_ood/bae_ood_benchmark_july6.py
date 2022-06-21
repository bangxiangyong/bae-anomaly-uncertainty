from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
import os
import numpy as np
from scipy.stats import spearmanr, percentileofscore, iqr

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, auc, precision_recall_curve, \
    confusion_matrix
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


def calc_auroc(nll_inliers_train_mean, nll_inliers_valid_mean):
    y_true = np.concatenate((np.zeros(nll_inliers_train_mean.shape[0]),
                             np.ones(nll_inliers_valid_mean.shape[0])))
    y_scores = np.concatenate((nll_inliers_train_mean, nll_inliers_valid_mean))

    auroc = roc_auc_score(y_true, y_scores)
    return auroc

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

def get_y_unc(hard_preds_inliers_test, hard_preds_outlier_test,
              decision_threshold=0.5):
    y_true = np.concatenate((np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
                             np.ones(hard_preds_outlier_test.mean(0).shape[0])))

    y_scores_unc = np.concatenate((hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2))

    y_scores = np.concatenate((hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0)))

    y_scores[np.argwhere(y_scores >= decision_threshold)] = 1
    y_scores[np.argwhere(y_scores < decision_threshold)] = 0
    y_hard_pred = y_scores.astype(int)

    return y_scores_unc, y_hard_pred, y_true


# random_seed = 987
random_seed = 1233333

bae_set_seed(random_seed)

use_cuda = True
clip_data_01 = True
activation = "leakyrelu"
last_activation = "sigmoid" # tanh
likelihood = "gaussian" # gaussian
train_size = 0.80
num_samples = 5
# multi_perc_thresholds = np.arange(85,100)
# multi_perc_thresholds = np.arange(75,100)
multi_perc_thresholds = np.arange(95,100)
# multi_perc_thresholds = np.arange(90,100)

perc_threshold = 95
# semi_supervised = True
semi_supervised = False
num_train_outliers = 5

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

# x_outliers_test = x_outliers[4:]
# x_outliers_train = x_outliers[:4]

x_inliers_train, x_inliers_test = train_test_split(x_inliers, train_size=train_size,shuffle=True, random_state=random_seed)
x_inliers_train, x_inliers_valid = train_test_split(x_inliers_train, train_size=train_size,shuffle=True, random_state=random_seed)


#=================SCALER=========================
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

nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=["se"])[:,0]
nll_inliers_test = bae_ensemble.predict_samples(x_inliers_test, select_keys=["se"])[:,0]
nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["se"])[:,0]
nll_outliers_test = bae_ensemble.predict_samples(x_outliers_test, select_keys=["se"])[:,0]
nll_outliers_train = bae_ensemble.predict_samples(x_outliers_train, select_keys=["se"])[:,0]


def convert_prob(nll_train, nll_test):
    convert_oprob = np.vectorize(lambda a: percentileofscore(nll_train, a, kind='weak'))
    oprob_id_test = convert_oprob(nll_test)
    return oprob_id_test

def convert_iqr(nll_train, nll_test):
    iqr_score = iqr(nll_train)
    outlierness = (nll_test-np.percentile(nll_train, 75))/iqr_score
    return outlierness

def convert_adjust_iqr(nll_train, nll_test):
    iqr_score = iqr(nll_train)
    mc_1 = medcouple(nll_train)
    multiplier = np.exp(3 * mc_1) if mc_1 >= 0 else np.exp(4 * mc_1)
    outlierness = (nll_test-np.percentile(nll_train, 75))/(iqr_score*multiplier)
    return outlierness


# Convert to a common scale wrt training data

# nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=["se"])[:,0]

# def nmlise_score(ref_dist, scores):
#     scores = scores/ref_dist.mean(0).mean(0)
#     return scores

# def nmlise_score(ref_dist, scores):
#     scores = scores/np.mean(ref_dist,0)
#     return scores
#
# # norm_nll_id_test = np.array([nmlise_score(nll_inliers_train, bae_pred) for bae_pred in nll_inliers_test])
# # norm_nll_ood_test = np.array([nmlise_score(nll_inliers_train, bae_pred) for bae_pred in nll_outliers_test])
#
# norm_nll_id_test = np.array([nmlise_score(bae_ref, bae_pred) for bae_ref,bae_pred in zip(nll_inliers_train,nll_inliers_test)])
# norm_nll_ood_test = np.array([nmlise_score(bae_ref, bae_pred) for bae_ref,bae_pred in zip(nll_inliers_train,nll_outliers_test)])
#
# # norm_nll_id_test = np.array([nmlise_score(bae_ref, bae_pred) for bae_ref,bae_pred in zip(nll_inliers_train,nll_inliers_test)])
# # norm_nll_ood_test = np.array([nmlise_score(bae_ref, bae_pred) for bae_ref,bae_pred in zip(nll_inliers_train,nll_outliers_test)])
#
# auroc_norm_ood = calc_auroc(norm_nll_id_test.mean(0).mean(-1), norm_nll_ood_test.mean(0).mean(-1))
#
# auroc_ood = calc_auroc(nll_inliers_test.mean(0).mean(-1),nll_outliers_test.mean(0).mean(-1))
#
# print(auroc_norm_ood)
# print(auroc_ood)
#
# plt.figure()
# for valid,test in zip(norm_nll_id_test, norm_nll_ood_test):
#     # plt.hist(valid.mean(-1), alpha=0.5, color="tab:blue",density=True)
#     # plt.hist(test.mean(-1), alpha=0.5, color="tab:orange",density=True)
#
#     sns.kdeplot(valid.mean(-1), alpha=0.5, color="tab:blue")
#     sns.kdeplot(test.mean(-1), alpha=0.5, color="tab:orange")
#
# sns.kdeplot(norm_nll_id_test.mean(0).mean(-1), alpha=0.5, color="tab:green")
# sns.kdeplot(norm_nll_ood_test.mean(0).mean(-1), alpha=0.5, color="tab:red")
#
#
# # auroc_norm_ood = calc_auroc(norm_nll_id_test.mean(0).mean(-1), norm_nll_ood_test.mean(0).mean(-1))
# norm_auroc = np.array([calc_auroc(bae_ref.mean(-1), bae_pred.mean(-1)) for bae_ref,bae_pred in
#                               zip(norm_nll_id_test, norm_nll_ood_test)])
#
# auroc_ = np.array([calc_auroc(bae_ref.mean(-1), bae_pred.mean(-1)) for bae_ref,bae_pred in
#                               zip(nll_inliers_test, nll_outliers_test)])


# ECDF
# def convert_ecdf(nll_ref, nll_test):
#     ecdf_ = ECDF(nll_ref.mean(0).mean(-1))
#     nll_test_ = ecdf_(nll_test)
#
#     return nll_test_
#
#
# ecdf_1 = np.array([convert_ecdf(nll_inliers_train, nll_i.mean(-1)) for nll_i in nll_inliers_test]).mean(0)
# ecdf_2 = np.array([convert_ecdf(nll_inliers_train, nll_i.mean(-1)) for nll_i in nll_outliers_test]).mean(0)
#
# auroc_i = calc_auroc(ecdf_1, ecdf_2)
#
# auroc_ood = calc_auroc(nll_inliers_test.mean(0).mean(-1),nll_outliers_test.mean(0).mean(-1))
#
#
# # inliers_mean = nll_inliers_test.mean(0)
#
# # feat_i = 0
# # auroc_i = calc_auroc(nll_inliers_test.mean(0)[:,feat_i],nll_outliers_test.mean(0)[:,feat_i])
#
# auroc_is = [calc_auroc(nll_inliers_test.mean(0)[:,feat_i],nll_outliers_test.mean(0)[:,feat_i]) for feat_i in range(input_dim)]
#
# bae_i = 3
# auroc_is = [calc_auroc(nll_inliers_test[bae_i, :,feat_i],nll_outliers_test[bae_i,:,feat_i]) for feat_i in range(input_dim)]
#

#===========================================
def apply_threshold(nll_test_,threshold):
    p_indices = np.argwhere(nll_test_ >= threshold)[:,0]
    n_indices = np.argwhere(nll_test_ < threshold)[:,0]

    nll_test_[p_indices] = 1
    nll_test_[n_indices] = 0
    return nll_test_

def convert_ecdf(nll_ref, nll_test, threshold=0.95):
    ecdf_ = ECDF(nll_ref)
    nll_test_ = ecdf_(nll_test)

    # apply threshold
    nll_test_ = apply_threshold(nll_test_,threshold)

    return nll_test_

def convert_tukey(nll_ref, nll_test, threshold=0.95):
    iqr_score = iqr(nll_ref)
    mc_1 = medcouple(nll_ref)
    # multiplier = np.exp(3 * mc_1) if mc_1 >= 0 else np.exp(4 * mc_1)
    multiplier = 1
    nll_test_ = (nll_test-np.percentile(nll_ref, 75))/(iqr_score*multiplier)

    # ecdf_ = ECDF(nll_ref)
    # nll_test_ = ecdf_(nll_test)

    # apply threshold
    nll_test_ = apply_threshold(nll_test_,threshold)

    return nll_test_

def apply_func(func, nll_ref, nll_test, axis=1, **kwargs):
    n_dims = nll_ref.shape[axis]
    # res = np.array([nll_test[:,i] for i in range(n_dims)])
    if axis == 1:
        res = np.array([func(nll_ref[:,i],nll_test[:,i], **kwargs) for i in range(n_dims)])
        res = res.T
    else :
        res = np.array([func(nll_ref[i], nll_test[i], **kwargs) for i in range(n_dims)])
    return res


# convert_ecdf(nll_inliers_train.mean(0), nll_inliers_test.mean(0), threshold=0.95)

# feat_id_test = apply_func(convert_ecdf, nll_inliers_train.mean(-1), nll_inliers_test.mean(-1), axis=0, threshold=0.80)
# feat_ood_test = apply_func(convert_ecdf, nll_inliers_train.mean(-1), nll_outliers_test.mean(-1), axis=0, threshold=0.80)

# threshold = 0.95
# feat_id_test = apply_func(convert_ecdf, nll_inliers_valid.mean(-1), nll_inliers_test.mean(-1), axis=0, threshold=threshold)
# feat_ood_test = apply_func(convert_ecdf, nll_inliers_valid.mean(-1), nll_outliers_test.mean(-1), axis=0, threshold=threshold)

# feat_id_test = apply_func(convert_ecdf, nll_inliers_train.mean(-1), nll_inliers_test.mean(-1), axis=0, threshold=threshold)
# feat_ood_test = apply_func(convert_ecdf, nll_inliers_train.mean(-1), nll_outliers_test.mean(-1), axis=0, threshold=threshold)

# threshold = 0.5
# feat_id_test = apply_func(convert_tukey, nll_inliers_valid.mean(-1), nll_inliers_test.mean(-1), axis=0, threshold=threshold)
# feat_ood_test = apply_func(convert_tukey, nll_inliers_valid.mean(-1), nll_outliers_test.mean(-1), axis=0, threshold=threshold)

threshold = 1.5
feat_id_test = apply_func(convert_tukey, nll_inliers_train.mean(-1), nll_inliers_test.mean(-1), axis=0, threshold=threshold)
feat_ood_test = apply_func(convert_tukey, nll_inliers_train.mean(-1), nll_outliers_test.mean(-1), axis=0, threshold=threshold)

# plt.figure()
# plt.hist(feat_id_test.mean(0))
# plt.hist(feat_ood_test.mean(0))
#
# auroc_ood = calc_auroc(feat_id_test.mean(0), feat_ood_test.mean(0))
# print(auroc_ood)

#===============UNCERTAINTY ANALYSIS============
y_unc, y_hard_pred, y_true = get_y_unc(feat_id_test, feat_ood_test)

unc_thresholds = np.unique(np.round(y_unc,3))
error_uncs = np.array([calc_error_unc(y_unc, y_true, y_hard_pred, unc_threshold=unc_) for unc_ in unc_thresholds])

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
y_unc_all = y_unc

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

print("AUPRC-TYPE1 (TP VS FP): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(auprc_type1, baseline_type1, lift_type1))
print("AUPRC-TYPE2 (TN VS FN): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(auprc_type2, baseline_type2, lift_type2))
print("AUPRC-ALL   (TP+TN VS FP+FN): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(auprc_type_all, baseline_all, lift_all))

auroc_type1 = roc_auc_score(error_type1, y_unc_type1) if (baseline_type1 > 0 or baseline_type1==1) else np.nan
auroc_type2 = roc_auc_score(error_type2, y_unc_type2) if (baseline_type2 > 0 or baseline_type2==1) else np.nan
auroc_type_all = roc_auc_score(error_all, y_unc) if (baseline_all > 0 or baseline_all==1) else np.nan

print("AUROC-TYPE1 (TP VS FP): {:.2f} ".format(auprc_type1))
print("AUROC-TYPE2 (TN VS FN): {:.2f} ".format(auroc_type2))
print("AUROC-ALL   (TP+TN VS FP+FN): {:.2f} ".format(auroc_type_all))


























# feat_id_test = apply_func(convert_ecdf, nll_inliers_train.mean(0), nll_inliers_test.mean(0), threshold=0.80)
# feat_ood_test = apply_func(convert_ecdf, nll_inliers_train.mean(0), nll_outliers_test.mean(0), threshold=0.80)

# feat_id_test = apply_func(convert_ecdf, nll_inliers_valid.mean(0), nll_inliers_test.mean(0), threshold=0.80)
# feat_ood_test = apply_func(convert_ecdf, nll_inliers_valid.mean(0), nll_outliers_test.mean(0), threshold=0.80)
#
#
# # auroc_feat_ood = calc_auroc(feat_id_test.mean(-1),feat_ood_test.mean(-1))
# # auroc_feat_ood = calc_auroc(feat_id_test.max(-1),feat_ood_test.max(-1))
#
# perc_feats = 85
# auroc_feat_ood = calc_auroc(np.percentile(feat_id_test, perc_feats, axis=1),
#                             np.percentile(feat_ood_test, perc_feats, axis=1))
#
# # auroc_feat_ood = calc_auroc(feat_id_test.max(-1),feat_ood_test.max(-1))
#
# print(auroc_ood)
# print(auroc_feat_ood)
#
# # plt.figure()
# # for dt in feat_id_test:
# #     plt.hist(dt, color="tab:blue", alpha=0.5, density=True)
# # for dt in feat_ood_test:
# #     plt.hist(dt, color="tab:orange", alpha=0.5, density=True)
# plt.figure()
# plt.hist(np.percentile(feat_id_test, perc_feats, axis=1), alpha=0.5, density=True)
# plt.hist(np.percentile(feat_ood_test, perc_feats, axis=1), alpha=0.5, density=True)
#
# # sns.kdeplot(feat_ood_test[1,:])
#
# feat_id_test.var(-1).mean(0)
#
# feat_ood_test.var(-1).mean(0)
#
#
# auroc_feat_ood = calc_auroc(feat_id_test.var(-1), feat_ood_test.var(-1))
#

