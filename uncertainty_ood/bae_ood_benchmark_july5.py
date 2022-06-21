from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
import os
import numpy as np
from scipy.stats import spearmanr, percentileofscore, iqr

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

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
clip_data_01 = True
activation = "leakyrelu"
last_activation = "tanh" # tanh
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
mat_file = mat_file_list[-2]
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
encoder_nodes = [input_dim*6,input_dim*4]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*6]
# encoder_nodes = [input_dim*4,input_dim*4,int(input_dim/3)]
# encoder_nodes = [input_dim*8,input_dim*4,int(input_dim*5)]
# encoder_nodes = [input_dim*8,int(input_dim/3)]
# encoder_nodes = [input_dim*8,input_dim*4,int(input_dim/2)]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*8]

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
auroc_threshold = 0.65

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

nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=["se"])
nll_inliers_test = bae_ensemble.predict_samples(x_inliers_test, select_keys=["se"])
nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["se"])
nll_outliers_test = bae_ensemble.predict_samples(x_outliers_test, select_keys=["se"])
nll_outliers_train = bae_ensemble.predict_samples(x_outliers_train, select_keys=["se"])

# Convert to a common scale wrt training data
# percentileofscore()


# nll_train = nll_inliers_train[0][0].mean(-1)
# convert_oprob = np.vectorize(lambda a : percentileofscore(nll_train, a, kind='mean'))
# oprob_id_test = convert_oprob(nll_inliers_test[0][0].mean(-1))

nll_train = nll_inliers_train[:,0].mean(-1)
nll_valid = nll_inliers_valid[:,0].mean(-1)

nll_test = nll_inliers_test[:,0].mean(-1)
nll_ood = nll_outliers_test[:,0].mean(-1)

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

def convert_null(nll_train, nll_test):
    return nll_test

nll_ref = nll_valid
convert_method = convert_adjust_iqr
probs = np.array([convert_method(train_, test_) for train_ , test_ in zip(nll_ref,nll_test)])

probs_mean = probs.mean(0)
probs_mean = probs[2]
probs_unc = (probs).var(0)

probs_outlier = np.array([convert_method(train_, test_) for train_ , test_ in zip(nll_ref,nll_ood)])

# probs_mean_out = probs_outlier.mean(0)
probs_mean_out = probs_outlier[2]
probs_unc_out = (probs_outlier).var(0)


plt.figure()
plt.hist(probs_mean,density=True, alpha=0.85)
plt.hist(probs_mean_out,density=True, alpha=0.85)


plt.figure()
plt.scatter(probs_mean, probs_unc)
plt.scatter(probs_mean_out, probs_unc_out)

plt.figure()
plt.boxplot([probs_mean,probs_mean_out])




