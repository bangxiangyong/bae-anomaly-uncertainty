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
from util.convergence import bae_fit_convergence, plot_convergence, bae_semi_fit_convergence
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.metrics import matthews_corrcoef
from util.evaluate_ood import plot_histogram_ood, plot_kde_ood
import torch

def calc_auroc_valid(bae_ensemble, x_inliers_train, x_inliers_valid, select_keys=["se"]):

    nll_inliers_train = bae_ensemble.predict_samples(x_inliers_train, select_keys=select_keys)
    nll_inliers_valid = bae_ensemble.predict_samples(x_inliers_valid, select_keys=select_keys)

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
last_activation = "sigmoid" # tanh
likelihood = "gaussian" # gaussian
train_size = 0.80
num_samples = 5
# multi_perc_thresholds = np.arange(85,100)
# multi_perc_thresholds = np.arange(75,100)
multi_perc_thresholds = np.arange(99,100)
# multi_perc_thresholds = np.arange(90,100)
perc_threshold = 97.5
# semi_supervised = True
semi_supervised = False
num_train_outliers = 2

#==============PREPARE DATA==========================
base_folder = "od_benchmark"
mat_file_list = os.listdir(base_folder)
mat_file = mat_file_list[1]
mat = loadmat(os.path.join(base_folder, mat_file))

X = mat['X']
y = mat['y'].ravel()

x_outliers, x_inliers = get_outliers_inliers(X, y)

# x_outliers_train, x_outliers_test = train_test_split(x_outliers, train_size = np.round(num_train_outliers/len(x_outliers),2),shuffle=True, random_state=random_seed)
# x_outliers_test = x_outliers[4:]
# x_outliers_train = x_outliers[:4]

if semi_supervised:
    x_outliers_train, x_outliers_test = train_test_split(x_outliers, train_size = np.round(num_train_outliers/len(x_outliers),2),shuffle=True, random_state=random_seed)
else:
    x_outliers_train = x_outliers.copy()
    x_outliers_test = x_outliers.copy()


x_inliers_train, x_inliers_test = train_test_split(x_inliers, train_size=train_size,shuffle=True, random_state=random_seed)
x_inliers_train, x_inliers_valid = train_test_split(x_inliers_train, train_size=train_size,shuffle=True, random_state=random_seed)


#=================SCALER=========================
# scaler = MinMaxScaler()
scaler = StandardScaler()
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
# weight_decay = 0.01
# weight_decay = 0.000001
lr = 0.025
latent_dim = 100

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
# encoder_nodes = [input_dim*6,input_dim*4,input_dim*2]
# encoder_nodes = [input_dim*6,input_dim*4,input_dim*2]
# encoder_nodes = [input_dim*2,input_dim*4,input_dim*8,input_dim*4,input_dim*2]
# encoder_nodes = [input_dim*10,input_dim*2]
encoder_nodes = [input_dim*10]
encoder = Encoder([DenseLayers(input_size=input_dim,
                               architecture=encoder_nodes[:-1],
                               output_size=encoder_nodes[-1], activation=activation,last_activation=activation
                               )])

# specify decoder-mu
# decoder_mu = infer_decoder(encoder,activation=activation,last_activation=last_activation) #symmetrical to encoder
# decoder_mu = torch.nn.Sequential(torch.nn.Linear(encoder_nodes[-1],1, bias=False),
#                                  torch.nn.Sigmoid())
decoder_mu = torch.nn.Sequential(torch.nn.Linear(encoder_nodes[-1],latent_dim, bias=False),
                                 )

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
# latent_data = np.random.randn(latent_dim)
latent_data = np.zeros(latent_dim)
# latent_data = np.ones(latent_dim)*0.01

def make_latent_copies(ori_data, latent_data, bae_samples=0):
    latent_data_ = np.ones((ori_data.shape[0], len(latent_data)))*latent_data
    if bae_samples == 0:
        return latent_data_
    else:
        latent_data_ = np.array([latent_data_.copy() for i in range(bae_samples)])
        return latent_data_

if semi_supervised:
    train_loader = convert_dataloader(x_inliers_train,
                                      y=np.concatenate((np.ones((x_inliers_train.shape[0],1)),
                                                        np.zeros((x_outliers_train.shape[0],1))
                                                        )),
                                      batch_size=250, shuffle=True)
else:
    train_loader = convert_dataloader(x_inliers_train,
                                      y=make_latent_copies(x_inliers_train,latent_data), batch_size=250,
                                      shuffle=True)

    run_auto_lr_range_v2(train_loader, bae_ensemble, run_full=True, window_size=1, num_epochs=15,supervised=True)
bae_ensemble.fit(train_loader,num_epochs=100,supervised=True)

num_epochs_per_cycle = 50
fast_window = num_epochs_per_cycle
slow_window = fast_window*10
n_stop_points = 10
cvg = 0

auroc_valids = []
auroc_threshold = 0.60




#==================================================================

def calc_auroc(nll_inliers_train_mean, nll_inliers_valid_mean):
    y_true = np.concatenate((np.zeros(nll_inliers_train_mean.shape[0]),
                             np.ones(nll_inliers_valid_mean.shape[0])))
    y_scores = np.concatenate((nll_inliers_train_mean, nll_inliers_valid_mean))

    auroc = roc_auc_score(y_true, y_scores)
    return auroc

outp_inliers_train_raw = bae_ensemble.predict_samples(x_inliers_train, select_keys=["y_mu"])
outp_inliers_test_raw = bae_ensemble.predict_samples(x_inliers_test, select_keys=["y_mu"])
outp_inliers_valid_raw = bae_ensemble.predict_samples(x_inliers_valid, select_keys=["y_mu"])
outp_outliers_test_raw = bae_ensemble.predict_samples(x_outliers_test, select_keys=["y_mu"])
outp_outliers_train_raw = bae_ensemble.predict_samples(x_outliers_train, select_keys=["y_mu"])


outp_inliers_train = ((outp_inliers_train_raw-make_latent_copies(x_inliers_train, latent_data, bae_samples=num_samples))**2)[:,0]
outp_inliers_test = ((outp_inliers_test_raw-make_latent_copies(x_inliers_test, latent_data, bae_samples=num_samples))**2)[:,0]
outp_inliers_valid = ((outp_inliers_valid_raw-make_latent_copies(x_inliers_valid, latent_data, bae_samples=num_samples))**2)[:,0]
outp_outliers_test = ((outp_outliers_test_raw-make_latent_copies(x_outliers_test, latent_data, bae_samples=num_samples))**2)[:,0]
outp_outliers_train = ((outp_outliers_train_raw-make_latent_copies(x_outliers_train, latent_data, bae_samples=num_samples))**2)[:,0]

# outp_inliers_test = ((outp_inliers_test-1)**2)[:,0]
# outp_outliers_test = ((outp_outliers_test-1)**2)[:,0]
# outp_outliers_train = ((outp_outliers_train-1)**2)[:,0]

# outp_inliers_train = ((outp_inliers_train-1))[:,0]
# outp_inliers_test = ((outp_inliers_test-1))[:,0]
# outp_outliers_test = ((outp_outliers_test-1))[:,0]
# outp_outliers_train = ((outp_outliers_train-1))[:,0]

# outp_inliers_train = ((outp_inliers_train))[:,0]
# outp_inliers_test = ((outp_inliers_test))[:,0]
# outp_outliers_test = ((outp_outliers_test))[:,0]
# outp_outliers_train = ((outp_outliers_train))[:,0]


plt.figure()
sns.kdeplot(outp_inliers_train.mean(0).mean(-1))
sns.kdeplot(outp_inliers_test.mean(0).mean(-1))
sns.kdeplot(outp_inliers_valid.mean(0).mean(-1))
sns.kdeplot(outp_outliers_test.mean(0).mean(-1))
sns.kdeplot(outp_outliers_train.mean(0).mean(-1))

plt.figure()
sns.kdeplot(outp_inliers_train.var(0).mean(-1))
sns.kdeplot(outp_inliers_test.var(0).mean(-1))
sns.kdeplot(outp_inliers_valid.var(0).mean(-1))
sns.kdeplot(outp_outliers_test.var(0).mean(-1))
# sns.kdeplot(outp_outliers_train.var(0)[0])

auroc_v1 = calc_auroc(outp_inliers_test.mean(0).mean(-1), outp_outliers_test.mean(0).mean(-1))
auroc_v2 = calc_auroc(outp_inliers_test.var(0).mean(-1), outp_outliers_test.var(0).mean(-1))

print("AUROC MEAN : {:.2f}".format(auroc_v1))
print("AUROC VAR : {:.2f}".format(auroc_v2))

auroc_mean_valid = calc_auroc(outp_inliers_train.mean(0).mean(-1), outp_inliers_valid.mean(0).mean(-1))
auroc_var_valid = calc_auroc(outp_inliers_train.var(0).mean(-1), outp_inliers_valid.var(0).mean(-1))

auroc_mean_test = calc_auroc(outp_inliers_train.mean(0).mean(-1), outp_inliers_test.mean(0).mean(-1))
auroc_var_test = calc_auroc(outp_inliers_train.var(0).mean(-1), outp_inliers_test.var(0).mean(-1))

auroc_mean_validtest = calc_auroc(outp_inliers_valid.mean(0).mean(-1), outp_inliers_test.mean(0).mean(-1))
auroc_var_validtest = calc_auroc(outp_inliers_valid.var(0).mean(-1), outp_inliers_test.var(0).mean(-1))


print("AUROC MEAN (TRAIN-VALID) : {:.2f}".format(auroc_mean_valid))
print("AUROC VAR (TRAIN-VALID) : {:.2f}".format(auroc_var_valid))

print("AUROC MEAN (TRAIN-TEST) : {:.2f}".format(auroc_mean_test))
print("AUROC VAR (TRAIN-TEST) : {:.2f}".format(auroc_var_test))

print("AUROC MEAN (VALID-TEST) : {:.2f}".format(auroc_mean_validtest))
print("AUROC VAR (VALID-TEST) : {:.2f}".format(auroc_var_validtest))

#================================SCATTER============================
outp_inliers_train_mean = outp_inliers_train_raw.mean(0)[0]
outp_inliers_test_mean = outp_inliers_test_raw.mean(0)[0]
outp_outliers_test_mean = outp_outliers_test_raw.mean(0)[0]

outp_inliers_train_mean = outp_inliers_train_raw[0][0]
outp_inliers_test_mean = outp_inliers_test_raw[0][0]
outp_outliers_test_mean = outp_outliers_test_raw[0][0]

outp_inliers_train_var = outp_inliers_train_raw.var(0)[0]
outp_inliers_test_var = outp_inliers_test_raw.var(0)[0]
outp_outliers_test_var = outp_outliers_test_raw.var(0)[0]

plt.figure()
plt.scatter(outp_inliers_train_mean[:,0], outp_inliers_train_mean[:,1])
plt.scatter(outp_inliers_test_mean[:,0], outp_inliers_test_mean[:,1])
plt.scatter(outp_outliers_test_mean[:,0], outp_outliers_test_mean[:,1])

plt.figure()
plt.scatter(outp_inliers_train_var[:,0], outp_inliers_train_var[:,1])
plt.scatter(outp_inliers_test_var[:,0], outp_inliers_test_var[:,1])
plt.scatter(outp_outliers_test_var[:,0], outp_outliers_test_var[:,1])
