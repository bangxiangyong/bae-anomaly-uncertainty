from pyod.utils import precision_n_scores
from pyod.utils.data import generate_data, get_outliers_inliers
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#generate random data with two features
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import DenseLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from util.generate_data import generate_moons
from util.helper import get_hard_predictions, generate_grid2d, plot_decision_boundary, convert_percentile

bae_set_seed(123)
# X_train, Y_train, X_test, Y_test = generate_data(n_train=200,train_only=False, n_features=2)
X_train, Y_train, X_test, Y_test = generate_moons(train_only=False,
                                                  n_samples=800,
                                                  test_size=0.5,
                                                  outlier_class = 1
                                                  )

full_X = X_train.copy()
# by default the outlier fraction is 0.1 in generate data function
outlier_fraction = 0.05

# store outliers and inliers in different numpy arrays
x_outliers_train, x_inliers_train = get_outliers_inliers(X_train, Y_train)
x_outliers_test, x_inliers_test = get_outliers_inliers(X_test,Y_test)

X_train = x_inliers_train

#separate the two features and use it to plot the data
# F1 = X_train[:,[0]].reshape(-1,1)
# F2 = X_train[:,[1]].reshape(-1,1)

# create a meshgrid
# xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
#
# # scatter plot
# plt.scatter(F1,F2)
# plt.xlabel('F1')
# plt.ylabel('F2')

#====================BAE===========================

# USE BAE
mid_activation = "relu"
input_dim = X_train.shape[-1]
encoder = Encoder([DenseLayers(input_size=input_dim,
                               architecture=[125],
                               output_size=25, activation=mid_activation, last_activation=mid_activation
                               )])

#specify decoder-muyt
decoder_mu = infer_decoder(encoder,activation=mid_activation,last_activation="sigmoid") #symmetrical to encoder

#combine them into autoencoder
autoencoder = Autoencoder(encoder, decoder_mu)

#convert into BAE-Ensemble
bae_ensemble = BAE_Ensemble(autoencoder=autoencoder,
                            anchored=True,
                            weight_decay=0.0001,
                            num_samples= 10,  likelihood="bernoulli")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# train mu network
train_loader = convert_dataloader(X_train_scaled, batch_size=150, shuffle=True)
run_auto_lr_range_v2(train_loader, bae_ensemble, run_full=True, window_size=2, num_epochs=10)
bae_ensemble.fit(train_loader,num_epochs=500)


train_nll = bae_ensemble.predict_samples(X_train_scaled, select_keys=["se"])

# get raw train scores
raw_train_scores = train_nll.mean(0)[0].mean(-1)
raw_train_scores = np.exp(raw_train_scores)
raw_train_scores_perc = convert_percentile(raw_train_scores, raw_train_scores) # convert to pct

# get threshold
anomaly_threshold = stats.scoreatpercentile(raw_train_scores_perc,100-(100*outlier_fraction))

# apply threshold to get hard predictions
hard_pred = get_hard_predictions(raw_train_scores_perc, anomaly_threshold)

# visualise grid
grid_2d, grid = generate_grid2d(full_X, span=1)
y_pred_grid = bae_ensemble.predict_samples(scaler.transform(grid_2d), select_keys=["se"]).mean(0)[0].mean(-1)
y_pred_grid = np.exp(y_pred_grid)
y_pred_grid = convert_percentile(raw_train_scores, y_pred_grid) # convert to pct

plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       x_inliers_test=x_inliers_test,
                       x_outliers_test=x_outliers_test,
                       grid_2d=grid_2d,
                       Z=y_pred_grid,
                       anomaly_threshold=anomaly_threshold)

plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       x_inliers_test=x_inliers_test,
                       x_outliers_test=x_outliers_test,
                       grid_2d=grid_2d,
                       Z=y_pred_grid)



# plot percentile conversion (ECDF)
plt.figure()
plt.scatter(raw_train_scores, convert_percentile(raw_train_scores, raw_train_scores))
plt.xlabel("Raw scores")
plt.ylabel("ECDF")

# evaluation of performance
y_nll_test = bae_ensemble.predict_samples(scaler.transform(X_test), select_keys=["se"]).mean(0).mean(-1)[0]
y_nll_test = np.exp(y_nll_test)
y_nll_test_perc = convert_percentile(raw_train_scores, y_nll_test) # convert to pct

hard_pred_test = get_hard_predictions(y_nll_test_perc, anomaly_threshold)

# AUROC & MCC
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_nll_test, pos_label=1)
auroc_test = metrics.auc(fpr, tpr)
mcc_test = matthews_corrcoef(Y_test, hard_pred_test)
prec_n_test = precision_n_scores(Y_test, y_nll_test)
print("AUROC: {:.2f}".format(auroc_test))
print("MCC: {:.2f}".format(mcc_test))
print("PREC@N: {:.2f}".format(prec_n_test))


def hard_predict(raw_train_scores, y_nll_test, outlier_fraction=0.1):
    y_nll_test_perc = convert_percentile(raw_train_scores, y_nll_test)  # convert to pct
    raw_train_scores_perc = convert_percentile(raw_train_scores, raw_train_scores)  # convert to pct
    anomaly_threshold = stats.scoreatpercentile(raw_train_scores_perc, 100 - (100 * outlier_fraction))

    hard_pred_test = get_hard_predictions(y_nll_test_perc, anomaly_threshold)

    return hard_pred_test


y_nll_grid2d = np.exp(bae_ensemble.predict_samples(scaler.transform(grid_2d), select_keys=["se"]).mean(-1)[:,0])
raw_train_scores = np.exp(train_nll.mean(-1)[:,0])

hard_pred_grid2d_samples = np.array([hard_predict(raw_train_scores_i, y_nll_test_i,
                                                outlier_fraction=outlier_fraction)
                                   for y_nll_test_i,raw_train_scores_i in zip(y_nll_grid2d,raw_train_scores)])
# hard_pred_grid2d_samples = np.array([convert_percentile(raw_train_scores_i, y_nll_test_i)
#                                      for raw_train_scores_i,y_nll_test_i in zip(raw_train_scores,y_nll_grid2d)])

uncertainty_grid2d = hard_pred_grid2d_samples.std(0)
# uncertainty_grid2d = np.array([p_i*(1-p_i) for p_i in hard_pred_grid2d_samples]).mean(0)
# uncertainty_grid2d = hard_pred_grid2d_samples.mean(0)
# uncertainty_grid2d = hard_pred_grid2d_samples.mean(0)*(1-hard_pred_grid2d_samples.mean(0))
# uncertainty_grid2d = np.percentile(hard_pred_grid2d_samples, 50,axis=0)*(1-np.percentile(hard_pred_grid2d_samples, 50,axis=0))
# uncertainty_grid2d = np.abs(np.percentile(hard_pred_grid2d_samples, 75,axis=0)-np.percentile(hard_pred_grid2d_samples, 25,axis=0))
# uncertainty_grid2d = np.percentile(hard_pred_grid2d_samples, 75,axis=0)

# plot uncertainty
plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       x_inliers_test=x_inliers_test,
                       x_outliers_test=x_outliers_test,
                       grid_2d=grid_2d,
                       Z=uncertainty_grid2d,
                       cmap="Greys"
                       )

entropy = hard_pred_grid2d_samples.mean(0)*(1-hard_pred_grid2d_samples.mean(0))
pred_std = hard_pred_grid2d_samples.std(0)
plt.figure()
plt.scatter(entropy,pred_std)

# filter on uncertainty
y_nll_test = np.exp(bae_ensemble.predict_samples(scaler.transform(X_test), select_keys=["se"]).mean(-1)[:,0])
raw_train_scores = np.exp(train_nll.mean(-1)[:,0])

hard_pred_test_samples = np.array([hard_predict(raw_train_scores_i, y_nll_test_i,
                                                outlier_fraction=outlier_fraction)
                                   for y_nll_test_i,raw_train_scores_i in zip(y_nll_test,raw_train_scores)])
# uncertainty_test = hard_pred_test_samples.std(0)
uncertainty_test = hard_pred_test_samples.mean(0)

prec_n_test_unc = []
ulim = uncertainty_test.max()-0.01
unc_range = np.linspace(uncertainty_test.min(),ulim,100)
for unc_lim in np.linspace(uncertainty_test.min(),ulim,100):
    unc_arg = np.argwhere(uncertainty_test>=unc_lim).flatten()
    prec_n_test_unc += [precision_n_scores(Y_test[unc_arg], y_nll_test.mean(0)[unc_arg])]

plt.figure()
plt.plot(unc_range,prec_n_test_unc)


#=============================================================================================


y_nll_grid2d = np.exp(bae_ensemble.predict_samples(scaler.transform(grid_2d), select_keys=["se"]).mean(-1)[:,0])
raw_train_scores = np.exp(train_nll.mean(-1)[:,0])

# hard_pred_grid2d_samples = np.array([hard_predict(raw_train_scores_i, y_nll_test_i,
#                                                 outlier_fraction=outlier_fraction)
                                   # for y_nll_test_i,raw_train_scores_i in zip(y_nll_grid2d,raw_train_scores)])
hard_pred_grid2d_samples = np.array([convert_percentile(raw_train_scores_i, y_nll_test_i)
                                     for raw_train_scores_i,y_nll_test_i in zip(raw_train_scores,y_nll_grid2d)])/100

uncertainty_grid2d = (hard_pred_grid2d_samples*(1-hard_pred_grid2d_samples)).mean(0)
# uncertainty_grid2d = 1-np.array([p_i*(1-p_i) for p_i in hard_pred_grid2d_samples]).mean(0)
# uncertainty_grid2d = hard_pred_grid2d_samples.mean(0)

# uncertainty_grid2d = hard_pred_grid2d_samples.mean(0)*(1-hard_pred_grid2d_samples.mean(0))
# uncertainty_grid2d = np.percentile(hard_pred_grid2d_samples, 50,axis=0)*(1-np.percentile(hard_pred_grid2d_samples, 50,axis=0))
# uncertainty_grid2d = np.abs(np.percentile(hard_pred_grid2d_samples, 75,axis=0)-np.percentile(hard_pred_grid2d_samples, 25,axis=0))
# uncertainty_grid2d = np.percentile(hard_pred_grid2d_samples, 75,axis=0)
# uncertainty_grid2d = np.array([p_i*(1-p_i) for p_i in hard_pred_grid2d_samples]).mean(0)

# plot uncertainty
plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       x_inliers_test=x_inliers_test,
                       x_outliers_test=x_outliers_test,
                       grid_2d=grid_2d,
                       Z=uncertainty_grid2d,
                       cmap="Greys"
                       )


#=================================

# get raw train scores
raw_train_scores = train_nll.mean(0)[0].mean(-1)
raw_train_scores = np.exp(raw_train_scores)
anomaly_threshold = stats.scoreatpercentile(raw_train_scores,100-(100*outlier_fraction))

# raw_train_scores_perc = convert_percentile(raw_train_scores, raw_train_scores) # convert to pct
#
# # get threshold
# anomaly_threshold = stats.scoreatpercentile(raw_train_scores_perc,100-(100*outlier_fraction))

# apply threshold to get hard predictions
hard_pred = [get_hard_predictions(raw_train_scores_perc, anomaly_threshold)]

# visualise grid
grid_2d, grid = generate_grid2d(full_X, span=1)
y_pred_grid = bae_ensemble.predict_samples(scaler.transform(grid_2d), select_keys=["se"]).mean(-1)[:,0]
y_pred_grid = np.exp(y_pred_grid)
# y_pred_grid = convert_percentile(raw_train_scores, y_pred_grid)/100 # convert to pct
# y_pred_grid = y_pred_grid*(1-y_pred_grid)
y_pred_grid = np.array([get_hard_predictions(pred_i, anomaly_threshold) for pred_i in y_pred_grid])
y_pred_grid = y_pred_grid.mean(0)
# y_pred_grid = y_pred_grid.std(0)
# y_pred_grid = (y_pred_grid *(1-y_pred_grid))

plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       x_inliers_test=x_inliers_test,
                       x_outliers_test=x_outliers_test,
                       grid_2d=grid_2d,
                       Z=y_pred_grid)

#=======================================


y_nll_grid2d = np.exp(bae_ensemble.predict_samples(scaler.transform(grid_2d), select_keys=["se"]).mean(-1)[:,0])
raw_train_scores = np.exp(train_nll.mean(-1)[:,0])

hard_pred_grid2d_samples = np.array([hard_predict(raw_train_scores_i, y_nll_test_i,
                                                outlier_fraction=outlier_fraction)
                                   for y_nll_test_i,raw_train_scores_i in zip(y_nll_grid2d,raw_train_scores)])
# hard_pred_grid2d_samples = np.array([convert_percentile(raw_train_scores_i, y_nll_test_i)
#                                      for raw_train_scores_i,y_nll_test_i in zip(raw_train_scores,y_nll_grid2d)])

uncertainty_grid2d = hard_pred_grid2d_samples.mean(0)
# uncertainty_grid2d = np.percentile(hard_pred_grid2d_samples, 50,axis=0)*(1-np.percentile(hard_pred_grid2d_samples, 50,axis=0))
# uncertainty_grid2d = np.abs(np.percentile(hard_pred_grid2d_samples, 75,axis=0)-np.percentile(hard_pred_grid2d_samples, 25,axis=0))
# uncertainty_grid2d = np.percentile(hard_pred_grid2d_samples, 75,axis=0)

# plot uncertainty
plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       x_inliers_test=x_inliers_test,
                       x_outliers_test=x_outliers_test,
                       grid_2d=grid_2d,
                       Z=uncertainty_grid2d,
                       cmap="Greys"
                       )









