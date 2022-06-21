import matplotlib
import matplotlib.pyplot as plt
# import required libraries
import numpy as np
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pyod.utils.data import get_outliers_inliers
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from statsmodels.distributions import ECDF

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import DenseLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood.calc_uncertainty_ood import calc_ood_threshold, convert_hard_predictions
from util.convergence import detect_convergence, plot_convergence
from util.generate_data import generate_moons, generate_circles
from util.helper import generate_grid2d, plot_decision_boundary

bae_set_seed(15)

def bae_predict(bae_ensemble, x, scaler):
    y_nll_samples = bae_ensemble.predict_samples(scaler.transform(x), select_keys=["se"])
    y_nll_samples_mean = y_nll_samples.mean(-1).mean(0)[0]
    # y_nll_samples_std = y_nll_samples.mean(-1).std(0)[0]
    y_nll_samples_std = (y_nll_samples.var(0).mean(-1)[0])**0.5

    return y_nll_samples_mean, y_nll_samples_std

def bae_recon(bae_ensemble, x, scaler):
    y_nll_samples = bae_ensemble.predict_samples(scaler.transform(x), select_keys=["y_mu"])
    y_nll_samples_std = (y_nll_samples.var(0).mean(-1)[0])**0.5

    return y_nll_samples_std


def bae_latent(bae_ensemble, x, scaler):
    _ , y_nll_samples_var = bae_ensemble.predict_latent(scaler.transform(x), transform_pca=False)
    y_nll_samples_var = y_nll_samples_var
    return y_nll_samples_var


def bae_fit(bae_ensemble, x, scaler, num_epoch):
    bae_ensemble.fit(scaler.transform(x), num_epochs=num_epoch)
    print("LOSS "+str(len(bae_ensemble.losses)) +":"+str(np.mean(bae_ensemble.losses)))
    return bae_ensemble

def bae_fit_convergence(bae_ensemble, x, scaler, num_epoch, fast_window=10, slow_window=100, n_stop_points=5):
    if isinstance(x,np.ndarray):
        bae_ensemble.fit(scaler.transform(x), num_epochs=num_epoch)
    else:
        bae_ensemble.fit(x, num_epochs=num_epoch)
    convergence = detect_convergence(bae_ensemble.losses,
                                     fast_window=fast_window,
                                     slow_window=slow_window,
                                     n_stop_points=n_stop_points)
    print("LOSS "+str(len(bae_ensemble.losses)) +":"+str(np.mean(bae_ensemble.losses)))

    return bae_ensemble, convergence



# X_train, Y_train, X_test, Y_test = generate_moons(train_only=False,
#                                                   n_samples=500, # each class
#                                                   test_size=0.5,
#                                                   outlier_class = 1
#                                                   )

X_train, Y_train, X_test, Y_test = generate_circles(train_only=False,
                                                  n_samples=500, # each class
                                                  test_size=0.5,
                                                  outlier_class = 1
                                                  )

outlier_fraction = 0.01

# store outliers and inliers in different numpy arrays
x_outliers_train, x_inliers_train = get_outliers_inliers(X_train, Y_train)
x_outliers_test, x_inliers_test = get_outliers_inliers(X_test,Y_test)
# X_train = x_inliers_train

X = x_inliers_train


# Min-Max scaler
scaler = MinMaxScaler()
scaler.fit(X)

evaluate_range = np.linspace(-4, 4.5, 100)

#====DEFINE BAE==========
span = 0.25
input_dim = X.shape[-1]
activation = "sigmoid"
last_activation = "sigmoid"
likelihood = "gaussian"
weight_decay = 0.0001
num_samples = 10
lr = 0.025
encoder = Encoder([DenseLayers(input_size=input_dim,
                               architecture=[50],
                               output_size=15, activation=activation,last_activation=activation
                               )])

#specify decoder-mu
decoder_mu = infer_decoder(encoder,activation=activation,last_activation=last_activation) #symmetrical to encoder

#combine them into autoencoder
autoencoder = Autoencoder(encoder, decoder_mu)

#convert into BAE-Ensemble
bae_ensemble = BAE_Ensemble(autoencoder=autoencoder,
                            anchored=True,
                            weight_decay=weight_decay,
                            num_samples=num_samples,
                            likelihood=likelihood,
                            learning_rate=lr,
                            verbose=False)

def get_best_lr(min_lr,max_lr, base_lr = 0.01):
    if max_lr > 0.1:
        max_lr_ = base_lr
    else:
        max_lr_ = max_lr
    if min_lr > 0.1:
        min_lr_ = base_lr
    else:
        min_lr_ = min_lr

    med_lr = (max_lr_+min_lr)/2
    return med_lr

# train mu network
train_loader = convert_dataloader(scaler.transform(X), batch_size=250, shuffle=True)
min_lr,max_lr, _ = run_auto_lr_range_v2(train_loader, bae_ensemble, run_full=True, window_size=1, num_epochs=15)
# med_lr = min_lr*1.25
# med_lr = max_lr*0.9
# med_lr = (max_lr+min_lr)/2
# bae_ensemble.set_learning_rate(med_lr)
# bae_ensemble.scheduler_enabled = False

# bae_ensemble.fit(train_loader,num_epochs=1750)
# bae_ensemble = bae_fit(bae_ensemble, X, scaler, num_epoch=500)
# y_nll_samples_mean, y_nll_samples_std = bae_predict(bae_ensemble, np.expand_dims(evaluate_range,1), scaler)

#===PLOT===
cmap ="Greys"
grid_2d, grid = generate_grid2d(X_train, span=span)
nll_mean, nll_std = bae_predict(bae_ensemble, grid_2d, scaler)

plot_threshold = True
outlier_fraction = 0.01
num_epochs_per_cycle = 10
total_epochs = 0

# fit till convergence
num_epochs_per_cycle = 10
fast_window = num_epochs_per_cycle
slow_window = fast_window*10
n_stop_points = 20
cvg = 0

while(cvg == 0):
    _, cvg = bae_fit_convergence(bae_ensemble, train_loader, scaler,
                                 num_epoch=num_epochs_per_cycle,
                                 fast_window=fast_window,
                                 slow_window=slow_window,
                                 n_stop_points=n_stop_points
                                 )
    bae_ensemble.fit()

fig, ax = plt.subplots(1,1)
plot_convergence(losses=bae_ensemble.losses,
                 fast_window=fast_window,
                 slow_window=slow_window,
                 n_stop_points= n_stop_points,
                 ax=ax)

# plot last image
y_nll_samples_mean, y_nll_samples_std = bae_predict(bae_ensemble, grid_2d, scaler)
if plot_threshold:
    y_train_mean, y_train_std = bae_predict(bae_ensemble, X, scaler)
    anomaly_threshold = stats.scoreatpercentile((y_train_mean), 100 - (100 * outlier_fraction))
    plot_decision_boundary(x_inliers_train=x_inliers_train,
                           x_outliers_train=x_outliers_train,
                           grid_2d=grid_2d,
                           Z=y_nll_samples_mean,
                           anomaly_threshold=anomaly_threshold
                           )
else:
    plot_decision_boundary(x_inliers_train=x_inliers_train,
                           x_outliers_train=x_outliers_train,
                           grid_2d=grid_2d,
                           Z=y_nll_samples_mean,
                           )

#==== recon var
y_nll_samples_mean, y_nll_samples_std = bae_predict(bae_ensemble, grid_2d, scaler)


if plot_threshold:
    y_train_mean, y_train_std = bae_predict(bae_ensemble, X, scaler)
    anomaly_threshold = stats.scoreatpercentile((y_train_std), 100 - (100 * outlier_fraction))
    plot_decision_boundary(x_inliers_train=x_inliers_train,
                           x_outliers_train=x_outliers_train,
                           grid_2d=grid_2d,
                           Z=y_nll_samples_std,
                           anomaly_threshold=anomaly_threshold
                           )
else:
    plot_decision_boundary(x_inliers_train=x_inliers_train,
                           x_outliers_train=x_outliers_train,
                           grid_2d=grid_2d,
                           Z=y_nll_samples_std,
                           )


#=== latent ====

# y_latent_var = bae_latent(bae_ensemble, grid_2d, scaler).mean(-1)

y_latent = bae_ensemble.predict_latent_samples(scaler.transform(grid_2d))
y_latent_var = y_latent.var(0).mean(-1)

if plot_threshold:
    # y_latent_var_train = bae_latent(bae_ensemble, X, scaler).mean(-1)
    y_latent = bae_ensemble.predict_latent_samples(scaler.transform(X))
    y_latent_var_train = y_latent.var(0).mean(-1)

    anomaly_threshold = stats.scoreatpercentile((y_latent_var_train), 100 - (100 * outlier_fraction))
    plot_decision_boundary(x_inliers_train=x_inliers_train,
                           x_outliers_train=x_outliers_train,
                           grid_2d=grid_2d,
                           Z=y_latent_var,
                           anomaly_threshold=anomaly_threshold
                           )
else:
    plot_decision_boundary(x_inliers_train=x_inliers_train,
                           x_outliers_train=x_outliers_train,
                           grid_2d=grid_2d,
                           Z=y_latent_var,
                           )

#==== recon var
y_recon_samples_std = bae_recon(bae_ensemble, grid_2d, scaler)

if plot_threshold:
    y_recon_samples_train = bae_recon(bae_ensemble, X, scaler)

    anomaly_threshold = stats.scoreatpercentile((y_recon_samples_train), 100 - (100 * outlier_fraction))
    plot_decision_boundary(x_inliers_train=x_inliers_train,
                           x_outliers_train=x_outliers_train,
                           grid_2d=grid_2d,
                           Z=y_recon_samples_std,
                           anomaly_threshold=anomaly_threshold
                           )
else:
    plot_decision_boundary(x_inliers_train=x_inliers_train,
                           x_outliers_train=x_outliers_train,
                           grid_2d=grid_2d,
                           Z=y_recon_samples_std,
                           )


#===ECDF=== HERE HERE

ecdf_recon = ECDF(y_recon_samples_train)
ecdf_nll_std = ECDF(y_train_std)
ecdf_nll_mu = ECDF(y_train_mean)

def bae_pred_all(bae_ensemble, x, scaler, return_mean=False):
    scaled_x = scaler.transform(x)
    y_pred_samples = bae_ensemble.predict_samples(scaled_x, select_keys=["se", "y_mu"])
    y_latent_samples = bae_ensemble.predict_latent_samples(scaled_x)

    y_nll_mean = y_pred_samples.mean(0)[0]
    y_nll_var = y_pred_samples.var(0)[0]
    y_recon_var = y_pred_samples.var(0)[1]
    y_latent_mean = y_latent_samples.mean(0)
    y_latent_var = y_latent_samples.var(0)
    y_latent_weighted = y_latent_mean/(y_latent_var**0.5)

    if return_mean:
        y_nll_mean =y_nll_mean.mean(-1)
        y_nll_var = y_nll_var.mean(-1)
        y_recon_var = y_recon_var.mean(-1)

    return {"nll_mean": y_nll_mean,
            "nll_var":y_nll_var,
            "recon_var":y_recon_var,
            "latent_mean": y_latent_mean,
            "latent_var": y_latent_var,
            "latent_weighted":y_latent_weighted
            }

def scale_score(nll_train, nll_test):
    ecdf = ECDF(nll_train)
    return ecdf(nll_test)


raw_preds_train = bae_pred_all(bae_ensemble, x_inliers_train, scaler)
raw_preds_test_outliers = bae_pred_all(bae_ensemble, x_outliers_test, scaler)
raw_preds_test_inliers = bae_pred_all(bae_ensemble, x_inliers_test, scaler)

nll_train_mean = raw_preds_train["nll_mean"]
nll_test_outlier_mean = raw_preds_test_outliers["nll_mean"]
nll_test_inlier_mean = raw_preds_test_inliers["nll_mean"]

ecdf_nll_mean = ECDF(nll_train_mean.mean(1))
scaled_nll_test_outlier_mean = ecdf_nll_mean(nll_test_outlier_mean.mean(1))
scaled_nll_test_inlier_mean = ecdf_nll_mean(nll_test_inlier_mean.mean(1))

#==== plot rescaled====
raw_preds_grid2d = bae_pred_all(bae_ensemble, grid_2d, scaler)
raw_preds_train = bae_pred_all(bae_ensemble, x_inliers_train, scaler)
raw_preds_test_outliers = bae_pred_all(bae_ensemble, x_outliers_test, scaler)
raw_preds_test_inliers = bae_pred_all(bae_ensemble, x_inliers_test, scaler)
key = "nll_mean"
key = "nll_var"

train_scores = raw_preds_train[key].mean(-1)
z = raw_preds_grid2d[key].mean(-1)
z_scaled = scale_score(train_scores,z)

if plot_threshold:
    anomaly_threshold = stats.scoreatpercentile((train_scores), 100 - (100 * outlier_fraction))
    plot_decision_boundary(x_inliers_train=x_inliers_train,
                           x_outliers_train=x_outliers_train,
                           grid_2d=grid_2d,
                           Z=z_scaled,
                           anomaly_threshold=anomaly_threshold
                           )
else:
    plot_decision_boundary(x_inliers_train=x_inliers_train,
                           x_outliers_train=x_outliers_train,
                           grid_2d=grid_2d,
                           Z=z_scaled,
                           )

train_scores1 = raw_preds_train["nll_mean"].mean(-1)
z1 = raw_preds_grid2d["nll_mean"].mean(-1)
z_scaled1 = scale_score(train_scores1,z1)

train_scores2 = raw_preds_train["nll_var"].mean(-1)
z2 = raw_preds_grid2d["nll_var"].mean(-1)
z_scaled2 = scale_score(train_scores2,z2)

z_scaled3 = (z_scaled1+z_scaled2)/2
z_scaled4 = ((z_scaled1*(1-z_scaled1))+(z_scaled2*(1-z_scaled2)))/2
# z_scaled4 = (z_scaled4**0.5)*2
# z_scaled4 = ((np.abs(0.5-z_scaled1))+(np.abs(0.5-z_scaled2)))/2

plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       grid_2d=grid_2d,
                       Z=z_scaled1,
                       )

plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       grid_2d=grid_2d,
                       Z=z_scaled2,
                       )

plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       grid_2d=grid_2d,
                       Z=z_scaled3,
                       )

plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       grid_2d=grid_2d,
                       Z=z_scaled4,
                       )

# plot_decision_boundary(x_inliers_train=x_inliers_train,
#                        x_outliers_train=x_outliers_train,
#                        grid_2d=grid_2d,
#                        Z=((z_scaled1*(1-z_scaled1))**0.5)*2,
#                        )

#=====================uncertainty of OOD ========================

raw_preds_grid2d = bae_pred_all(bae_ensemble, grid_2d, scaler)
raw_preds_train = bae_pred_all(bae_ensemble, x_inliers_train, scaler)
raw_preds_test_outliers = bae_pred_all(bae_ensemble, x_outliers_test, scaler)
raw_preds_test_inliers = bae_pred_all(bae_ensemble, x_inliers_test, scaler)
key = "nll_mean"
key = "nll_var"


threshold = calc_ood_threshold(training_scores=raw_preds_train["nll_mean"].mean(-1),perc_threshold=99)

hard_preds = convert_hard_predictions(test_scores=raw_preds_train["nll_mean"].mean(-1), ood_threshold=threshold)

def bae_predict_ood_v1(bae_ensemble, x_train, x_test, scaler, keys=["nll_mean","nll_var"], perc_threshold=99):
    """
    Combine nll_mean and nll_var method
    """
    preds_train = bae_pred_all(bae_ensemble, x_train, scaler, return_mean=True)
    preds_test = bae_pred_all(bae_ensemble, x_test, scaler, return_mean=True)
    thresholds = {key:calc_ood_threshold(training_scores=preds_train[key],perc_threshold=perc_threshold) for key in keys}
    hard_preds = np.array([convert_hard_predictions(test_scores=preds_test[key], ood_threshold=thresholds[key]) for key in
                  keys])

    return hard_preds

def bae_predict_ood_v2(bae_ensemble, x_train, x_test, scaler, perc_threshold=99):
    """
    Apply Threshold on each samples. Threshold obtained from each BAE sample's training scores.
    """

    preds_train = bae_ensemble.predict_samples(scaler.transform(x_train), select_keys=["se"]).mean(-1)[:,0]
    preds_test = bae_ensemble.predict_samples(scaler.transform(x_test), select_keys=["se"]).mean(-1)[:,0]

    thresholds = [calc_ood_threshold(training_scores=preds_train_i,perc_threshold=perc_threshold) for preds_train_i in preds_train]
    hard_preds = np.array([convert_hard_predictions(test_scores=preds_test_i, ood_threshold=thresholds_i) for preds_test_i,thresholds_i in
                  zip(preds_test,thresholds)])

    return hard_preds

def bae_predict_ood_v3(bae_ensemble, x_train, x_test, scaler, perc_threshold=99):
    """
    Apply Threshold on each samples. Threshold obtained from mean of training scores.
    """

    preds_train = bae_ensemble.predict_samples(scaler.transform(x_train), select_keys=["se"]).mean(-1)[:,0]
    preds_test = bae_ensemble.predict_samples(scaler.transform(x_test), select_keys=["se"]).mean(-1)[:,0]

    threshold = calc_ood_threshold(training_scores=preds_train.mean(0),perc_threshold=perc_threshold)
    hard_preds = np.array([convert_hard_predictions(test_scores=preds_test_i, ood_threshold=threshold) for preds_test_i in
                  preds_test])

    return hard_preds


# hard_preds_inlier = bae_predict_ood_v1(bae_ensemble, x_inliers_train, x_inliers_test, scaler, keys=["nll_mean","nll_var"])
# hard_preds_grid2d = bae_predict_ood_v1(bae_ensemble, x_inliers_train, grid_2d, scaler, keys=["nll_mean","nll_var"], perc_threshold = 99)
hard_preds_grid2d = bae_predict_ood_v2(bae_ensemble, X, grid_2d, scaler, perc_threshold = 99)
# hard_preds_grid2d = bae_predict_ood_v3(bae_ensemble, x_inliers_train, grid_2d, scaler, perc_threshold = 95)

fig, (ax1,ax2) = plt.subplots(1,2)
plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       grid_2d=grid_2d,
                       Z=hard_preds_grid2d.mean(0), fig=fig, ax=ax1,
                       anomaly_threshold=0.1
                       )

plot_decision_boundary(x_inliers_train=x_inliers_train,
                       x_outliers_train=x_outliers_train,
                       grid_2d=grid_2d,
                       Z=(hard_preds_grid2d.std(0)*2), fig=fig, ax=ax2,
                       )

plt.figure()
plt.hist(bae_predict_ood_v1(bae_ensemble, x_inliers_train, x_inliers_train, scaler, perc_threshold = 99).mean(0), density=True,bins=20)
plt.hist(bae_predict_ood_v1(bae_ensemble, x_inliers_train, x_inliers_test, scaler, perc_threshold = 99).mean(0), density=True,bins=20)
plt.hist(bae_predict_ood_v1(bae_ensemble, x_inliers_train, x_outliers_test, scaler, perc_threshold = 99).mean(0), density=True,bins=20)

plt.figure()
sns.kdeplot(bae_predict_ood_v2(bae_ensemble, x_inliers_train, x_inliers_train, scaler, perc_threshold = 99).mean(0))
sns.kdeplot(bae_predict_ood_v2(bae_ensemble, x_inliers_train, x_inliers_test, scaler, perc_threshold = 99).mean(0))
sns.kdeplot(bae_predict_ood_v2(bae_ensemble, x_inliers_train, x_outliers_test, scaler, perc_threshold = 99).mean(0))



#=====================latent level============================
latent_train_mean = raw_preds_train["latent_mean"]
latent_outliers_mean = raw_preds_test_outliers["latent_mean"]
latent_inliers_mean = raw_preds_test_inliers["latent_mean"]

# latent_train_mean = raw_preds_train["latent_var"]
# latent_outliers_mean = raw_preds_test_outliers["latent_var"]
# latent_inliers_mean = raw_preds_test_inliers["latent_var"]

# shade
scale = 1
n_latents = np.arange(latent_train_mean.shape[-1]) + 1
plt.figure()
plt.plot(n_latents, latent_train_mean.mean(0))
plt.fill_between(n_latents,
                 latent_train_mean.mean(0) - scale * latent_train_mean.std(0),
                 latent_train_mean.mean(0) + scale * latent_train_mean.std(0),
                 alpha=0.45
                 )

plt.plot(n_latents, latent_inliers_mean.mean(0))
plt.fill_between(n_latents,
                 latent_inliers_mean.mean(0) - scale * latent_inliers_mean.std(0),
                 latent_inliers_mean.mean(0) + scale * latent_inliers_mean.std(0),
                 alpha=0.45
                 )

plt.plot(n_latents, latent_outliers_mean.mean(0))
plt.fill_between(n_latents,
                 latent_outliers_mean.mean(0) - scale * latent_outliers_mean.std(0),
                 latent_outliers_mean.mean(0) + scale * latent_outliers_mean.std(0),
                 alpha=0.45
                 )

# raw samples
for i in latent_train_mean:
    plt.plot(n_latents, i, color="tab:blue", alpha=0.25)
for i in latent_inliers_mean:
    plt.plot(n_latents, i, color="tab:orange", alpha=0.25)
for i in latent_outliers_mean:
    plt.plot(n_latents, i, color="tab:green", alpha=0.25)



latent_train_mean = raw_preds_train["latent_mean"]
latent_outliers_mean = raw_preds_test_outliers["latent_mean"]
latent_inliers_mean = raw_preds_test_inliers["latent_mean"]

# latent_train_mean = raw_preds_train["latent_var"]
# latent_outliers_mean = raw_preds_test_outliers["latent_var"]
# latent_inliers_mean = raw_preds_test_inliers["latent_var"]

# latent_train_mean = raw_preds_train["latent_mean"]/raw_preds_train["latent_var"]
# latent_outliers_mean = raw_preds_test_outliers["latent_mean"]/raw_preds_test_outliers["latent_var"]
# latent_inliers_mean = raw_preds_test_inliers["latent_mean"]/raw_preds_test_inliers["latent_var"]


latent_distance_train = np.abs(latent_train_mean- latent_train_mean.mean(0))
latent_distance_inliers = np.abs(latent_inliers_mean- latent_train_mean.mean(0))
latent_distance_outliers = np.abs(latent_outliers_mean- latent_train_mean.mean(0))

plt.figure()
plt.hist(latent_distance_train.mean(-1), density=True)
plt.hist(latent_distance_inliers.mean(-1), density=True)
plt.hist(latent_distance_outliers.mean(-1), density=True)

plt.figure()
sns.kdeplot(latent_distance_train.mean(-1))
sns.kdeplot(latent_distance_inliers.mean(-1))
sns.kdeplot(latent_distance_outliers.mean(-1))

#======LATENT FROM PCA=====


pca_train_mu,pca_train_var = bae_ensemble.predict_latent(x_inliers_train)
pca_inlier_mu,pca_inlier_var = bae_ensemble.predict_latent(x_inliers_test)
pca_outlier_mu,pca_outlier_var = bae_ensemble.predict_latent(x_outliers_test)


plt.figure()
plt.scatter(pca_train_mu[:,0],pca_train_mu[:,1])
plt.scatter(pca_inlier_mu[:,0],pca_inlier_mu[:,1])
plt.scatter(pca_outlier_mu[:,0],pca_outlier_mu[:,1])

# plt.figure()
# plt.scatter(pca_train_var[:,0],pca_train_var[:,1])
# plt.scatter(pca_inlier_var[:,0],pca_inlier_var[:,1])
# plt.scatter(pca_outlier_var[:,0],pca_outlier_var[:,1])
#












