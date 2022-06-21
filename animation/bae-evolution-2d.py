import matplotlib
import matplotlib.pyplot as plt
# import required libraries
import numpy as np
from matplotlib.animation import FuncAnimation
from pyod.utils.data import get_outliers_inliers
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import DenseLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from util.generate_data import generate_moons
from util.helper import generate_grid2d, plot_decision_boundary

bae_set_seed(15)

X_train, Y_train, X_test, Y_test = generate_moons(train_only=False,
                                                  n_samples=500,
                                                  test_size=0.5,
                                                  outlier_class = 1
                                                  )

# full_X = X_train.copy()
# by default the outlier fraction is 0.1 in generate data function
outlier_fraction = 0.01

# store outliers and inliers in different numpy arrays
x_outliers_train, x_inliers_train = get_outliers_inliers(X_train, Y_train)
# x_outliers_test, x_inliers_test = get_outliers_inliers(X_test,Y_test)
# X_train = x_inliers_train

X = x_inliers_train

def bae_predict(bae_ensemble, x, scaler):
    y_nll_samples = bae_ensemble.predict_samples(scaler.transform(x), select_keys=["se"])
    y_nll_samples_mean = y_nll_samples.mean(-1).mean(0)[0]
    y_nll_samples_std = y_nll_samples.mean(-1).std(0)[0]

    return y_nll_samples_mean, y_nll_samples_std

def bae_fit(bae_ensemble, x, scaler, num_epoch):
    bae_ensemble.fit(scaler.transform(x), num_epochs=num_epoch)
    return bae_ensemble

# Min-Max scaler
scaler = MinMaxScaler()
scaler.fit(X)

evaluate_range = np.linspace(-4, 4.5, 100)

# model
span = 0.25
input_dim = X.shape[-1]
activation = "sigmoid"
last_activation = "sigmoid"
likelihood = "bernoulli"
weight_decay = 0.0001
num_samples = 5
lr = 0.025
encoder = Encoder([DenseLayers(input_size=input_dim,
                               architecture=[50,50],
                               output_size=3, activation=activation,last_activation=activation
                               )])

#specify decoder-mu
decoder_mu = infer_decoder(encoder,activation=activation,last_activation=last_activation) #symmetrical to encoder

#combine them into autoencoder
autoencoder = Autoencoder(encoder, decoder_mu)

#convert into BAE-Ensemble
bae_ensemble = BAE_Ensemble(autoencoder=autoencoder,
                            anchored=True,
                            weight_decay=weight_decay,
                            num_samples=num_samples, likelihood=likelihood, learning_rate=lr)


# train mu network
train_loader = convert_dataloader(scaler.transform(X), batch_size=250, shuffle=True)
min_lr,max_lr, _ = run_auto_lr_range_v2(train_loader, bae_ensemble, run_full=True, window_size=1, num_epochs=15)
# med_lr = min_lr*1.25
# med_lr = max_lr*0.9
med_lr = (max_lr+min_lr)/2
bae_ensemble.set_learning_rate(med_lr)
bae_ensemble.scheduler_enabled = False

# bae_ensemble.fit(X,num_epochs=100)
# bae_ensemble = bae_fit(bae_ensemble, X, scaler, num_epoch=500)
# y_nll_samples_mean, y_nll_samples_std = bae_predict(bae_ensemble, np.expand_dims(evaluate_range,1), scaler)

cmap ="Greys"
grid_2d, grid = generate_grid2d(X_train, span=span)

nll_mean, nll_std = bae_predict(bae_ensemble, grid_2d, scaler)


# plot_decision_boundary(x_inliers_train=x_inliers_train,
#                        x_outliers_train=x_outliers_train,
#                        grid_2d=grid_2d,
#                        Z=nll_mean)
#
#


fig, ax = plt.subplots(1,1)
line, = ax.plot([], [], lw=2)
scat = ax.scatter([], [])
between = ax.fill_between(evaluate_range,evaluate_range,evaluate_range)
plot_threshold = True
outlier_fraction = 0.01

num_epochs_per_cycle = 10
total_epochs = 0


def init():
    line.set_data([], [])
    scat.set_offsets([])

    return line, scat,

def animate(i):
    # global bae_ensemble
    ax.clear()
    if i >1:
        bae_ensemble.fit(train_loader,num_epochs=num_epochs_per_cycle)
        # bae_fit(bae_ensemble,X,scaler,num_epochs_per_cycle)

    total_epochs = i*num_epochs_per_cycle
    print("EPOCHS: "+str(total_epochs))
    ax.set_title("EPOCHS: "+str(total_epochs))
    # else:
    #     bae_ensemble.reset_parameters()
    y_nll_samples_mean, y_nll_samples_std = bae_predict(bae_ensemble, grid_2d, scaler)

    reshaped_Z = y_nll_samples_mean.reshape(100, 100)
    # reshaped_Z = np.log(reshaped_Z)

    contour = ax.contourf(grid[0], grid[1], reshaped_Z, levels=35, cmap=cmap)
    # plt.colorbar(contour)

    inlier_train = ax.scatter(x_inliers_train[:, 0], x_inliers_train[:, 1], c='tab:green', s=20, edgecolor='k')
    outlier_train = ax.scatter(x_outliers_train[:, 0], x_outliers_train[:, 1], c='tab:orange', s=20, edgecolor='k')

    if plot_threshold:
        y_train_mean, y_train_std = bae_predict(bae_ensemble, X, scaler)
        # anomaly_threshold = stats.scoreatpercentile(np.log(y_train_mean), 100 - (100 * outlier_fraction))
        anomaly_threshold = stats.scoreatpercentile((y_train_mean), 100 - (100 * outlier_fraction))

        a = ax.contour(grid[0], grid[1], reshaped_Z, levels=[anomaly_threshold], linewidths=1.5, colors='red')
        ax.contourf(grid[0], grid[1], reshaped_Z, levels=[reshaped_Z.min(), anomaly_threshold], colors='tab:blue',alpha=0.5)


    return outlier_train, inlier_train, contour, a


save_gif = False

if save_gif:
    matplotlib.use("Agg")
anim = FuncAnimation(fig, animate, frames=50, interval=25, blit=False)

anim_title = 'bae_2d_wide_moons_'+activation
if save_gif:
    anim.save(anim_title+'.gif', writer='imagemagick')
    # anim.save(anim_title+'.mp4', writer='ffmpeg')
plt.show()

plt.figure()
plt.plot(bae_ensemble.losses)

import pickle
pickle.dump(bae_ensemble.losses,open("loss.p","wb"))
