import matplotlib
import matplotlib.pyplot as plt
# import required libraries
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import DenseLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed


bae_set_seed(13)


def bae_predict(bae_ensemble, x, scaler, key="se"):
    y_nll_samples = bae_ensemble.predict_samples(scaler.transform(x), select_keys=[key])
    y_nll_samples_mean = y_nll_samples.mean(-1).mean(0)[0]
    y_nll_samples_std = y_nll_samples.mean(-1).std(0)[0]

    return y_nll_samples_mean, y_nll_samples_std


def bae_fit(bae_ensemble, x, scaler, num_epoch):
    bae_ensemble.fit(scaler.transform(x), num_epochs=num_epoch)
    return bae_ensemble

# sample data
n_samples= 100
X1 = np.random.normal(loc=-3.0,scale=0.15,size=n_samples)
X2 = np.random.normal(loc=-0.0,scale=0.05,size=n_samples)
X3 = np.random.normal(loc=3.0,scale=0.25,size=n_samples)
# X = np.expand_dims(np.concatenate((X1,X2,X3)),1)
X = np.expand_dims(np.concatenate((X1,X3)),1)
scaler = MinMaxScaler()
scaler.fit(X)
# scaler.data_max_ = 1.5
# scaler.data_min_ = -1.5
X = X*0.75
evaluate_range = np.linspace(-4, 4.5, 100)

# model
input_dim = 1
activation = "leakyrelu"
# last_activation = "relu"
last_activation = "tanh"
likelihood = "gaussian"
weight_decay = 0.0001
num_samples = 5
lr = 0.005
nodes = [10,100,10]
encoder = Encoder([DenseLayers(input_size=input_dim,
                               architecture=nodes[:-1],
                               output_size=nodes[-1], activation=activation,last_activation=activation
                               )])

#specify decoder-mu
decoder_mu = infer_decoder(encoder,activation=activation,last_activation=last_activation) #symmetrical to encoder

#combine them into autoencoder
autoencoder = Autoencoder(encoder, decoder_mu)

#convert into BAE-Ensemble
bae_ensemble = BAE_Ensemble(autoencoder=autoencoder,
                            anchored=False,
                            weight_decay=weight_decay,
                            num_samples=num_samples, likelihood=likelihood, learning_rate=lr)


# train mu network
train_loader = convert_dataloader(scaler.transform(X), batch_size=100, shuffle=True)
# run_auto_lr_range_v2(train_loader, bae_ensemble, run_full=True, window_size=2, num_epochs=10)

# bae_ensemble.fit(X,num_epochs=100)
# bae_ensemble = bae_fit(bae_ensemble, X, scaler, num_epoch=1)
# y_nll_samples_mean, y_nll_samples_std = bae_predict(bae_ensemble, np.expand_dims(evaluate_range,1), scaler)

fig, ax = plt.subplots(1,1)
line, = ax.plot([], [], lw=2)
scat = ax.scatter([], [])
between = ax.fill_between(evaluate_range,evaluate_range,evaluate_range)
n_epoch_per_cycle = 5
x_scaled = scaler.transform(X)
reset_param = False

def init():
    line.set_data([], [])
    scat.set_offsets([])
    return line, scat,

def animate(i):
    # global bae_ensemble
    ax.clear()
    if i >1:
        bae_ensemble.fit(train_loader,num_epochs=n_epoch_per_cycle)
    elif i ==0 and reset_param:
        bae_ensemble.reset_parameters()
    y_nll_samples_mean, y_nll_samples_std = bae_predict(bae_ensemble, np.expand_dims(evaluate_range, 1), scaler)
    evaluate_range_scaled = scaler.transform(np.expand_dims(evaluate_range, 1)).reshape(-1)

    line, = ax.plot(evaluate_range_scaled, y_nll_samples_mean, color="tab:blue")
    scat = ax.scatter(x_scaled,x_scaled*0, color="tab:orange")
    between = ax.fill_between(evaluate_range_scaled, y_nll_samples_mean - 3 * y_nll_samples_std,
                                 y_nll_samples_mean + 3 * y_nll_samples_std, alpha=0.3, color="tab:blue")
    # ax.set_ylim(-0.04, 0.005)
    ax.set_ylim(-0.005, 0.04)
    ax.set_title("EPOCH: "+str(i*n_epoch_per_cycle))
    ax.set_ylabel("Reconstruction loss")
    return line, scat, between

# matplotlib.use("Agg")
anim = FuncAnimation(fig, animate, frames=50, interval=10, blit=False)

# anim.save('bae_nll_1d_'+str(last_activation)+"_"+str(nodes)+'.gif', writer='imagemagick')

# plt.show()

#=============reconstructed space=========
fig, ax_rc = plt.subplots(1,1)
# line, = ax_rc.plot([], [], lw=2)
# scat = ax_rc.scatter([], [])
# between = ax_rc.fill_between(evaluate_range,evaluate_range,evaluate_range)

def animate_rc(i):
    # global bae_ensemble
    ax_rc.clear()
    if i >1:
        bae_ensemble.fit(train_loader,num_epochs=n_epoch_per_cycle)
    elif i == 0 and reset_param:
        bae_ensemble.reset_parameters()



    y_rcon_mean, y_rcon_std = bae_predict(bae_ensemble, np.expand_dims(evaluate_range, 1), scaler, key="y_mu")

    line_identity, = ax_rc.plot(x_scaled, x_scaled)
    evaluate_range_scaled = scaler.transform(np.expand_dims(evaluate_range, 1)).reshape(-1)
    line_recon, = ax_rc.plot(evaluate_range_scaled, y_rcon_mean)
    between = ax_rc.fill_between(evaluate_range_scaled, y_rcon_mean - 2 * y_rcon_std,
                                 y_rcon_mean + 2 * y_rcon_std, alpha=0.3, color="tab:orange")

    training_pts = ax_rc.scatter(x_scaled, x_scaled)

    ax_rc.legend([line_identity, line_recon, training_pts], ["Identity", "Reconstructed", "Training points"])

    ax_rc.set_title("EPOCH: "+str(i*n_epoch_per_cycle))
    ax_rc.set_xlim(-0.1,1.1)
    ax_rc.set_ylim(-0.1,1.1)

    return line_identity, line_recon, training_pts, between

anim_rc = FuncAnimation(fig, animate_rc, frames=50, interval=10, blit=False)
# anim.save('bae_recon_1d_'+str(last_activation)+"_"+str(nodes)+'.gif', writer='imagemagick')
# plt.show()

# animate_rc(0)

#
# plt.figure()
# x_scaled = scaler.transform(X)
# y_rcon_mean, y_rcon_std = bae_predict(bae_ensemble, np.expand_dims(evaluate_range, 1), scaler, key="y_mu")
#
# line_identity, = plt.plot(x_scaled, x_scaled)
# line_recon, = plt.plot(scaler.transform(np.expand_dims(evaluate_range, 1)), y_rcon_mean)
#
# training_pts = plt.scatter(x_scaled, x_scaled)
#
# plt.legend([line_identity,line_recon, training_pts],["Identity","Reconstructed","Training points"])
#





