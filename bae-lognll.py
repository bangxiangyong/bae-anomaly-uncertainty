# import required libraries
from scipy.stats import norm, spearmanr
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import random
from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import DenseLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from kde_example import kde

rnd_seed = 150
torch.manual_seed(rnd_seed)
np.random.seed(rnd_seed)

def torch_log_gaussian_loss(x,fx,std=1.):
    return 0.5*((x-fx)/std)**2
    # return 0.5*((x-fx)/std)**2-torch.log(1/torch.tensor(std))-torch.log(1/(torch.sqrt(torch.tensor(2*3.142))))

def np_log_gaussian_loss(x,fx,std=1):
    return 0.5*((x-fx)/std)**2-np.log(1/std)-np.log(1/(np.sqrt(2*3.142)))

def np_log_gaussian_loss(x,fx,std=1):
    return ((x-fx)/std)**2

# class Autoencoder(torch.nn.Module):
#     def __init__(self, n_inputs=2, n_hidden=2):
#         super(Autoencoder, self).__init__()
#         self.encoder = torch.nn.Linear(n_inputs, n_hidden,bias=True)
#         self.hidden = torch.nn.Linear(n_hidden, n_hidden,bias=True)
#         self.decoder = torch.nn.Linear(n_hidden, n_inputs,bias=True)
#
#     def forward(self, x):
#         # return torch.tanh(self.decoder(self.encoder(x)))
#         # return torch.sigmoid(self.decoder(self.encoder(x)))
#         # return self.decoder(self.encoder(x))
#         # return self.decoder(torch.sigmoid(self.encoder(x)))
#         # return self.decoder(torch.relu(self.encoder(x)))
#         # return self.decoder(torch.tanh(self.encoder(x)))
#         # return torch.sigmoid(self.decoder(self.hidden(torch.relu(self.encoder(x)))))
#         # return torch.sigmoid(self.decoder(torch.relu(self.encoder(x))))
#         # return torch.tanh(self.decoder(torch.tanh(self.encoder(x))))
#         # return self.decoder(self.hidden((self.encoder(x))))
#         return torch.sigmoid(self.decoder(self.hidden(torch.relu(self.encoder(x)))))
#         # return (self.decoder(self.encoder(x)))
#
# def plot_points(X):
#        plt.plot(X[:, 0], X[:, 1], 'x')
#        plt.axis('equal')
#        plt.show()
#        plt.grid(True)
#
#


# scaler = StandardScaler()
# scaler = MinMaxScaler()

multi_modal = True
heteroscedestic = False

n_samples= 250
X1 = np.random.normal(loc=0.2,scale=0.015,size=n_samples)

X3 = np.random.normal(loc=0.8,scale=0.08,size=n_samples)

if multi_modal:
    X = np.expand_dims(np.concatenate((X1,X3)),1)
else:
    X = np.expand_dims(X1,1)

# plot True Gaussian
evaluate_range = np.linspace(-0.05, 1.05, 100)
pdf_X1 = norm.pdf(evaluate_range, loc=0.2, scale=0.015)
pdf_X3 = norm.pdf(evaluate_range, loc=0.8, scale=0.08)

# KDE estimated
kde_res = np.array([kde(x, X, bw_h=0.01) for x in evaluate_range ])


# Visualizing the distribution
if multi_modal:
    true_pdf = pdf_X1+pdf_X3
else:
    true_pdf = pdf_X1

color = "tab:blue"
plt.plot(evaluate_range,true_pdf, color=color)
plt.plot(evaluate_range,kde_res, color="tab:orange")

plt.xlabel('Heights')
plt.ylabel('Probability Density')

# USE BAE
input_dim = 1
encoder = Encoder([DenseLayers(input_size=input_dim,
                               architecture=[50],
                               output_size=25, activation="sigmoid",last_activation="sigmoid"
                               )])

#specify decoder-mu
decoder_mu = infer_decoder(encoder,activation="sigmoid",last_activation="sigmoid") #symmetrical to encoder
decoder_sig = infer_decoder(encoder,activation="relu",last_activation="none") #symmetrical to encoder

#combine them into autoencoder
if heteroscedestic:
    autoencoder = Autoencoder(encoder, decoder_mu,decoder_sig=decoder_sig)
else:
    autoencoder = Autoencoder(encoder, decoder_mu)

#convert into BAE-Ensemble
bae_ensemble = BAE_Ensemble(autoencoder=autoencoder,
                            anchored=True,
                            weight_decay=0.0001,
                            num_samples=10, likelihood="bernoulli")


# train mu network
train_loader = convert_dataloader(X, batch_size=450, shuffle=True)
run_auto_lr_range_v2(train_loader, bae_ensemble, run_full=True, window_size=2, num_epochs=10)
# bae_ensemble.fit(X,num_epochs=450)
bae_ensemble.fit(train_loader,num_epochs=500)

if heteroscedestic:
    bae_ensemble.fit(X,num_epochs=100, mode="sigma", sigma_train="joint")

if heteroscedestic:
    bae_ensemble.fit(X,num_epochs=100, mode="sigma", sigma_train="joint")

# predict samples
if heteroscedestic:
    y_nll_samples = bae_ensemble.predict_samples(evaluate_range,select_keys=["nll_sigma"])
else:
    y_nll_samples = bae_ensemble.predict_samples(evaluate_range, select_keys=["nll_homo"])

# y_nll_samples = -y_nll_samples
y_nll_samples = np.exp(-y_nll_samples)
# true_pdf = np.log(true_pdf)

# y_nll_samples = np.exp(-y_nll_samples)
# true_pdf = true_pdf

# calculate mean
y_nll_samples_mean = y_nll_samples.mean(0)[0]
y_nll_samples_std = y_nll_samples.std(0)[0]

# y_nll_lb = np.percentile(y_nll_samples[:,0],5,axis=0)
# y_nll_ub = np.percentile(y_nll_samples[:,0],95,axis=0)
y_nll_samples_mean = np.percentile(y_nll_samples[:,0],50,axis=0)

y_nll_lb = y_nll_samples_mean-2*y_nll_samples_std
y_nll_ub = y_nll_samples_mean+2*y_nll_samples_std

sp_corr = spearmanr(true_pdf, y_nll_samples_mean)[0]

fig, (ax1,ax2,ax3) = plt.subplots(3,1)
ax1.plot(evaluate_range, y_nll_samples_mean)
ax1.fill_between(evaluate_range, y_nll_ub, y_nll_lb, alpha=0.5)

# ax1.plot(evaluate_range,true_pdf, color="tab:orange")
ax1.scatter(X,X*0+(y_nll_samples_mean.min()), c="blue")

ax2.plot(evaluate_range, y_nll_samples_std)
ax2.scatter(X,X*0, c="blue")

# ax3.scatter(true_pdf, y_nll_samples_mean, c="blue")
# ax3.set_title('SPMAN : {:.2f}'.format(sp_corr))
# ax3.set_xlabel("TRUE PDF")
# ax3.set_ylabel("PRED. PDF")

ax1.set_title("AE PDF")
ax2.set_title("UNCERTAINTY")
ax3.set_title("TRUE PDF")

print('SPMAN : {:.2f}'.format(sp_corr))
plt.tight_layout()

# ax3.plot(evaluate_range, np.log(true_pdf))
ax3.plot(evaluate_range, (true_pdf))
# ax4.scatter(y_nll_samples_mean,true_pdf)