

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import random
from sklearn.preprocessing import MinMaxScaler

# from kde_example import kde, kde_mse
from kde_example import kde

rnd_seed = 151215
torch.manual_seed(rnd_seed)
np.random.seed(rnd_seed)

def torch_log_gaussian_loss(x,fx,std=1.):
    return 0.5*((x-fx)/std)**2
    # return 0.5*((x-fx)/std)**2-torch.log(1/torch.tensor(std))-torch.log(1/(torch.sqrt(torch.tensor(2*3.142))))

def np_log_gaussian_loss(x,fx,std=1):
    return 0.5*((x-fx)/std)**2-np.log(1/std)-np.log(1/(np.sqrt(2*3.142)))

def np_log_gaussian_loss(x,fx,std=1):
    return ((x-fx)/std)**2

class Autoencoder(torch.nn.Module):
    def __init__(self, n_inputs=2, n_hidden=2):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Linear(n_inputs, n_hidden,bias=True)
        self.hidden = torch.nn.Linear(n_hidden, n_hidden,bias=True)
        self.decoder = torch.nn.Linear(n_hidden, n_inputs,bias=True)

    def forward(self, x):
        # return torch.tanh(self.decoder(self.encoder(x)))
        # return torch.sigmoid(self.decoder(self.encoder(x)))
        # return self.decoder(self.encoder(x))
        # return self.decoder(torch.sigmoid(self.encoder(x)))
        # return self.decoder(torch.relu(self.encoder(x)))
        # return self.decoder(torch.tanh(self.encoder(x)))
        # return torch.sigmoid(self.decoder(self.hidden(torch.relu(self.encoder(x)))))
        # return torch.sigmoid(self.decoder(torch.relu(self.encoder(x))))
        # return torch.tanh(self.decoder(torch.tanh(self.encoder(x))))
        # return self.decoder(self.hidden((self.encoder(x))))
        return torch.sigmoid(self.decoder(self.hidden(torch.relu(self.encoder(x)))))
        # return (self.decoder(self.encoder(x)))

def plot_points(X):
       plt.plot(X[:, 0], X[:, 1], 'x')
       plt.axis('equal')
       plt.show()
       plt.grid(True)




# scaler = StandardScaler()
scaler = MinMaxScaler()
# X = np.expand_dims(np.random.randn(1500),1)
n_samples= 500
X1 = np.random.normal(loc=-3.0,scale=0.15,size=n_samples)
X2 = np.random.normal(loc=0.0,scale=0.05,size=n_samples)
X3 = np.random.normal(loc=3.0,scale=0.25,size=n_samples)
X4 = np.random.normal(loc=10.0,scale=0.1,size=n_samples)
# X = np.expand_dims(np.concatenate((X1,X2,X3,X4)),1)
X = np.expand_dims(np.concatenate((X1,X2,X3)),1)
# X = np.expand_dims(np.concatenate((X1,X3)),1)
# X = np.expand_dims(X1,1)

X_scaled = scaler.fit_transform(X)
X_tensor = torch.from_numpy(X_scaled).float()

n_epochs = 250
lr=0.001
std = 0.01
ae_model = Autoencoder(n_inputs=1, n_hidden=50)
optim = torch.optim.Adam(ae_model.parameters(),lr=lr, weight_decay=0.001)
# optim = torch.optim.SGD(ae_model.parameters(),lr=lr, weight_decay=0.001)

# fit ae
for epoch in range(n_epochs):
    y_pred_space = ae_model(X_tensor)
    # loss = ((y_pred_space - X_tensor) ** 2).mean(-1).mean(-1)*1
    loss = torch_log_gaussian_loss(x=X_tensor,fx=y_pred_space,std=std).mean()


    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)


xlow, xhigh = np.min(X),np.max(X)
span = (xhigh-xlow)/2
xlow =xlow -span
xhigh =xhigh +span

evaluate_space = np.expand_dims(np.linspace(xlow,xhigh,100),1)
evaluate_space_scaled = scaler.transform(evaluate_space)
evaluate_space_tensor = torch.from_numpy(evaluate_space_scaled).float()

y_pred_data = ae_model(X_tensor).detach().numpy()
y_pred_nll = ((y_pred_data - X_scaled) ** 2).mean(-1)
y_pred_space = ae_model(evaluate_space_tensor).detach().numpy()
y_pred_space_nll = ((y_pred_space - evaluate_space_scaled) ** 2).mean(-1)

# nll_threshold = -np.percentile(y_pred_data,0.95)
nll_threshold = np.percentile(-y_pred_nll.flatten(),90)
nll_logthreshold = np.percentile(-np.log(y_pred_nll).flatten(),90)


fig, (ax1,ax2) = plt.subplots(1,2)
ymin = np.min(-(y_pred_space_nll))
ax1.plot(evaluate_space_scaled, -(y_pred_space_nll))
ax1.scatter(X_scaled,X_scaled*0+(ymin-0.01), c="blue")
ax1.scatter(y_pred_data,y_pred_data*0+(ymin-0.0001), c="red")
ax1.hlines(nll_threshold, xlow,xhigh)

ymin = np.min(-np.log(y_pred_space_nll))
ax2.plot(evaluate_space_scaled, -np.log(y_pred_space_nll))
ax2.scatter(X_scaled,X_scaled*0+(ymin-0.01), c="blue")
ax2.scatter(y_pred_data,y_pred_data*0+(ymin-0.0001), c="red")
ax2.hlines(nll_logthreshold, xlow,xhigh)

plt.figure()
plt.scatter(X_scaled,y_pred_data)

plt.figure()
plt.plot(evaluate_space_scaled, y_pred_space)
plt.plot(evaluate_space_scaled, evaluate_space_scaled)

#========================================================

def gaussian_pdf(x, mu,std):
    return (1/(std*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/std)**2)

# def gaussian_logpdf(x, mu,std):
#     return -np.log(1/(std*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/std)**2)


# true_gaussian = gaussian_pdf(evaluate_space_scaled,X_scaled[0],std=1)

# true_dist = np.log(true_dist)
# true_dist = np.array([np_log_gaussian_loss(x=evaluate_space_scaled,fx=x,std=std).mean(-1) for x in X_scaled])
# true_dist = -true_dist.mean(0)

# true_dist = np.array([np_log_gaussian_loss(x=evaluate_space_scaled,fx=x,std=std).mean(-1) for x in X_scaled])
# true_dist = -true_dist.mean(0)
# true_dist = -true_dist

true_dist = np.array([kde(x, X_scaled, bw_h=std/1.) for x in evaluate_space_scaled ])
# true_dist = np.array([kde_mse(x, X_scaled, bw_h=std/1.) for x in evaluate_space_scaled ])
y_pred_dist = np_log_gaussian_loss(x=evaluate_space_scaled,fx=y_pred_space,std=std).mean(-1)


plt.figure()
plt.plot(evaluate_space_scaled,true_dist)
plt.plot(evaluate_space_scaled,np.exp(-y_pred_dist))
# plt.plot(evaluate_space_scaled,np.exp(-y_pred_dist)/np.mean(np.exp(-y_pred_dist))) # normalised
plt.legend(["KDE", "AE"])

# true_dist = np.array([kde(x, X_scaled, bw_h=std/1.) for x in evaluate_space_scaled ])
# y_pred_dist = np_log_gaussian_loss(x=evaluate_space_scaled,fx=y_pred_space,std=std).mean(-1)


# plt.figure()
# plt.plot(evaluate_space_scaled,np.log(true_dist))
# plt.plot(evaluate_space_scaled,y_pred_dist)

# plt.figure()
# plt.plot(evaluate_space_scaled,true_dist)
# plt.plot(evaluate_space_scaled,y_pred_dist)


# plt.figure()
# for td in true_dist:
#     plt.plot(evaluate_space_scaled[:,0],td)
# plt.plot(evaluate_space_scaled,-y_pred_dist)

for param in ae_model.parameters():
    print(param)












