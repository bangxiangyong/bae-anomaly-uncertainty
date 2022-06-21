#

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import torch
from baetorch.baetorch.util.truncated_gaussian import TruncatedNormal

x = np.linspace(0, 1, 100)
mu = np.linspace(0, 1, 100)

# clip extreme values
min_ = 1e-11
max_ = 1 - min_

x = np.clip(x, min_, max_)
mu = np.clip(x, min_, max_)

b_ll = x * np.log(mu) + (1 - x) * np.log(1 - mu)


def norm_const_cb(mu):
    # mu_temp = 1-2*mu
    return np.piecewise(
        mu, [mu == 0.5, mu != 0.5], [2, 2 * np.arctanh(1 - 2 * mu) / (1 - 2 * mu)]
    )


cb_ll = b_ll + np.log(norm_const_cb(mu))


# Gaussian ll
sigma = 1.0
gauss_ll = -0.5 * (((x - mu) / sigma) ** 2) - np.log(sigma) - np.log(np.sqrt(2 * np.pi))

# trunc gauss
logz = np.log(
    0.5
    * (erf((1 - mu) / (np.sqrt(2) * sigma)) - 0.5 * erf((-mu) / (np.sqrt(2) * sigma)))
)
tgauss_ll = gauss_ll - logz


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

ax1.plot(x, b_ll)
ax2.plot(x, cb_ll)
ax3.plot(x, gauss_ll)
ax4.plot(x, tgauss_ll)

tcgaus_ll = []
for i, x_i in enumerate(x):
    x_ = TruncatedNormal(
        loc=torch.Tensor([mu[i]]), scale=torch.Tensor([1.0]), a=0.0, b=1.0
    ).log_prob(torch.Tensor([x_i]))
    tcgaus_ll.append(x_.item())

ax4.plot(x, tcgaus_ll)


# ===========
from scipy.stats import truncnorm, beta
import matplotlib.pyplot as plt


a, b = 0.0, 1.0
scale = 0.5
fig, ax = plt.subplots(1, 1)
# x = np.linspace(truncnorm.ppf(0.01, a, b), truncnorm.ppf(0.99, a, b), 100)
tgaus_ll_scipy = [
    np.log(truncnorm.pdf(x[i], a, b, loc=mu[i], scale=scale)) for i in range(len(x))
]
tgaus_ll_pytorch = [
    TruncatedNormal(loc=torch.Tensor([mu[i]]), scale=torch.Tensor([scale]), a=a, b=b)
    .log_prob(torch.Tensor([x[i]]))
    .item()
    for i in range(len(x))
]
# ax.plot(x, tgaus_ll_scipy)
# ax.plot(x, tgaus_ll_pytorch)
# ax.plot(x, truncnorm.pdf(x, a, b), "r-", lw=5, alpha=0.6, label="truncnorm pdf")


def log_scipy_tgauss(x, a, b, loc, scale):
    new_a, new_b = (a - loc) / scale, (b - loc) / scale
    # new_x = np.log(truncnorm.pdf(x, new_a, new_b, loc=loc, scale=scale))
    new_x = truncnorm.logpdf(x, new_a, new_b, loc=loc, scale=scale)
    # new_x = truncnorm.pdf(x, new_a, new_b, loc=loc, scale=scale)
    return new_x


tgaus_ll_scipy = [
    log_scipy_tgauss(x[i], a, b, loc=mu[i], scale=1.0) for i in range(len(x))
]
ax.plot(x, tgaus_ll_scipy)


ax4.plot(x, tgaus_ll_scipy)


# ==========GLOBAL MINIMUM======
def beta_logpdf(x, a, b):
    return beta(a, b).logpdf(x)


def logit_gauss(x, mu):
    const = -np.log(np.sqrt(2 * np.pi))
    body = -((np.log(x / (1 - x)) - mu) ** 2) - np.log(x * (1 - x))
    return body + const


def u_quad(x, mu):
    const = np.log(12)
    body = -3 * np.log(1 - mu) + 2 * np.log(x - 0.5 + 0.5 * mu)
    return body + const


tgaus_ll_scipy = [
    log_scipy_tgauss(x[i], a, b, loc=mu[i], scale=1.0) for i in range(len(x))
]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

all_tgaus = []
for mu_i in mu:
    # tgaus_ll_scipy_temp = [
    #     log_scipy_tgauss(x[i], a, b, loc=mu_i, scale=1.0) for i in range(len(x))
    # ]
    # tgaus_ll_scipy_temp = [
    #     -0.5 * (((x[i] - mu_i) / sigma) ** 2)
    #     - np.log(sigma)
    #     - np.log(np.sqrt(2 * np.pi))
    #     for i in range(len(x))
    # ]
    # tgaus_ll_scipy_temp = [
    #     log_scipy_tgauss(x[i], a, b, loc=0.5, scale=mu_i) for i in range(len(x))
    # ]
    # tgaus_ll_scipy_temp = [logit_gauss(x=x[i], mu=mu_i) for i in range(len(x))]
    tgaus_ll_scipy_temp = [beta_logpdf(x=x[i], a=mu_i, b=mu_i) for i in range(len(x))]
    # tgaus_ll_scipy_temp = [u_quad(x=x[i], mu=mu_i) for i in range(len(x))]

    ax1.plot(x, tgaus_ll_scipy_temp)
    all_tgaus.append(tgaus_ll_scipy_temp)
all_tgaus = np.array(all_tgaus)
ax2.plot(x, all_tgaus.max(axis=0))
# ax2.plot(x, tgaus_ll_scipy)

# ====================================
def beta_logpdf(x, a, b):
    return beta(a, b).logpdf(x)


all_a = np.linspace(0.0001, 1, 10)
all_b = np.linspace(0.0001, 1, 10)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

all_beta = []
for a_i in all_a:
    for b_i in all_b:
        # beta_i = [beta_logpdf(x=x[i], a=a_i, b=b_i) for i in range(len(x))]
        # beta_i = [
        #     log_scipy_tgauss(x[i], a=0, b=1, loc=a_i, scale=b_i) for i in range(len(x))
        # ]
        beta_i = [
            log_scipy_tgauss(x[i], a=0, b=1, loc=a_i, scale=b_i) for i in range(len(x))
        ]
        all_beta.append(beta_i)
        ax1.plot(x, beta_i)
all_beta = np.array(all_beta)
ax2.plot(x, all_beta.max(axis=0))
# ax2.plot(x, tgaus_ll_scipy)

#======================================

def calc_gauss_ll(x,mu,sigma):
    gauss_ll = -0.5 * (((x - mu) / sigma) ** 2) - np.log(sigma) - np.log(np.sqrt(2 * np.pi))
    return gauss_ll

def calc_bern_ll(x, mu):
    return x * np.log(mu) + (1 - x) * np.log(1 - mu)

def transform(x,min,max):
    return (x-min)/(max-min)

def inv_transform(x, min=-100, max=100):
    return (x * (max - min))+min

def inv_transform_var(x, min=-100, max=100):
    return (x * (max - min))

ori_ll = calc_gauss_ll(x,mu=0.5,sigma=0.1)

new_ll = calc_gauss_ll(inv_transform(x),mu=inv_transform(0.5), sigma=inv_transform_var(0.1))

ori_argsort = np.argsort(ori_ll)
new_argsort = np.argsort(new_ll)



ori_bern_ll = calc_bern_ll(x,mu)
new_bern_ll = calc_bern_ll(inv_transform(x),inv_transform(mu))

#======================================

# def (x,mu,sigma):
#     gauss_ll = -0.5 * (((x - mu) / sigma) ** 2) - np.log(sigma) - np.log(np.sqrt(2 * np.pi))
#     return gauss_ll
