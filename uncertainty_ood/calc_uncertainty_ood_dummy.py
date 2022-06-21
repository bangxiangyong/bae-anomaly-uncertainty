import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns

def entropy(a):
    b = 1-a
    return -(a*np.log2(a))-(b*np.log2(b))

batch_entropy = np.vectorize(lambda x: entropy(x))


# given N training OOD scores, determine threshold based on percentile of scores
training_scores = np.array([4,3,2,1,3,1,2,10,12])
perc_threshold = 95
score_threshold = stats.scoreatpercentile(training_scores,perc_threshold)

# convert test predictions to {0,1}
test_scores = np.array([0.001,0,1,2,19,1000,80,900])
test_scores = np.array([0.001,0,1,2,19,1000])
test_scores = np.array([0.001,0,1,2,1,2,1,2,1,2])
hard_pred = np.zeros(len(test_scores)).astype(int)
hard_pred[np.argwhere(test_scores >= score_threshold)[:,0]] = 1



# hard_pred = np.array([0,0,0,0,0,0,1,1,1,1,1,1])
# hard_pred = np.array([0,0,0,0,0,0,0,0,0,0,1,1])
# hard_pred = np.array([0,0,0,1,1,1,1,1,1,1,1,1])
# hard_pred = np.array([0,0,0,0,0,0,0,0])
# hard_pred = np.array([0,0,0,0,1,1])
hard_pred = np.array([0,1,1,1,1,1])
# hard_pred = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1])
# hard_pred = np.array([1,1,1,1,1,1])

# beta posterior
N = len(hard_pred)
a = (np.sum(hard_pred))+1
b = N-a+2


posterior_dist = stats.beta(a,b)
posterior_mean = a/(a+b)
posterior_var = (a*b)/(((a+b)**2)*(a+b+1))
posterior_std = np.sqrt(posterior_var)


# posterior_var_a = a*b
# posterior_var_b = np.power(a+b,2)


x_Axis = np.linspace(0,1,100)
posterior_pdf = posterior_dist.pdf(x_Axis)
u_bound = stats.beta.ppf(0.975, a, b)
l_bound = stats.beta.ppf(0.025, a, b)

plt.figure()
plt.plot(x_Axis, posterior_pdf)

plt.vlines(u_bound, np.max(posterior_pdf), np.min(posterior_pdf))
plt.vlines(l_bound, np.max(posterior_pdf), np.min(posterior_pdf))
plt.vlines(posterior_mean, np.max(posterior_pdf), np.min(posterior_pdf))

print(hard_pred)
print(u_bound)
print(posterior_mean)
print(l_bound)
print("-----")
print(posterior_std)
print(np.std(hard_pred))
print(u_bound-l_bound)
print(np.sqrt(posterior_mean*(1-posterior_mean)))

print("SKEWNESS")
print(posterior_dist.moment(3))

print("KURTOSIS")
print(posterior_dist.moment(4))

print("---------")
print(u_bound-0.5)
print(posterior_mean-0.5)
# print(np.abs(l_bound-0.5))
# print(np.abs(u_bound-0.5)-np.abs(l_bound-0.5))

print((u_bound-0.5)+(l_bound-0.5))
print(entropy(posterior_mean))
print(1-(posterior_mean))

# 1 = Don't know (not confident), 0 = Know (confident)

# convert ensembled of hard predictions to uncertainty score
def calc_unc_1(hard_pred):
    N = len(hard_pred)
    a = (np.sum(hard_pred)) + 1
    b = N - a + 2

    posterior_mean = a / (a + b)
    return posterior_mean*(1-posterior_mean)

def calc_unc_2(hard_pred):
    N = len(hard_pred)
    a = (np.sum(hard_pred)) + 1
    b = N - a + 2

    posterior_var = (a * b) / (((a + b) ** 2) * (a + b + 1))
    posterior_std = np.sqrt(posterior_var)
    return posterior_std


def calc_unc_3(hard_pred):
    N = len(hard_pred)
    a = (np.sum(hard_pred)) + 1
    b = N - a + 2

    posterior_var = (a * b) / (((a + b) ** 2) * (a + b + 1))
    return posterior_var

def calc_unc_4(hard_pred):
    empiric_var = np.var(hard_pred)
    return empiric_var

def calc_unc_5(hard_pred):
    empiric_mean = np.mean(hard_pred)
    empiric_var = empiric_mean*(1-empiric_mean)
    return empiric_var

def calc_unc_6(hard_pred):
    N = len(hard_pred)
    a = (np.sum(hard_pred)) + 1
    b = N - a + 2

    u_bound = stats.beta.ppf(0.975, a, b)
    l_bound = stats.beta.ppf(0.025, a, b)

    return u_bound-l_bound


hard_pred = np.array([0,0,0,0,0,1,1,1,1,1])
unc_raw = calc_unc_1(hard_pred)

min_inp = np.ones_like(hard_pred)
max_inp = np.copy(min_inp)
max_inp[:(int(len(hard_pred)/2))] *= 0

unc_min = calc_unc_1(min_inp)
unc_max = calc_unc_1(max_inp)

unc_scaled = (calc_unc_1(hard_pred) - unc_min)/(unc_max-unc_min)

print(unc_scaled)

def calc_unc_scaled(hard_pred, unc_method=calc_unc_1):
    min_inp = np.ones_like(hard_pred)
    max_inp = np.copy(min_inp)
    max_inp[:(int(len(hard_pred) / 2))] *= 0

    unc_min = unc_method(min_inp)
    unc_max = unc_method(max_inp)

    unc_scaled = (unc_method(hard_pred) - unc_min) / (unc_max - unc_min)

    return unc_scaled

def create_dummy_binary(n_zeros, N):
    max_inp = np.ones(N).astype(int)
    max_inp[:n_zeros] *= 0
    return max_inp

def prob_entropy(hard_pred):
    prob = np.mean(hard_pred)
    entr_ = entropy(prob)

    return entr_

N = 30
dummy_binaries = np.array([create_dummy_binary(n_zeros=i,N=N) for i in np.arange(N+1)])

unc_scaled = [calc_unc_scaled(dummy, unc_method = calc_unc_1) for dummy in dummy_binaries]
unc_scaled2 = [calc_unc_scaled(dummy, unc_method = calc_unc_2) for dummy in dummy_binaries]
unc_scaled3 = [calc_unc_scaled(dummy, unc_method = calc_unc_3) for dummy in dummy_binaries]
unc_scaled4 = [calc_unc_scaled(dummy, unc_method = calc_unc_4) for dummy in dummy_binaries]
unc_scaled5 = [calc_unc_scaled(dummy, unc_method = calc_unc_5) for dummy in dummy_binaries]
unc_scaled6 = [calc_unc_scaled(dummy, unc_method = calc_unc_6) for dummy in dummy_binaries]

# unc_entropy = [prob_entropy(dummy) for dummy in dummy_binaries]

plt.figure()
plt.plot(np.arange(N+1), unc_scaled)
plt.plot(np.arange(N+1), unc_scaled2)
plt.plot(np.arange(N+1), unc_scaled3)
plt.plot(np.arange(N+1), unc_scaled4)
plt.plot(np.arange(N+1), unc_scaled5)
plt.plot(np.arange(N+1), unc_scaled6)

# plt.plot(np.arange(N+1), unc_entropy)


# print(posterior_dist.entropy())
# print(stats.entropy([posterior_mean, 1-posterior_mean], base=2))

# plt.figure()
# plt.plot(x_Axis, np.sqrt(x_Axis*(1-x_Axis))*2)
#
#
# plt.plot(x_Axis, batch_entropy(x_Axis))
#


# batch_entropy

# plt.figure()
# plt.plot(x_Axis, np.sqrt(x_Axis*(1-x_Axis)))


# print(u_bound-0.5)
# print(l_bound-0.5)


# stats.beta.ppf(95, a, b)

# ecdf = ECDF(ood_scores)
#
# print(score_threshold)
# print(ecdf(ood_scores))
# print(ecdf(score_threshold))
#
# print(np.median(ood_scores))
#
# # ECDF plot
# batch_percentileofscore = np.vectorize(lambda x:
#                                        stats.percentileofscore(ood_scores, x,
#                                                                kind="mean"))
# batch_np_percentile= np.vectorize(lambda x:
#                                        np.percentile(ood_scores, x,
#                                                                kind="mean"))
#
# x_range = np.linspace(np.min(ood_scores), np.max(ood_scores), 100)
#
# # plot of percentiles
# plt.figure()
# plt.plot(x_range, batch_percentileofscore(x_range)/100)
# sns.ecdfplot(data=ood_scores)
#