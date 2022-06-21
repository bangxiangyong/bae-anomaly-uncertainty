import matplotlib.pyplot as plt
import numpy as np
from pyod.utils import generate_data
from scipy.stats import binom
from statsmodels.distributions import ECDF

from baetorch.baetorch.util.seed import bae_set_seed

bae_set_seed(123)

def apply_unc_thresholds(test_point, unc_bounds=(90,100), resolution=0.1):
    unc_thresholds = np.arange(unc_bounds[0], unc_bounds[1], resolution)
    res = []
    for threshold in unc_thresholds:
        res_ = 1 if test_point>=threshold else 0
        res.append(res_)
    return np.array(res)

unc_bounds = [94,96]
resolution = 0.1
# eval_points = np.linspace(0,100,200)
# testp_results = np.array([apply_unc_thresholds(i, unc_bounds=unc_bounds, resolution=resolution)
#          for i in eval_points])
#
# testp_mean = testp_results.mean(-1)
# testp_var = 1-(testp_results.var(-1)*4)

def get_mean_var_testp(unc_bounds=(90,100),
                       resolution=0.1,
                       eval_points = np.linspace(0,100,100)):

    testp_results = np.array([apply_unc_thresholds(i, unc_bounds=unc_bounds, resolution=resolution)
                              for i in eval_points])

    testp_mean = testp_results.mean(-1)
    testp_var = 1 - (testp_results.var(-1) * 4)

    return eval_points, testp_mean, testp_var

eval_points, testp_mean, testp_var = get_mean_var_testp(unc_bounds=unc_bounds,
                                                        resolution=0.1)

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(eval_points, testp_mean)
ax2.plot(eval_points, testp_var)


#=======================DEVELOP SIGMOID/LINEAR MODEL ========================

def convert_prob(x, threshold_lb=90, threshold_ub=100):
    m = 1/(threshold_ub-threshold_lb)
    c = -threshold_lb/(threshold_ub-threshold_lb)

    prob_y= np.clip(m*x+c,0,1)
    unc_y = prob_y*(1-prob_y)*4

    return prob_y, unc_y

x = np.linspace(0,100,250)
prob_y, unc_y = convert_prob(x=x, threshold_lb=75, threshold_ub=100)

plt.figure()
plt.plot(x,prob_y)
plt.plot(x,unc_y)


# #===============EXCEED================
# from pyod.models.knn import KNN
#
# contamination = 0.1  # percentage of outliers
# n_train = 200  # number of training points
# n_test = 100  # number of testing points
#
# X_train, y_train, X_test, y_test = generate_data(
#     n_train=n_train, n_test=n_test, contamination=contamination)
#
# # Train an anomaly detector (for instance, here we use kNNO)
# detector = KNN().fit(X_train)
#
# # Compute the anomaly scores in the training set
# train_scores_knno = detector.decision_function(X_train)
#
# # Compute the anomaly scores in the test set
# test_scores_knno = detector.decision_function(X_test)
#
# # Predict the class of each test example
# prediction_knno = detector.predict(X_test)
#
# # Estimate the confidence in class predictions with ExCeeD
# # knno_confidence = ExCeeD(train_scores_knno, test_scores_knno, prediction_knno, contamination)
#
# train_scores = train_scores_knno
# test_scores = test_scores_knno
# prediction = prediction_knno
#
# n = len(train_scores)
# n = 1500
# n_anom = np.int(n * contamination)  # expected anomalies
#
# count_instances = np.vectorize(lambda x: np.count_nonzero(train_scores <= x))
# n_instances = count_instances(test_scores)
#
# prob_func = np.vectorize(lambda x: (1 + x) / (2 + n))
# posterior_prob = prob_func(n_instances)  # Outlier probability according to ExCeeD
#
# conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
# exWise_conf = conf_func(posterior_prob)
# ans = np.place(exWise_conf, prediction == 0, 1 - exWise_conf[prediction == 0])  # if the example is classified as normal,
#
# ##==============
# prob = np.linspace(0,1,100)
# conf_ = np.array([1 - binom.cdf(n - n_anom, n, p) for p in prob])
#
# # use 1 - confidence.
# plt.figure()
# plt.plot(prob,conf_)
# plt.plot(prob,1-conf_)
#
# plt.figure()
# plt.hist(posterior_prob, density=True)
#









