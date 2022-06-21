import numpy as np
from scipy.stats import binom
from baetorch.baetorch.evaluation import convert_hard_pred
from baetorch.baetorch.models_v2.outlier_proba import Outlier_CDF


def calc_exceed(len_train, test_prob, test_hard_pred, contamination=0):
    n = len_train
    n_anom = np.int(n * contamination)  # expected anomalies
    # contamniation = 0
    if contamination == 0:
        ex_conf = np.power(test_prob, n)
    else:
        conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
        ex_conf = conf_func(test_prob)

    np.place(ex_conf, test_hard_pred == 0, 1 - ex_conf[test_hard_pred == 0])

    ex_unc = 1 - ex_conf
    # return np.round(ex_unc, 5)
    # print("EXCEED-MEAN:" + str(np.mean(ex_unc)))
    return ex_unc


def calc_exceed_v2(
    train_scores, test_scores, dist="norm", norm_scaling=True, contamination=0
):
    # outlier cdf
    out_cdf = Outlier_CDF(dist_type=dist, norm_scaling=norm_scaling).fit(train_scores)
    test_prob = out_cdf.predict(test_scores)

    test_hard_pred = convert_hard_pred(test_prob, p_threshold=0.5)

    n = len(train_scores)
    n_anom = np.int(n * contamination)  # expected anomalies

    conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
    exWise_conf = conf_func(test_prob)
    np.place(exWise_conf, test_hard_pred == 0, 1 - exWise_conf[test_hard_pred == 0])

    return 1 - exWise_conf, test_prob, test_hard_pred
