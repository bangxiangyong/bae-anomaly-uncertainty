import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns

def calc_ood_threshold(training_scores= np.array([2.5,3,3.1,10]),perc_threshold=95):
    # given N training OOD scores, determine threshold based on percentile of scores
    training_scores
    score_threshold = stats.scoreatpercentile(training_scores,perc_threshold)

    return score_threshold

def convert_hard_predictions(test_scores : np.ndarray, ood_threshold : float):
    # given OOD threshold, convert test raw scores to {0,1}
    hard_pred = np.zeros(len(test_scores)).astype(int)
    hard_pred[np.argwhere(test_scores >= ood_threshold)[:,0]] = 1

    return hard_pred

def calc_unc_(hard_pred):
    # convert ensembled of hard predictions to uncertainty score
    N = len(hard_pred)
    a = (np.sum(hard_pred)) + 1
    b = N - a + 2

    posterior_mean = a / (a + b)
    posterior_var = posterior_mean * (1 - posterior_mean)
    return posterior_var

def calc_unc_scaled(hard_pred, unc_method=calc_unc_):
    # Scaled to [0,1]. Convert ensembled of hard predictions to uncertainty score
    min_inp = np.ones_like(hard_pred)
    max_inp = np.copy(min_inp)
    max_inp[:(int(len(hard_pred) / 2))] *= 0

    unc_min = unc_method(min_inp)
    unc_max = unc_method(max_inp)

    unc_scaled = (unc_method(hard_pred) - unc_min) / (unc_max - unc_min)

    return unc_scaled


