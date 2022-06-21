import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import spearmanr, beta, gamma, lognorm, norm, uniform, expon
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from statsmodels.distributions import ECDF

from uncertainty_ood.calc_uncertainty_ood import (
    calc_ood_threshold,
    convert_hard_predictions,
)
from util.evaluate_ood import plot_kde_ood, plot_histogram_ood
import pandas as pd


def plot_kde_auroc(
    nll_inliers_train_mean, nll_inliers_test_mean, nll_outliers_test_mean, mode="kde"
):
    y_true = np.concatenate(
        (
            np.zeros(nll_inliers_test_mean.shape[0]),
            np.ones(nll_outliers_test_mean.shape[0]),
        )
    )
    y_scores = np.concatenate((nll_inliers_test_mean, nll_outliers_test_mean))

    auroc = roc_auc_score(y_true, y_scores)
    auroc_text = "AUROC: {:.3f}".format(auroc, 2)

    fig, ax = plt.subplots(1, 1)

    if mode == "kde":
        plot_kde_ood(
            nll_inliers_train_mean,
            nll_inliers_test_mean,
            nll_outliers_test_mean,
            fig=fig,
            ax=ax,
        )

    if mode == "hist":
        plot_histogram_ood(
            nll_inliers_train_mean,
            nll_inliers_test_mean,
            nll_outliers_test_mean,
            fig=fig,
            ax=ax,
        )

    ax.set_title(auroc_text)


def bae_pred_all(bae_ensemble, x, return_mean=False):
    y_pred_samples = bae_ensemble.predict_samples(x, select_keys=["se", "y_mu"])
    y_latent_samples = bae_ensemble.predict_latent_samples(x)

    y_nll_mean = y_pred_samples.mean(0)[0]
    y_nll_var = y_pred_samples.var(0)[0]
    y_recon_var = y_pred_samples.var(0)[1]
    y_latent_mean = y_latent_samples.mean(0)
    y_latent_var = y_latent_samples.var(0)

    if return_mean:
        y_nll_mean = y_nll_mean.mean(-1)
        y_nll_var = y_nll_var.mean(-1)
        y_recon_var = y_recon_var.mean(-1)

    return {
        "nll_mean": y_nll_mean,
        "nll_var": y_nll_var,
        "recon_var": y_recon_var,
        "latent_mean": y_latent_mean,
        "latent_var": y_latent_var,
    }


def bae_predict_ood_v1(
    bae_ensemble, x_train, x_test, keys=["nll_mean", "nll_var"], perc_threshold=99
):
    """
    Combine nll_mean and nll_var method
    """
    preds_train = bae_pred_all(bae_ensemble, x_train, return_mean=True)
    preds_test = bae_pred_all(bae_ensemble, x_test, return_mean=True)
    thresholds = {
        key: calc_ood_threshold(
            training_scores=preds_train[key], perc_threshold=perc_threshold
        )
        for key in keys
    }
    hard_preds = np.array(
        [
            convert_hard_predictions(
                test_scores=preds_test[key], ood_threshold=thresholds[key]
            )
            for key in keys
        ]
    )

    return hard_preds


def bae_predict_ood_v2(bae_ensemble, x_train, x_test, perc_threshold=99):
    """
    Apply Threshold on each samples. Threshold obtained from each BAE sample's training scores.
    """

    preds_train = bae_ensemble.predict_samples(x_train, select_keys=["se"]).mean(-1)[
        :, 0
    ]
    preds_test = bae_ensemble.predict_samples(x_test, select_keys=["se"]).mean(-1)[:, 0]

    thresholds = [
        calc_ood_threshold(training_scores=preds_train_i, perc_threshold=perc_threshold)
        for preds_train_i in preds_train
    ]
    hard_preds = np.array(
        [
            convert_hard_predictions(
                test_scores=preds_test_i, ood_threshold=thresholds_i
            )
            for preds_test_i, thresholds_i in zip(preds_test, thresholds)
        ]
    )

    return hard_preds


def evaluate_f1_score(hard_preds_inliers_test, hard_preds_outlier_test):
    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )
    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    threshold = 0.5
    y_scores[np.argwhere(y_scores >= threshold)] = 1
    y_scores[np.argwhere(y_scores < threshold)] = 0

    f1_score_ = f1_score(y_true, y_scores)
    print("F1-Score: {:.2f}".format(f1_score_))

    return f1_score_


def evaluate_avgprc_misclass(hard_preds_inliers_test, hard_preds_outlier_test):
    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )

    y_scores_unc = np.concatenate(
        (hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2)
    )

    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    threshold = 0.5
    y_scores[np.argwhere(y_scores >= threshold)] = 1
    y_scores[np.argwhere(y_scores < threshold)] = 0

    error = np.abs(y_scores - y_true)

    avgprc = average_precision_score(error, y_scores_unc)
    print(
        "AVG-PRC: {:.2f} , BASELINE: {:.2f}, AVG-PRC-RATIO: {:.2f}".format(
            avgprc, error.mean(), avgprc / error.mean()
        )
    )

    return avgprc


def evaluate_auprc_misclass(hard_preds_inliers_test, hard_preds_outlier_test):
    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )

    y_scores_unc = np.concatenate(
        (hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2)
    )

    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    threshold = 0.5
    y_scores[np.argwhere(y_scores >= threshold)] = 1
    y_scores[np.argwhere(y_scores < threshold)] = 0

    error = np.abs(y_scores - y_true)

    precision, recall, thresholds = precision_recall_curve(error, y_scores_unc)
    auprc = auc(recall, precision)

    print(
        "AUPRC: {:.2f} , BASELINE: {:.2f}, AUPRC-RATIO: {:.2f}".format(
            auprc, error.mean(), auprc / error.mean()
        )
    )

    return auprc


def evaluate_unc(
    hard_preds_inliers_test,
    hard_preds_outlier_test,
    uncertainty_threshold=0.95,
    decision_threshold=0.5,
):
    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )

    y_scores_unc = np.concatenate(
        (hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2)
    )

    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    decision_threshold = 0.5
    y_scores[np.argwhere(y_scores >= decision_threshold)] = 1
    y_scores[np.argwhere(y_scores < decision_threshold)] = 0
    y_scores = y_scores.astype(int)

    f1_score_high_unc = f1_score(
        y_true[np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0]],
        y_scores[np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0]],
    )
    f1_score_low_unc = f1_score(
        y_true[np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0]],
        y_scores[np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0]],
    )
    f1_score_all = f1_score(y_true, y_scores)

    perc_uncertain = (
        len(np.argwhere(y_scores_unc >= uncertainty_threshold)[:, 0])
        / y_scores_unc.shape[0]
    )
    perc_certain = (
        len(np.argwhere(y_scores_unc < uncertainty_threshold)[:, 0])
        / y_scores_unc.shape[0]
    )

    print("HIGH UNC: {:.2f}".format((f1_score_high_unc)))
    print("LOW UNC: {:.2f}".format((f1_score_low_unc)))
    print("W/O UNC: {:.2f}".format((f1_score_all)))
    print("% UNC: {:.2f}".format((perc_uncertain)))
    print("% CER: {:.2f}".format((perc_certain)))

    return f1_score_high_unc, f1_score_low_unc, perc_uncertain, perc_certain


def get_y_unc(hard_preds_inliers_test, hard_preds_outlier_test, decision_threshold=0.5):

    y_true = np.concatenate(
        (
            np.zeros(hard_preds_inliers_test.mean(0).shape[0]),
            np.ones(hard_preds_outlier_test.mean(0).shape[0]),
        )
    )

    y_scores_unc = np.concatenate(
        (hard_preds_inliers_test.std(0) * 2, hard_preds_outlier_test.std(0) * 2)
    )

    y_scores = np.concatenate(
        (hard_preds_inliers_test.mean(0), hard_preds_outlier_test.mean(0))
    )

    y_scores[np.argwhere(y_scores >= decision_threshold)] = 1
    y_scores[np.argwhere(y_scores < decision_threshold)] = 0
    y_hard_pred = y_scores.astype(int)

    return y_scores_unc, y_hard_pred, y_true


def calc_error_unc(y_scores_unc, y_true, y_hard_pred, unc_threshold=0):
    indices = np.argwhere(y_scores_unc <= unc_threshold)[:, 0]
    conf_matr = confusion_matrix(y_true[indices], y_hard_pred[indices])
    if len(conf_matr) > 1:
        tn, fp, fn, tp = conf_matr.ravel()
    else:
        return ()
    fpr = fp / (fp + tn)
    fdr = fp / (fp + tp)
    fnr = fn / (fn + tp)
    forate = fn / (fn + tn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    mcc = ((tp * tn) - (fp * fn)) / (
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    if np.isnan(mcc):
        mcc = 0
    f1 = tp / (tp + 0.5 * (fp + fn))
    perc = len(indices) / len(y_scores_unc)
    ba = (tpr + tnr) / 2

    return fpr, fdr, fnr, forate, tpr, tnr, mcc, f1, perc, ba


def convert_ecdf_output(bae_nll_train, bae_nll_test, scaling=True):
    ecdf_outputs = []
    for bae_sample_train_i, bae_sample_test_i in zip(bae_nll_train, bae_nll_test):
        ecdf_model = ECDF(bae_sample_train_i)
        ecdf_output = ecdf_model(bae_sample_test_i)
        if scaling:
            ecdf_output = np.clip(
                (ecdf_output - ecdf_model(bae_sample_train_i.mean()))
                / (1 - ecdf_model(bae_sample_train_i.mean())),
                0,
                1,
            )

        ecdf_outputs.append(ecdf_output)
    return np.array(ecdf_outputs)


def convert_single_ecdf(
    bae_sample_train_i, bae_sample_test_i, scaling=True, min_level="mean"
):
    ecdf_model = ECDF(bae_sample_train_i)
    ecdf_output = ecdf_model(bae_sample_test_i)
    if scaling:
        if min_level == "quartile":
            ecdf_output = np.clip(
                (ecdf_output - ecdf_model(np.percentile(bae_sample_train_i, 75)))
                / (1 - ecdf_model(np.percentile(bae_sample_train_i, 75))),
                0,
                1,
            )
        elif min_level == "median":
            ecdf_output = np.clip(
                (ecdf_output - ecdf_model(np.percentile(bae_sample_train_i, 50)))
                / (1 - ecdf_model(np.percentile(bae_sample_train_i, 50))),
                0,
                1,
            )
        elif min_level == "mean":
            ecdf_output = np.clip(
                (ecdf_output - ecdf_model(bae_sample_train_i.mean()))
                / (1 - ecdf_model(bae_sample_train_i.mean())),
                0,
                1,
            )
    return ecdf_output


def convert_erf(bae_nll_train, bae_nll_test):
    ecdf_outputs = []
    for bae_sample_train_i, bae_sample_test_i in zip(bae_nll_train, bae_nll_test):
        pre_erf_score = (bae_sample_test_i - np.mean(bae_sample_train_i)) / (
            np.std(bae_sample_train_i) * np.sqrt(2)
        )
        erf_score = erf(pre_erf_score)
        erf_score = erf_score.clip(0, 1).ravel()

        ecdf_outputs.append(erf_score)
    return np.array(ecdf_outputs)


def convert_minmax(bae_nll_train, bae_nll_test):
    ecdf_outputs = []
    for bae_sample_train_i, bae_sample_test_i in zip(bae_nll_train, bae_nll_test):

        scaler_ = MinMaxScaler().fit(bae_sample_train_i.reshape(-1, 1))
        score = scaler_.transform(bae_sample_test_i.reshape(-1, 1)).ravel().clip(0, 1)

        ecdf_outputs.append(score)
    return np.array(ecdf_outputs)


def convert_single_gmm(
    bae_sample_train_i, bae_sample_test_i, scaling=True, max_gmm_search=3
):

    # scale data for better fitting to GMM
    if scaling:
        nll_scaler = RobustScaler()
        nll_inliers_valid_scaled = nll_scaler.fit_transform(
            bae_sample_train_i.reshape(-1, 1)
        )
        nll_inliers_test_scaled = nll_scaler.transform(bae_sample_test_i.reshape(-1, 1))
    else:
        nll_inliers_valid_scaled = bae_sample_train_i.reshape(-1, 1)
        nll_inliers_test_scaled = bae_sample_test_i.reshape(-1, 1)

    # search for gmm k components
    bics = []
    for k_components in range(2, max_gmm_search + 1):
        gm = GaussianMixture(n_components=k_components).fit(nll_inliers_valid_scaled)
        gm_bic = gm.bic(nll_inliers_valid_scaled)
        bics.append(gm_bic)
    best_k = np.argmin(bics) + 2

    # select k-th component corresponding to min nll
    gm_squash = GaussianMixture(n_components=best_k).fit(nll_inliers_valid_scaled)
    gmm_proba_valid = gm_squash.predict_proba(nll_inliers_valid_scaled)
    select_k = np.argmax(gmm_proba_valid[np.argmin(nll_inliers_valid_scaled)])

    # get outlier proba
    outlier_proba = 1 - gm_squash.predict_proba(nll_inliers_test_scaled)[:, select_k]
    return outlier_proba


def convert_cdf(
    bae_nll_train, bae_nll_test, dist=gamma, scaling=True, min_level="mean", **kwargs
):
    """
    Converts to outlier score to outlier probability depending on the chosen distribution.

    `min_level` is only applicable when scaling is applied. Options include either "mean" , or int percentile.

    """
    cdf_outputs = []
    dist_dict = {
        "gamma": gamma,
        "expon": expon,
        "beta": beta,
        "lognorm": lognorm,
        "norm": norm,
        "uniform": uniform,
        "expon": expon,
    }
    if isinstance(dist, str) and dist in dist_dict:
        dist = dist_dict[dist]

    for bae_sample_train_i, bae_sample_test_i in zip(bae_nll_train, bae_nll_test):
        if isinstance(dist, str) and dist == "ecdf":
            cdf_score = convert_single_ecdf(
                bae_sample_train_i,
                bae_sample_test_i,
                scaling=scaling,
                min_level=min_level,
            )
        elif isinstance(dist, str) and dist == "gmm":
            cdf_score = convert_single_gmm(
                bae_sample_train_i, bae_sample_test_i, scaling=scaling, **kwargs
            )
        else:
            prob_args = dist.fit(bae_sample_train_i)
            dist_ = dist(*prob_args)
            cdf_score = dist_.cdf(bae_sample_test_i)

            if scaling:
                if min_level == "quartile":
                    cdf_score = np.clip(
                        (cdf_score - dist_.cdf(np.percentile(bae_sample_train_i, 75)))
                        / (1 - dist_.cdf(np.percentile(bae_sample_train_i, 75))),
                        0,
                        1,
                    )
                elif min_level == "median":
                    cdf_score = np.clip(
                        (cdf_score - dist_.cdf(np.percentile(bae_sample_train_i, 50)))
                        / (1 - dist_.cdf(np.percentile(bae_sample_train_i, 50))),
                        0,
                        1,
                    )
                elif min_level == "mean":
                    cdf_score = np.clip(
                        (cdf_score - dist_.cdf(bae_sample_train_i.mean()))
                        / (1 - dist_.cdf(bae_sample_train_i.mean())),
                        0,
                        1,
                    )
        cdf_outputs.append(cdf_score)
    return np.array(cdf_outputs)


def convert_prob(x, threshold_lb=90, threshold_ub=100):
    prob_y = x.copy()
    if threshold_lb is not None and threshold_ub is not None:
        m = 1 / (threshold_ub - threshold_lb)
        c = -threshold_lb / (threshold_ub - threshold_lb)

        prob_y = np.clip(m * prob_y + c, 0, 1)

    elif (threshold_ub is None) and (threshold_lb is not None):
        np.place(prob_y, prob_y < threshold_lb, 0)

    elif threshold_lb is None and threshold_ub is not None:
        np.place(prob_y, prob_y >= threshold_ub, 0)

    unc_y = prob_y * (1 - prob_y)

    return prob_y, unc_y


def convert_hard_pred(prob, p_threshold=0.5):
    hard_inliers_test = np.piecewise(
        prob, [prob < p_threshold, prob >= p_threshold], [0, 1]
    ).astype(int)
    return hard_inliers_test


def get_pred_unc_depracated(prob, unc, type=["epistemic", "aleatoric"]):
    if "epistemic" in type:
        epi = prob.var(0)
    else:
        epi = np.zeros(unc.shape[1:])
    if "aleatoric" in type:
        alea = unc.mean(0)
    else:
        alea = np.zeros(unc.shape[1:])
    return epi + alea


def get_pred_unc(prob, unc):
    epi = prob.var(0)
    alea = unc.mean(0)
    return {"epistemic": epi, "aleatoric": alea, "total": epi + alea}


def get_y_results(
    prob_inliers_test_mean,
    prob_outliers_test_mean,
    total_unc_inliers_test,
    total_unc_outliers_test,
    p_threshold=0.5,
):

    hard_inliers_test = convert_hard_pred(
        prob_inliers_test_mean, p_threshold=p_threshold
    )
    hard_outliers_test = convert_hard_pred(
        prob_outliers_test_mean, p_threshold=p_threshold
    )

    y_unc = {}
    for key in ["epistemic", "aleatoric", "total"]:
        y_unc.update(
            {
                key: np.concatenate(
                    (
                        total_unc_inliers_test[key],
                        total_unc_outliers_test[key],
                    )
                )
            }
        )
    y_soft_pred = np.concatenate((prob_inliers_test_mean, prob_outliers_test_mean))
    y_hard_pred = np.concatenate((hard_inliers_test, hard_outliers_test))
    # y_true = np.concatenate(
    #     (np.zeros_like(hard_inliers_test), np.ones_like(hard_outliers_test))
    # )

    y_true = get_y_true(hard_inliers_test, hard_outliers_test)

    return y_true, y_hard_pred, y_unc, y_soft_pred


def get_y_true(inliers, outliers):
    y_true = np.concatenate((np.zeros(len(inliers)), np.ones(len(outliers))))
    return y_true


def get_pred_optimal(
    prob_inliers_test_mean,
    prob_outliers_test_mean,
):
    optimal_threshold = get_optimal_threshold(
        prob_inliers_test_mean, prob_outliers_test_mean
    )

    hard_inliers_test = convert_hard_pred(
        prob_inliers_test_mean, p_threshold=optimal_threshold
    )
    hard_outliers_test = convert_hard_pred(
        prob_outliers_test_mean, p_threshold=optimal_threshold
    )

    y_true = np.concatenate(
        (
            np.zeros(prob_inliers_test_mean.shape[0]),
            np.ones(prob_outliers_test_mean.shape[0]),
        )
    )
    y_hard_optim = np.concatenate((hard_inliers_test, hard_outliers_test))

    return y_true, y_hard_optim


def evaluate_mcc_f1_unc(y_true, y_hard_pred, y_unc):
    unc_thresholds_ = np.unique(np.round(y_unc, 3))
    unc_thresholds = []
    error_uncs = []
    for unc_ in unc_thresholds_:
        error_unc = calc_error_unc(y_unc, y_true, y_hard_pred, unc_threshold=unc_)
        if len(error_unc) > 0:
            error_uncs.append(error_unc)
            unc_thresholds.append(unc_)
    unc_thresholds = np.array(unc_thresholds)
    error_uncs = np.array(error_uncs)

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1)
    ax1.plot(unc_thresholds, error_uncs[:, 0], "-o")
    ax2.plot(unc_thresholds, error_uncs[:, 1], "-o")
    ax3.plot(unc_thresholds, error_uncs[:, 3], "-o")
    ax4.plot(unc_thresholds, error_uncs[:, 5], "-o")
    ax5.plot(unc_thresholds, error_uncs[:, 6], "-o")
    ax6.plot(unc_thresholds, error_uncs[:, 7], "-o")
    ax7.plot(unc_thresholds, error_uncs[:, 8], "-o")

    spman_mcc = spearmanr(unc_thresholds, error_uncs[:, 6])[0]
    spman_f1 = spearmanr(unc_thresholds, error_uncs[:, 7])[0]

    mcc_diff = error_uncs[0, 6] - error_uncs[-1, 6]
    f1_diff = error_uncs[0, 7] - error_uncs[-1, 7]

    print("SPMAN MCC : {:.3f}".format(spman_mcc))
    print("SPMAN F1 : {:.3f}".format(spman_f1))
    print("DIFF MCC : {:.3f}".format(mcc_diff))
    print("DIFF F1 : {:.3f}".format(f1_diff))
    print("PERC HIGH : {:.3f}".format(error_uncs[0, 8]))
    print("MCC HIGH : {:.3f}".format(error_uncs[0, 6]))
    print("MCC LOW : {:.3f}".format(error_uncs[-1, 6]))
    print("F1 HIGH : {:.3f}".format(error_uncs[0, 7]))
    print("F1 LOW : {:.3f}".format(error_uncs[-1, 7]))
    print("MCC MEAN : {:.3f}".format(error_uncs[:, 6].mean()))
    print("F1 MEAN : {:.3f}".format(error_uncs[:, 7].mean()))


def get_indices_error(y_true, y_hard_pred, y_unc):

    indices_tp = np.argwhere((y_true == 1) & (y_hard_pred == 1))[:, 0]
    indices_tn = np.argwhere((y_true == 0) & (y_hard_pred == 0))[:, 0]
    indices_fp = np.argwhere((y_true == 0) & (y_hard_pred == 1))[:, 0]
    indices_fn = np.argwhere((y_true == 1) & (y_hard_pred == 0))[:, 0]
    indices_0_error = np.concatenate((indices_tp, indices_tn))
    indices_all_error = np.concatenate((indices_fp, indices_fn))

    error_type1 = np.concatenate(
        (np.ones(len(indices_fp)), np.zeros(len(indices_tp)))
    ).astype(int)
    error_type2 = np.concatenate(
        (np.ones(len(indices_fn)), np.zeros(len(indices_tn)))
    ).astype(int)
    error_all = np.abs((y_true - y_hard_pred))

    y_unc_type1 = np.concatenate((y_unc[indices_fp], y_unc[indices_tp]))
    y_unc_type2 = np.concatenate((y_unc[indices_fn], y_unc[indices_tn]))
    y_unc_all = y_unc.copy()

    return (
        indices_tp,
        indices_tn,
        indices_fp,
        indices_fn,
        indices_0_error,
        indices_all_error,
        error_type1,
        error_type2,
        error_all,
        y_unc_type1,
        y_unc_type2,
        y_unc_all,
    )


def evaluate_unc_perf(y_true, y_hard_pred, y_unc, verbose=True):
    (
        indices_tp,
        indices_tn,
        indices_fp,
        indices_fn,
        indices_0_error,
        indices_all_error,
        error_type1,
        error_type2,
        error_all,
        y_unc_type1,
        y_unc_type2,
        y_unc_all,
    ) = get_indices_error(y_true, y_hard_pred, y_unc)

    precision_type1, recall_type1, thresholds = precision_recall_curve(
        error_type1, y_unc_type1
    )
    precision_type2, recall_type2, thresholds = precision_recall_curve(
        error_type2, y_unc_type2
    )
    precision_type_all, recall_type_all, thresholds = precision_recall_curve(
        error_all, y_unc
    )

    auprc_type1 = auc(recall_type1, precision_type1)
    auprc_type2 = auc(recall_type2, precision_type2)
    auprc_type_all = auc(recall_type_all, precision_type_all)

    baseline_type1 = error_type1.mean()
    baseline_type2 = error_type2.mean()
    baseline_all = error_all.mean()

    lift_type1 = auprc_type1 / baseline_type1
    lift_type2 = auprc_type2 / baseline_type2
    lift_all = auprc_type_all / baseline_all

    auroc_type1 = (
        roc_auc_score(error_type1, y_unc_type1)
        if (baseline_type1 > 0 or baseline_type1 == 1)
        else np.nan
    )
    auroc_type2 = (
        roc_auc_score(error_type2, y_unc_type2)
        if (baseline_type2 > 0 or baseline_type2 == 1)
        else np.nan
    )
    auroc_type_all = (
        roc_auc_score(error_all, y_unc)
        if (baseline_all > 0 or baseline_all == 1)
        else np.nan
    )

    if verbose:
        print(
            "AUPRC-TYPE1 (TP VS FP): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(
                auprc_type1, baseline_type1, lift_type1
            )
        )
        print(
            "AUPRC-TYPE2 (TN VS FN): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(
                auprc_type2, baseline_type2, lift_type2
            )
        )
        print(
            "AUPRC-ALL   (TP+TN VS FP+FN): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(
                auprc_type_all, baseline_all, lift_all
            )
        )

        print("AUROC-TYPE1 (TP VS FP): {:.2f} ".format(auroc_type1))
        print("AUROC-TYPE2 (TN VS FN): {:.2f} ".format(auroc_type2))
        print("AUROC-ALL   (TP+TN VS FP+FN): {:.2f} ".format(auroc_type_all))

    return (
        auprc_type1,
        auprc_type2,
        auprc_type_all,
        baseline_type1,
        baseline_type2,
        baseline_all,
        lift_type1,
        lift_type2,
        lift_all,
        auroc_type1,
        auroc_type2,
        auroc_type_all,
    )


def eval_auroc_ood(y_true, dict_y_soft_preds):
    """
    Evaluate auprc and auroc for ood detection.
    The soft scores are expected to be in dictionary.
    """
    auprc_base = y_true.mean()
    for i, (key, y_soft_pred) in enumerate(dict_y_soft_preds.items()):
        precision_, recall_, thresholds_ = precision_recall_curve(y_true, y_soft_pred)
        auprc_ = auc(recall_, precision_)
        avg_prc = average_precision_score(y_true, y_soft_pred)
        auroc_ = roc_auc_score(y_true, y_soft_pred)
        auprc_ratio = auprc_ / auprc_base
        res_ = {
            "auprc-base": auprc_base,
            "method": key,
            "auprc": auprc_,
            "avg_prc": avg_prc,
            "auprc-ratio": auprc_ratio,
            "auroc": auroc_,
        }

        if i == 0:
            res = pd.DataFrame([res_])
        else:
            res = res.append(pd.DataFrame([res_]))

    return res


def eval_retained_unc(y_true, y_hard_pred, y_unc_scaled, y_soft_pred):
    unc_thresholds_ = np.unique(y_unc_scaled)
    retained_metrics = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for unc_ in unc_thresholds_:
            metrics = calc_performance_unc(
                y_unc_scaled,
                y_true,
                y_hard_pred,
                y_soft_pred=y_soft_pred,
                unc_threshold=unc_,
            )
            if len(metrics) > 0:
                retained_metrics_ = metrics
                retained_metrics.append(retained_metrics_)

    retained_metrics = pd.DataFrame(retained_metrics)
    res_wmean = pd.DataFrame(
        (
            (retained_metrics.T * retained_metrics["perc"]).sum(1)
            / retained_metrics["perc"].sum()
        ).round(3)
    ).T
    filter_metrics = [
        "f1",
        "mcc",
        "gmean_ss",
        "auroc",
        "auprc",
        "auprc-ratio",
        "avg_prc",
    ]
    res_max = pd.DataFrame(retained_metrics.max(0)[filter_metrics]).T
    res_spman = calc_spman_metrics(retained_metrics)[filter_metrics]

    res_baselines = pd.DataFrame(
        [calc_metrics2(y_true, y_hard_pred, y_soft_pred=y_soft_pred)]
    )
    return retained_metrics, res_wmean, res_max, res_spman, res_baselines


def evaluate_error_unc(y_true, y_hard_pred, y_unc, verbose=True, return_df=True):

    error_all = np.abs((y_true - y_hard_pred))

    precision_type_all, recall_type_all, thresholds = precision_recall_curve(
        error_all, y_unc
    )
    auprc_type_all = auc(recall_type_all, precision_type_all)

    baseline_all = error_all.mean()

    lift_all = auprc_type_all / baseline_all

    avg_prc = average_precision_score(error_all, y_unc)
    auroc_type_all = (
        roc_auc_score(error_all, y_unc)
        if (baseline_all > 0 or baseline_all == 1)
        else np.nan
    )

    if verbose:
        print(
            "AUPRC-ERR : {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(
                auprc_type_all, baseline_all, lift_all
            )
        )
        print("AUROC-ERR : {:.2f} ".format(auroc_type_all))

    res = {
        "AUPR-BASE-ERR": np.round(baseline_all, 3),
        "AUPRC-ERR": np.round(auprc_type_all, 3),
        "AUPR-RATIO-ERR": np.round(lift_all, 3),
        "AVGPRC-ERR": np.round(avg_prc, 3),
        "AUROC-ERR": np.round(auroc_type_all, 3),
    }

    if return_df:
        res = pd.DataFrame([res])

    return res


def rename_col_res(res_baselines, res_wmean, res_max, res_spman):
    res_baselines.columns = ["base-" + col for col in res_baselines.columns]
    res_wmean.columns = ["wmean-" + col for col in res_wmean.columns]
    res_max.columns = ["max-" + col for col in res_max.columns]
    res_spman.columns = ["spman-" + col for col in res_spman.columns]

    return pd.concat((res_baselines, res_wmean, res_max, res_spman), axis=1)


def plot_unc_tptnfpfn(y_true, y_hard_pred, y_unc):
    (
        indices_tp,
        indices_tn,
        indices_fp,
        indices_fn,
        indices_0_error,
        indices_all_error,
        error_type1,
        error_type2,
        error_all,
        y_unc_type1,
        y_unc_type2,
        y_unc_all,
    ) = get_indices_error(y_true, y_hard_pred, y_unc)

    labels = ["TP", "TN", "TP+TN", "Type 1 (FP)", "Type 2 (FN)", "FP+FN"]
    fig, ax1 = plt.subplots(1, 1)
    ax1.boxplot(
        [
            y_unc[indices_tp],
            y_unc[indices_tn],
            y_unc[indices_0_error],
            y_unc[indices_fp],
            y_unc[indices_fn],
            y_unc[indices_all_error],
        ],
        notch=False,
    )

    ax1.set_xticks(np.arange(1, len(labels) + 1))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Uncertainty")


def calc_f1_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn)


def calc_mcc_score(tp, fp, fn, tn):
    mcc = ((tp * tn) - (fp * fn)) / (
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return mcc


def calc_perc_indices(indices_len, total_samples):
    perc = indices_len / total_samples
    return perc


def calc_precision(tp, fp):
    return tp / (tp + fp)


def calc_fdrate(tp, fp):
    fdr = fp / (fp + tp)
    return fdr


def calc_forate(fn, tn):
    forate = fn / (fn + tn)
    return forate


def calc_gmean_ss(tp, fp, fn, tn):
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return np.sqrt(tpr * tnr)


def calc_hard_metrics(tp, fp, fn, tn, indices_len=None, total_samples=None):
    results = {
        "f1": calc_f1_score(tp, fp, fn),
        "mcc": calc_mcc_score(tp, fp, fn, tn),
        # "precision": calc_precision(tp, fp),
        "gmean_ss": calc_gmean_ss(tp, fp, fn, tn),
        # "forate": calc_forate(fn, tn),
        # "fdr": calc_fdrate(tp, fp),
    }
    if indices_len is not None and total_samples is not None:
        results.update({"perc": calc_perc_indices(indices_len, total_samples)})
    return results


def calc_metrics2(y_true, y_hard_pred, y_soft_pred=None):
    """
    Returns same as calc_metrics but accepts y_true and y_hard_preds
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_hard_pred).ravel()
    res_baselines = calc_hard_metrics(tp, fp, fn, tn)

    if y_soft_pred is not None:
        soft_res = calc_soft_metrics(y_true, y_soft_pred)
        res_baselines.update(soft_res)
    return res_baselines


def calc_soft_metrics(y_true, y_soft_pred):
    if np.sum(y_true) > 10:
        auroc_ = roc_auc_score(y_true, y_soft_pred)
        avg_prc = average_precision_score(y_true, y_soft_pred)
        precision, recall, thresholds = precision_recall_curve(y_true, y_soft_pred)
        auprc_ = auc(recall, precision)
        auprc_base = y_true.mean()
        res_baselines = {
            "auroc": auroc_,
            "auprc": auprc_,
            "auprc-base": y_true.mean(),
            "auprc-ratio": auprc_ / auprc_base,
            "avg_prc": avg_prc,
        }
    else:
        res_baselines = {
            "auroc": np.nan,
            "auprc": np.nan,
            "auprc-base": np.nan,
            "auprc-ratio": np.nan,
            "avg_prc": np.nan,
        }
    return res_baselines


def calc_spman_metrics(retained_metrics):
    res_spman = {}
    perc_retained = retained_metrics["perc"].values
    for key in retained_metrics.columns:
        sel_col = retained_metrics[key].values
        sp_ = spearmanr(perc_retained, sel_col)
        res_spman.update({key: -1 * sp_[0] if sp_[1] <= 0.05 else 0})
    res_spman = pd.DataFrame([res_spman])
    return res_spman


def calc_performance_unc(
    y_scores_unc, y_true, y_hard_pred, y_soft_pred=None, unc_threshold=0
):
    # specify retained and rejected indices
    retained_indices = np.argwhere(y_scores_unc <= unc_threshold)[:, 0]

    # get their confusion matrix
    retained_matr = confusion_matrix(
        y_true[retained_indices], y_hard_pred[retained_indices]
    )

    # sanity check for existence of confusion matrix
    # if doesn't exist, then exit
    if len(retained_matr) > 1:
        tn, fp, fn, tp = retained_matr.ravel()
        retained_metrics = calc_hard_metrics(
            tp, fp, fn, tn, len(retained_indices), len(y_scores_unc)
        )
        if y_soft_pred is not None:
            retained_metrics.update(
                calc_soft_metrics(
                    y_true[retained_indices], y_soft_pred[retained_indices]
                )
            )

        retained_metrics.update({"threshold": unc_threshold})

    else:
        return ()

    return retained_metrics


def get_optimal_threshold(nll_inliers_test_mean, nll_outliers_test_mean):
    y_true = np.concatenate(
        (
            np.zeros(nll_inliers_test_mean.shape[0]),
            np.ones(nll_outliers_test_mean.shape[0]),
        )
    )
    y_scores = np.concatenate((nll_inliers_test_mean, nll_outliers_test_mean))
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = roc_thresholds[optimal_idx]
    return optimal_threshold


def calc_outlier_unc(bae_outprob):
    """
    Expect the BAE samples of outlier probabilities with shape of (BAE samples , Num examples)
    """

    epistemic_unc = bae_outprob.var(0)
    aleatoric_unc = (bae_outprob * (1 - bae_outprob)).mean(0)
    total_unc = epistemic_unc + aleatoric_unc

    return {"epi": epistemic_unc, "alea": aleatoric_unc, "total": total_unc}
