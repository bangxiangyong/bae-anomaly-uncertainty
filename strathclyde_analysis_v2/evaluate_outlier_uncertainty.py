import datetime

import numpy as np

from baetorch.baetorch.evaluation import (
    concat_ood_score,
    evaluate_random_retained_unc,
    evaluate_misclas_detection,
    convert_hard_pred,
    summarise_retained_perf,
    evaluate_retained_unc_v2,
    calc_auroc,
)
from baetorch.baetorch.models_v2.outlier_proba import BAE_Outlier_Proba
from baetorch.baetorch.models_v2.vae import VAE
from uncertainty_ood.exceed import calc_exceed
from uncertainty_ood_v2.util.get_predictions import flatten_nll, calc_e_nll
from util.evaluate_ood import evaluate_bce_se
from util.exp_manager import ExperimentManager


def evaluate_ood_unc(
    bae_model,
    x_id_train,
    x_id_test,
    x_ood_test,
    exp_name,
    exp_params,
    nll_key="nll",
    eval_ood_unc=False,
    cdf_dists=["norm", "uniform", "ecdf"],
    norm_scalings=[True, False],
    ret_flatten_nll=True,
    exp_man=None,
    round_deci=0,
    hard_threshold=0.5,
):
    # === PREDICTIONS ===
    bae_id_pred = bae_model.predict(x_id_test, select_keys=[nll_key])
    bae_ood_pred = bae_model.predict(x_ood_test, select_keys=[nll_key])

    # get ood scores
    e_nll_id = flatten_nll(bae_id_pred[nll_key]).mean(0)
    e_nll_ood = flatten_nll(bae_ood_pred[nll_key]).mean(0)
    var_nll_id = flatten_nll(bae_id_pred[nll_key]).var(0)
    var_nll_ood = flatten_nll(bae_ood_pred[nll_key]).var(0)

    eval_auroc = {
        "E_AUROC": calc_auroc(e_nll_id, e_nll_ood),
        "V_AUROC": calc_auroc(var_nll_id, var_nll_ood),
    }

    if exp_man is None:
        exp_man = ExperimentManager()
    res = exp_man.concat_params_res(exp_params, eval_auroc)
    exp_man.update_csv(exp_params=res, csv_name=exp_name + "AUROC.csv")

    # special case for evaluating bce vs mse
    if (
        bae_model.likelihood == "gaussian"
        and not bae_model.twin_output
        and bae_model.homoscedestic_mode == "none"
    ) or bae_model.likelihood == "bernoulli":
        eval_auroc_bce_se = evaluate_bce_se(bae_model, x_id_test, x_ood_test)
        if exp_man is None:
            exp_man = ExperimentManager()
        res = exp_man.concat_params_res(exp_params, eval_auroc_bce_se)
        exp_man.update_csv(exp_params=res, csv_name=exp_name + "BCE_VS_SE.csv")

    # === EVALUATE OUTLIER UNCERTAINTY ===
    if eval_ood_unc:
        # convert to outlier probability
        # 1. get reference distribution of NLL scores
        bae_id_ref_pred = bae_model.predict(x_id_train, select_keys=[nll_key])

        all_y_true = np.concatenate(
            (np.zeros_like(e_nll_id), np.ones_like(e_nll_ood))
        ).astype(int)
        all_var_nll_unc = np.concatenate((var_nll_id, var_nll_ood))
        concat_e_nll = concat_ood_score(e_nll_id, e_nll_ood)[1]

        # 2. define cdf distribution of OOD scores
        for cdf_dist in cdf_dists:
            bae_proba_model = BAE_Outlier_Proba(
                dist_type=cdf_dist,
                norm_scaling=True,
                fit_per_bae_sample=False if isinstance(bae_model, VAE) else True,
            )
            bae_proba_model.fit(bae_id_ref_pred[nll_key])

            for norm_scaling in norm_scalings:
                id_proba_mean, id_proba_unc = bae_proba_model.predict(
                    bae_id_pred[nll_key], norm_scaling=norm_scaling
                )
                ood_proba_mean, ood_proba_unc = bae_proba_model.predict(
                    bae_ood_pred[nll_key], norm_scaling=norm_scaling
                )

                # CONVERT HARD PRED
                all_proba_mean = np.concatenate((id_proba_mean, ood_proba_mean))
                all_hard_proba_pred = convert_hard_pred(
                    all_proba_mean, p_threshold=hard_threshold
                )
                all_hard_proba_pred_ex = convert_hard_pred(
                    concat_e_nll, p_threshold=np.max(e_nll_id)
                )
                # EXCEED UNCERTAINTY
                all_exceed_unc = calc_exceed(
                    len(calc_e_nll(bae_id_ref_pred)),
                    all_proba_mean,
                    # all_hard_proba_pred,
                    all_hard_proba_pred_ex,
                    contamination=0.0,
                )

                # Evalute uncertainty performances
                retained_percs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

                # Evaluate retained performance
                retained_varnll_res = evaluate_retained_unc_v2(
                    all_outprob_mean=concat_e_nll,
                    all_hard_pred=all_hard_proba_pred,
                    all_y_true=all_y_true,
                    all_unc=all_var_nll_unc,
                )

                retained_exceed_res = evaluate_retained_unc_v2(
                    all_outprob_mean=concat_e_nll,
                    all_hard_pred=all_hard_proba_pred,
                    # all_hard_pred=all_hard_proba_pred_ex,
                    all_y_true=all_y_true,
                    all_unc=all_exceed_unc,
                    round_deci=round_deci,
                )

                retained_random_res = evaluate_random_retained_unc(
                    all_outprob_mean=concat_e_nll,
                    all_hard_pred=all_hard_proba_pred,
                    all_y_true=all_y_true,
                    repetition=10,
                    retained_percs=retained_percs,
                )

                # evaluate misclassification detection
                misclas_varnll_res = evaluate_misclas_detection(
                    all_y_true,
                    all_hard_proba_pred,
                    all_var_nll_unc,
                    return_boxplot=True,
                )
                misclas_exceed_res = evaluate_misclas_detection(
                    all_y_true,
                    all_hard_proba_pred,
                    # all_hard_proba_pred_ex,
                    all_exceed_unc,
                    return_boxplot=True,
                )

                # Save all results in dicts
                retained_res_all = {}
                misclas_res_all = {}
                retained_res_all.update(
                    {
                        "varnll": retained_varnll_res,
                        "exceed": retained_exceed_res,
                        "random": retained_random_res,
                    }
                )

                misclas_res_all.update(
                    {
                        "varnll": misclas_varnll_res,
                        "exceed": misclas_exceed_res,
                    }
                )

                for proba_unc_key in ["epi", "alea", "total"]:
                    all_proba_unc = np.concatenate(
                        (id_proba_unc[proba_unc_key], ood_proba_unc[proba_unc_key])
                    )
                    retained_prob_unc_res = evaluate_retained_unc_v2(
                        all_outprob_mean=concat_e_nll,
                        all_hard_pred=all_hard_proba_pred,
                        all_y_true=all_y_true,
                        all_unc=all_proba_unc,
                        round_deci=round_deci,
                    )

                    misclas_prob_unc_res = evaluate_misclas_detection(
                        all_y_true,
                        all_hard_proba_pred,
                        all_proba_unc,
                        return_boxplot=True,
                    )

                    retained_res_all.update(
                        {"proba-" + proba_unc_key: retained_prob_unc_res}
                    )
                    misclas_res_all.update(
                        {"proba-" + proba_unc_key: misclas_prob_unc_res}
                    )

                # Save uncertainty evaluation results in CSV
                if exp_man is None:
                    exp_man = ExperimentManager()
                unc_method = {"dist": cdf_dist, "norm": norm_scaling}
                base_method_columns = exp_man.concat_params_res(exp_params, unc_method)
                pickle_retained = exp_man.encode(
                    exp_man.concat_params_res(
                        exp_params,
                        unc_method,
                        {"restype": "retained", "date": datetime.datetime.now()},
                    )
                )
                pickle_misclas = exp_man.encode(
                    exp_man.concat_params_res(
                        exp_params,
                        unc_method,
                        {"restype": "misclas", "date": datetime.datetime.now()},
                    )
                )

                for unc_method_name in retained_res_all.keys():
                    summary_ret_res = summarise_retained_perf(
                        retained_res_all[unc_method_name], flatten_key=True
                    )
                    retained_csv = exp_man.concat_params_res(
                        base_method_columns,
                        {"unc_method": unc_method_name},
                        summary_ret_res,
                    )
                    exp_man.update_csv(
                        retained_csv,
                        insert_pickle=pickle_retained,
                        csv_name=exp_name + "retained_perf.csv",
                    )
                exp_man.encode_pickle(pickle_retained, data=retained_res_all)

                # handle misclas results
                for unc_method_name in misclas_res_all.keys():
                    misclas_csv = exp_man.concat_params_res(
                        base_method_columns,
                        {"unc_method": unc_method_name},
                        misclas_res_all[unc_method_name]["all_err"],
                    )
                    exp_man.update_csv(
                        misclas_csv,
                        insert_pickle=pickle_misclas,
                        csv_name=exp_name + "misclas_perf.csv",
                    )
                exp_man.encode_pickle(pickle_misclas, data=misclas_res_all)

    # return results
    if ret_flatten_nll and eval_ood_unc:
        return (e_nll_id, e_nll_ood, var_nll_id, var_nll_ood), (
            eval_auroc,
            retained_res_all,
            misclas_res_all,
        )
    elif ret_flatten_nll and not eval_ood_unc:
        return (e_nll_id, e_nll_ood, var_nll_id, var_nll_ood), (eval_auroc, {}, {})
    elif not ret_flatten_nll and eval_ood_unc:
        return (bae_id_pred, bae_ood_pred), (
            eval_auroc,
            retained_res_all,
            misclas_res_all,
        )
    else:
        return (bae_id_pred, bae_ood_pred), (eval_auroc, {}, {})
