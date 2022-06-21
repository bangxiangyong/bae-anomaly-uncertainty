import numpy as np
from scipy.stats import gamma, beta, lognorm, uniform, expon

from baetorch.baetorch.evaluation import (
    calc_avgprc_perf,
    evaluate_misclas_detection,
    concat_ood_score,
    evaluate_retained_unc,
    evaluate_random_retained_unc,
    retained_top_unc_indices,
    calc_auroc,
    retained_random_indices,
)
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.outlier_proba import BAE_Outlier_Proba
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood_v2.util.get_predictions import (
    calc_e_nll,
    flatten_nll,
    calc_exceed,
)
from uncertainty_ood_v2.util.prepare_anomaly_datasets import (
    list_anomaly_dataset,
    load_anomaly_dataset,
)

# ===EXPERIMENT PARAMETERS===
# fixed variables
from util.uncertainty import (
    convert_hard_pred,
    get_optimal_threshold,
)

base_folder = "anomaly_datasets"
train_size = 0.8
# valid_size = 0.1
valid_size = 0.0

# manipulated variables
random_seed = 123
dataset_id = -1

# set random seed
bae_set_seed(random_seed)

# ===DATASET PREPARATION===
anomaly_datasets = list_anomaly_dataset()
if valid_size == 0:
    x_id_train, x_id_test, x_ood = load_anomaly_dataset(
        mat_file_id=anomaly_datasets[dataset_id],
        base_folder=base_folder,
        train_size=train_size,
        random_seed=random_seed,
    )
else:
    x_id_train, x_id_test, x_ood, x_valid = load_anomaly_dataset(
        mat_file_id=anomaly_datasets[dataset_id],
        base_folder=base_folder,
        train_size=train_size,
        random_seed=random_seed,
        valid_size=valid_size,
    )

x_id_train_loader = convert_dataloader(
    x_id_train,
    batch_size=len(x_id_train) // 3,
    shuffle=True,
)

# ===DEFINE BAE===

# bae params
skip = True
# skip = False
use_cuda = True
# twin_output = True
twin_output = False
homoscedestic_mode = "every"
# homoscedestic_mode = "none"
# homoscedestic_mode = "single"
likelihood = "gaussian"
# likelihood = "laplace"
# likelihood = "bernoulli"
# likelihood = "cbernoulli"
# likelihood = "truncated_gaussian"
weight_decay = 0.0000000001
# weight_decay = 0.000000001
# weight_decay = 0.0001
# weight_decay = 0.01
# weight_decay = 0.000001
# weight_decay = 0.000
# anchored = False
anchored = True
# sparse_scale = 0.0000001
sparse_scale = 0.00
n_stochastic_samples = 100
# n_ensemble = 10
n_ensemble = 5
input_dim = x_id_train.shape[1]
num_epochs = 300
norm = "layer"

chain_params = [
    {
        "base": "linear",
        "architecture": [input_dim, input_dim * 4, input_dim * 4],
        "activation": "selu",
        "norm": norm,
        "last_norm": norm,
    }
]

bae_model = BAE_Ensemble(
    chain_params=chain_params,
    last_activation="sigmoid",
    twin_output=twin_output,
    # twin_params={"activation": "none", "norm": False},
    # twin_params={"activation": "leakyrelu", "norm": False},  # truncated gaussian
    # twin_params={"activation": "leakyrelu", "norm": True},  # truncated gaussian
    # twin_params={"activation": "leakyrelu", "norm": False}
    # if likelihood == "gaussian"
    # else {"activation": "leakyrelu", "norm": False},
    # twin_params={"activation": "softplus", "norm": "none"},
    twin_params={"activation": "none", "norm": "none"},
    skip=skip,
    use_cuda=use_cuda,
    homoscedestic_mode=homoscedestic_mode,
    likelihood=likelihood,
    weight_decay=weight_decay,
    num_samples=n_ensemble,
    sparse_scale=sparse_scale,
    anchored=anchored,
)

#
# bae_model = BAE_SGHMC(
#     chain_params=chain_params,
#     last_activation="sigmoid",
#     twin_output=twin_output,
#     # twin_params={"activation": "none", "norm": False},
#     # twin_params={"activation": "leakyrelu", "norm": False},  # truncated gaussian
#     # twin_params={"activation": "leakyrelu", "norm": True},  # truncated gaussian
#     # twin_params={"activation": "leakyrelu", "norm": False}
#     # if likelihood == "gaussian"
#     # else {"activation": "leakyrelu", "norm": False},
#     # twin_params={"activation": "softplus", "norm": True},
#     twin_params={"activation": "none", "norm": "none"},
#     skip=skip,
#     use_cuda=use_cuda,
#     homoscedestic_mode=homoscedestic_mode,
#     likelihood=likelihood,
#     weight_decay=weight_decay,
#     num_samples=n_ensemble,
#     # anchored=True,
# )

min_lr, max_lr, half_iter = run_auto_lr_range_v4(
    x_id_train_loader,
    bae_model,
    window_size=1,
    num_epochs=10,
    run_full=False,
)

if isinstance(bae_model, BAE_SGHMC):
    bae_model.fit(x_id_train_loader, burn_epoch=num_epochs, sghmc_epoch=num_epochs // 2)
else:
    # time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs + 50)
    time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)


# === PREDICTIONS ===
# start predicting
nll_key = "nll"

bae_id_pred = bae_model.predict(x_id_test, select_keys=[nll_key])
bae_ood_pred = bae_model.predict(x_ood, select_keys=[nll_key])

# get ood scores
e_nll_id = flatten_nll(bae_id_pred[nll_key]).mean(0)
e_nll_ood = flatten_nll(bae_ood_pred[nll_key]).mean(0)
var_nll_id = flatten_nll(bae_id_pred[nll_key]).var(0)
var_nll_ood = flatten_nll(bae_ood_pred[nll_key]).var(0)

concat_e_nll = concat_ood_score(e_nll_id, e_nll_ood)[1]


# convert to outlier probability
# 1. get reference distribution of NLL scores
if valid_size == 0:
    bae_id_ref_pred = bae_model.predict(x_id_train, select_keys=[nll_key])
else:
    bae_id_ref_pred = bae_model.predict(x_valid, select_keys=[nll_key])

all_y_true = np.concatenate((np.zeros_like(e_nll_id), np.ones_like(e_nll_ood)))
all_var_nll_unc = np.concatenate((var_nll_id, var_nll_ood))

exceed_id_unc, exceed_id_pred = calc_exceed(
    calc_e_nll(bae_id_ref_pred), e_nll_id, contamination=0.0
)
exceed_ood_unc, exceed_ood_pred = calc_exceed(
    calc_e_nll(bae_id_ref_pred), e_nll_ood, contamination=0.0
)

all_exceed_unc = np.concatenate((exceed_id_unc, exceed_ood_unc))
all_exceed_hard_pred = np.concatenate((exceed_id_pred, exceed_ood_pred))

# 2. define cdf distribution of OOD scores
# loop here
cdf_dist = "norm"
norm_scaling = True
bae_proba_model = BAE_Outlier_Proba(
    dist_type=cdf_dist, norm_scaling=norm_scaling, fit_per_bae_sample=True
)

bae_proba_model.fit(bae_id_ref_pred[nll_key])
id_proba_mean, id_proba_unc = bae_proba_model.predict(bae_id_pred[nll_key])
ood_proba_mean, ood_proba_unc = bae_proba_model.predict(bae_ood_pred[nll_key])


all_hard_pred = convert_hard_pred(
    concat_e_nll, p_threshold=get_optimal_threshold(e_nll_id, e_nll_ood)
)
# all_hard_pred = convert_hard_pred(np.concatenate((id_proba_mean, ood_proba_mean)),
#                                   p_threshold=get_optimal_threshold(id_proba_mean, ood_proba_mean))
# all_hard_pred = convert_hard_pred(np.concatenate((id_proba_mean, ood_proba_mean)),
#                                   p_threshold=0.8)

all_proba_mean = np.concatenate((id_proba_mean, ood_proba_mean))
# concat_e_nll = all_proba_mean
# evaluate performances
retained_percs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# retained_percs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

retained_varnll_res = evaluate_retained_unc(
    all_outprob_mean=concat_e_nll,
    all_hard_pred=all_hard_pred,
    all_y_true=all_y_true,
    all_unc=all_var_nll_unc,
    retained_percs=retained_percs,
)

retained_exceed_res = evaluate_retained_unc(
    all_outprob_mean=concat_e_nll,
    all_hard_pred=all_exceed_hard_pred,
    all_y_true=all_y_true,
    all_unc=all_exceed_unc,
    retained_percs=retained_percs,
)

retained_random_res = evaluate_random_retained_unc(
    all_outprob_mean=concat_e_nll,
    all_hard_pred=all_hard_pred,
    all_y_true=all_y_true,
    repetition=50,
    retained_percs=retained_percs,
)

misclas_varnll_res = evaluate_misclas_detection(
    all_y_true, all_hard_pred, all_var_nll_unc, return_boxplot=True
)
misclas_exceed_res = evaluate_misclas_detection(
    all_y_true, all_hard_pred, all_exceed_unc, return_boxplot=True
)

# evaluate proba unc
for proba_unc_key in ["epi", "alea", "total"]:
    all_proba_unc = np.concatenate(
        (id_proba_unc[proba_unc_key], ood_proba_unc[proba_unc_key])
    )
    retained_prob_unc_res = evaluate_retained_unc(
        all_outprob_mean=concat_e_nll,
        all_hard_pred=all_hard_pred,
        all_y_true=all_y_true,
        all_unc=all_proba_unc,
        retained_percs=retained_percs,
    )
    misclas_prob_unc_res = evaluate_misclas_detection(
        all_y_true, all_hard_pred, all_proba_unc, return_boxplot=True
    )
    print(retained_prob_unc_res["auroc"])
    print(retained_prob_unc_res["f1_score"])
    print(misclas_prob_unc_res["all_err"])

print(retained_exceed_res["f1_score"])
# # === ACTIVE LEARNING ===
# import copy
#
# retained_perc = 0.9
# (retained_unc_indices, retained_id_indices, retained_ood_indices), (
#     referred_unc_indices,
#     referred_id_indices,
#     referred_ood_indices,
# ) = retained_top_unc_indices(
#     all_y_true, all_proba_unc, retained_perc=retained_perc, return_referred=True
# )
#
# # (retained_unc_indices, retained_id_indices, retained_ood_indices), (
# #     referred_unc_indices,
# #     referred_id_indices,
# #     referred_ood_indices,
# # ) = retained_top_unc_indices(
# #     all_y_true, all_exceed_unc, retained_perc=retained_perc, return_referred=True
# # )
#
# # (retained_unc_indices, retained_id_indices, retained_ood_indices), (
# #     referred_unc_indices,
# #     referred_id_indices,
# #     referred_ood_indices,
# # ) = retained_random_indices(
# #     all_y_true,
# #     retained_perc=retained_perc,
# #     return_referred=True,
# # )
#
# referred_unc_indices_nn = np.array(
#     [
#         number
#         for number in np.arange(len(all_y_true))
#         if number not in retained_unc_indices
#     ]
# )
#
# concat_x = np.concatenate((x_id_test, x_ood))
#
# # referred indices are fed back to training data.
# # retained indices are used as the new testing data.
# x_id_train_new = concat_x[referred_unc_indices][referred_id_indices]
# x_ood_train_new = concat_x[referred_unc_indices][referred_ood_indices]
# x_id_test_new = concat_x[retained_unc_indices][retained_id_indices]
# x_ood_test_new = concat_x[retained_unc_indices][retained_ood_indices]
#
# x_id_train_loader_new = convert_dataloader(
#     np.concatenate((x_id_train, x_id_train_new)),
#     batch_size=len(x_id_train) // 4,
#     shuffle=True,
# )
#
# y_scaler = 1
# bae_model_new = copy.deepcopy(bae_model)
#
#
# # bae_model_new.init_anchored_weight()
# # bae_model_new.weight_decay = 0.01
# bae_model_new.scheduler_enabled = False
# bae_model_new.set_learning_rate(min_lr)
# bae_model_new.set_optimisers()
#
# if isinstance(bae_model_new, BAE_SGHMC):
#     bae_model_new.fit(
#         x=x_id_train_loader_new,
#         y=x_ood_train_new if len(x_ood_train_new) > 0 else None,
#         burn_epoch=num_epochs,
#         sghmc_epoch=num_epochs // 2,
#         clear_sghmc_params=True,
#     )
# else:
#     # time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs + 50)
#     time_method(
#         bae_model_new.fit,
#         x=x_id_train_loader_new,
#         y=x_ood_train_new if len(x_ood_train_new) > 0 else None,
#         num_epochs=20,
#     )
#
# nll_key = "nll"
# bae_id_pred = bae_model_new.predict(x_id_test_new, select_keys=[nll_key])
# bae_ood_pred = bae_model_new.predict(x_ood_test_new, select_keys=[nll_key])
#
# # get ood scores
# e_nll_id_new = flatten_nll(bae_id_pred[nll_key]).mean(0)
# e_nll_ood_new = flatten_nll(bae_ood_pred[nll_key]).mean(0)
# var_nll_id_new = flatten_nll(bae_id_pred[nll_key]).var(0)
# var_nll_ood_new = flatten_nll(bae_ood_pred[nll_key]).var(0)
#
# print(calc_auroc(var_nll_id_new,var_nll_ood_new))
# print(calc_auroc(e_nll_id_new,e_nll_ood_new))
#
# # var_nll_id = calc_var_nll(bae_id_pred)
# # var_nll_ood = calc_var_nll(bae_ood_pred)
#


# 2. define cdf distribution of OOD scores
# cdf_dist = "ecdf"
# unc_type = "total"
# scaling = (
#     False if cdf_dist == "uniform" else True
# )  # only enable outlier prob. scaling for uniform dist.
# # scaling = False
#
# # convert to OOD probability based on selected cdf distribution
# id_outprob = convert_cdf(
#     flatten_nll(bae_id_ref_pred["nll"]),
#     flatten_nll(bae_id_pred["nll"]),
#     dist=cdf_dist,
#     scaling=scaling,
#     min_level="mean",
# )
#
# ood_outprob = convert_cdf(
#     flatten_nll(bae_id_ref_pred["nll"]),
#     flatten_nll(bae_ood_pred["nll"]),
#     dist=cdf_dist,
#     scaling=scaling,
#     min_level="mean",
# )
#
# # Mean score of outlier probability
# id_outprob_mean = id_outprob.mean(0)
# ood_outprob_mean = ood_outprob.mean(0)
#
# # Convert to outlier uncertainty
# # can index by type of uncertainty : epi, alea or total
# id_unc = calc_outlier_unc(id_outprob)[unc_type]
# ood_unc = calc_outlier_unc(ood_outprob)[unc_type]
#
#
# # === PREPARE & CONCAT MODEL PREDICTIONS ===
# # p_threshold = 0.5
# p_threshold = np.percentile(flatten_nll(bae_id_ref_pred["nll"]).mean(0), 0.5)
#
# # concatenate prob scores
# all_y_true, all_outprob_mean, all_hard_pred = concat_ood_score(
#     id_outprob_mean, ood_outprob_mean, p_threshold=p_threshold
# )
#
# # prepare various measures of uncertainties
# all_prob_unc = np.concatenate((id_unc, ood_unc))
# all_var_nll_unc = np.concatenate((var_nll_id, var_nll_ood))
# all_exceed_unc = np.concatenate(
#     (
#         calc_exceed(calc_e_nll(bae_id_ref_pred), e_nll_id),
#         calc_exceed(calc_e_nll(bae_id_ref_pred), e_nll_ood),
#     )
# )
#
# # === EVALUATIONS ===
#
# # === EVALUATION OF CLASSIFIER USING E-NLL & VAR NLL & WAIC ? ===
# e_nll_perf = calc_avgprc_perf(*concat_ood_score(e_nll_id, e_nll_ood))
# var_nll_perf = calc_avgprc_perf(*concat_ood_score(var_nll_id, var_nll_ood))
#
# # === EVALUATION : RETAINED PERF ===
# retained_percs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# retained_prob_unc_res = evaluate_retained_unc(
#     all_outprob_mean=all_outprob_mean,
#     all_hard_pred=all_hard_pred,
#     all_y_true=all_y_true,
#     all_unc=all_prob_unc,
#     retained_percs=retained_percs,
# )
#
# retained_unc_res = evaluate_retained_unc(
#     all_outprob_mean=all_outprob_mean,
#     all_hard_pred=all_hard_pred,
#     all_y_true=all_y_true,
#     all_unc=all_var_nll_unc,
#     retained_percs=retained_percs,
# )
#
# retained_unc_res_varenll = evaluate_retained_unc(
#     all_outprob_mean=concat_ood_score(e_nll_id, e_nll_ood)[1],
#     all_hard_pred=all_hard_pred,
#     all_y_true=all_y_true,
#     all_unc=all_var_nll_unc,
#     retained_percs=retained_percs,
# )
#
# retained_unc_res = evaluate_retained_unc(
#     all_outprob_mean=concat_ood_score(e_nll_id, e_nll_ood)[1],
#     all_hard_pred=all_hard_pred,
#     all_y_true=all_y_true,
#     all_unc=all_exceed_unc,
#     retained_percs=retained_percs,
# )
#
# retained_unc_res = evaluate_random_retained_unc(
#     all_outprob_mean=all_outprob_mean,
#     all_hard_pred=all_hard_pred,
#     all_y_true=all_y_true,
#     repetition=50,
#     retained_percs=retained_percs,
# )
#
# retained_prob_unc_res_enll = evaluate_retained_unc(
#     all_outprob_mean=concat_ood_score(e_nll_id, e_nll_ood)[1],
#     all_hard_pred=all_hard_pred,
#     all_y_true=all_y_true,
#     all_unc=all_prob_unc,
#     retained_percs=retained_percs,
# )
#
# print(retained_prob_unc_res["f1_score"])
#
# # === EVALUATION : AUPRC-ERROR ===
# if n_ensemble > 1:
#     misclas_det_res = evaluate_misclas_detection(
#         all_y_true, all_hard_pred, all_prob_unc, return_boxplot=True
#     )
#     misclas_det_res = evaluate_misclas_detection(
#         all_y_true, all_hard_pred, all_var_nll_unc, return_boxplot=True
#     )
#     misclas_det_res = evaluate_misclas_detection(
#         all_y_true, all_hard_pred, all_exceed_unc, return_boxplot=True
#     )
#
# # ========================================
#
# # make it work for all AEs
#
# # convert to OOD probability based on selected cdf distribution
# nll_key = "nll"
#
#
# fit_per_bae_sample = True  # either fit on every BAE sample or on mean of BAE samples
# dist = gamma
# dist_ = []
# norm_scalings = []
# min_level = "mean"
# cdf_scaling = True
#
#
# bae_id_ref_pred = bae_model.predict(x_id_train, select_keys=[nll_key])
# bae_id_ref_pred = flatten_nll(bae_id_ref_pred["nll"])
#
# from statsmodels.distributions import ECDF
#
#
# class ECDF_Wrapper(ECDF):
#     def fit(self, x):
#         return super(ECDF_Wrapper, self).__init__(x)
#
#     def predict(self, x):
#         return self(x)
#
#
# class Outlier_CDF:
#     """
#     Class to get outlier probabilities by fitting distributions to outlier scores.
#
#     Available distributions are gamma, beta, lognorm, norm, uniform, exponential and ECDF.
#     Also normalisation scaling is available as option based on:
#      https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2011/SDM11-outlier-preprint.pdf
#     """
#
#     dist_dict = {
#         "gamma": gamma,
#         "beta": beta,
#         "lognorm": lognorm,
#         "norm": norm,
#         "uniform": uniform,
#         "expon": expon,
#         "ecdf": ECDF_Wrapper,
#     }
#
#     def __init__(self, dist_type="gamma", norm_scaling=None, norm_scaling_level="mean"):
#         if norm_scaling is None:
#             norm_scaling = False if dist_type == "uniform" else True
#         self.norm_scaling = norm_scaling
#         self.norm_scaling_level = norm_scaling_level
#         self.norm_scaling_param = None
#
#         self.dist_type = dist_type
#         self.dist_class = self.dist_dict[dist_type]
#
#     def fit(self, outlier_score):
#         # fit a selected distribution to outlier scores
#         if self.dist_type != "ecdf":
#             self.dist_ = self.dist_class(*self.dist_class.fit(outlier_score))
#         else:
#             self.dist_ = self.dist_class.fit(outlier_score)
#
#         # if norm_scaling is required
#         if self.norm_scaling:
#             self.fit_norm_scaler(self.predict(outlier_score, norm_scaling=False))
#
#         return self
#
#     def fit_norm_scaler(self, outlier_score):
#         self.norm_scaling_param = self.get_norm_scaling_level(
#             outlier_score, min_level=self.norm_scaling_level
#         )
#
#     def predict(self, outlier_score, norm_scaling=None):
#         if norm_scaling is None:
#             norm_scaling = self.norm_scaling
#
#         if self.dist_type != "ecdf":
#             outlier_proba = self.dist_.cdf(outlier_score)
#         else:
#             outlier_proba = self.dist_.predict(outlier_score)
#
#         # if norm_scaling is enabled
#         if norm_scaling:
#             if self.norm_scaling_param is None:
#                 raise ValueError(
#                     "Please fit the CDF first. Norm scaler is found to be None."
#                 )
#
#             outlier_proba = np.clip(
#                 (outlier_proba - self.norm_scaling_param)
#                 / (1 - self.norm_scaling_param),
#                 0,
#                 1,
#             )
#
#         return outlier_proba
#
#     def get_norm_scaling_level(self, nll, min_level="quartile"):
#         """
#         Return level of NLL scores for cdf scaling.
#         """
#         if min_level == "quartile":
#             return np.percentile(nll, 75)
#         elif min_level == "median":
#             return np.percentile(nll, 50)
#         elif min_level == "mean":
#             return np.mean(nll)
#         else:
#             raise NotImplemented
#
#
# class BAE_Outlier_Proba:
#     def __init__(
#         self,
#         dist_type="ecdf",
#         norm_scaling=None,
#         norm_scaling_level="mean",
#         fit_per_bae_sample=True,
#     ):
#         self.dist_type = dist_type
#         if norm_scaling is None:
#             norm_scaling = False if dist_type == "uniform" else True
#         self.norm_scaling = norm_scaling
#         self.norm_scaling_level = norm_scaling_level
#         self.fit_per_bae_sample = fit_per_bae_sample
#
#     def fit(self, bae_nll_samples, dist_type=None, norm_scaling=None):
#         # House keeping on handling default dist_type and default norm_scaling
#         # If None is supplied, resort to internal saved param
#         # Otherwise, overrides the internal saved param.
#
#         if dist_type is None:
#             dist_type = self.dist_type
#         else:
#             self.dist_type = dist_type
#         if norm_scaling is None:
#             norm_scaling = self.norm_scaling
#         else:
#             self.norm_scaling = norm_scaling
#         dist_ = []
#         # fit a cdf on every BAE model's prediction
#         # resulting in ensemble of cdfs
#         if fit_per_bae_sample:
#             for bae_pred_sample in bae_nll_samples:
#                 dist_.append(
#                     Outlier_CDF(
#                         dist_type=dist_type,
#                         norm_scaling=norm_scaling,
#                         norm_scaling_level=self.norm_scaling_level,
#                     ).fit(bae_pred_sample)
#                 )
#
#         # fit a cdf on the mean of BAE models' prediction
#         else:
#             dist_.append(
#                 Outlier_CDF(dist_type=dist_type, norm_scaling=norm_scaling).fit(
#                     bae_nll_samples.mean(0)
#                 )
#             )
#         self.dist_ = dist_
#
#     def predict_proba(self, bae_nll_samples):
#         # predictions
#         outlier_probas = np.array(
#             [dist_.predict(bae_pred_sample) for bae_pred_sample in bae_nll_samples]
#         )
#         return outlier_probas
#
#     def calc_ood_unc(self, prob):
#         unc = prob * (1 - prob)
#         epi = prob.var(0)
#         alea = unc.mean(0)
#         return {"epi": epi, "alea": alea, "total": epi + alea}
#
#     def predict(self, bae_nll_samples):
#         """
#         Given BAE NLL samples, return the OOD proba mean and proba uncertainty.
#         """
#         proba_samples = self.predict_proba(bae_nll_samples)
#         ood_proba_mean = proba_samples.mean(0)
#         ood_proba_unc = self.calc_ood_unc(
#             proba_samples
#         )  # dict with keys of epi/alea/total
#         return ood_proba_mean, ood_proba_unc
#

# ood_prob_mean = outlier_probas.mean(0)
# ood_prob_unc = calc_ood_unc(outlier_probas)


# min_scale = np.clip(
#     (cdf_score - dist_.cdf(np.percentile(bae_sample_train_i, 75)))
#     / (1 - dist_.cdf(np.percentile(bae_sample_train_i, 75))),
#     0,
#     1,
# )
