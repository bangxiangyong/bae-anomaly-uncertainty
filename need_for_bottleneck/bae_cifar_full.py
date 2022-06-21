import itertools

import numpy as np
from tqdm import tqdm

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from need_for_bottleneck.prepare_data_cifar import get_id_set, get_ood_set
from strathclyde_analysis_v2.evaluate_outlier_uncertainty import evaluate_ood_unc
from util.exp_manager import ExperimentManager

bae_set_seed(100)

# exp name and filenames
exp_name = "CIFAR_"
auroc_filename = exp_name + "AUROC.csv"
bce_se_filename = exp_name + "BCE_VS_SE.csv"
retained_perf_filename = exp_name + "retained_perf.csv"
misclas_perf_filename = exp_name + "misclas_perf.csv"


# Loop over all grid search combinations
# fmt: off
n_random_seeds = 3
random_seeds = np.random.randint(0, 1000, n_random_seeds)
full_likelihood = ["mse", "homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss", "bernoulli", "cbernoulli",
                   "beta"]
homoscedestic_mode_map = {"bernoulli": "none", "cbernoulli": "none", "homo-gauss": "every", "hetero-gauss": "none",
                          "homo-tgauss": "none", "hetero-tgauss": "none", "mse": "none", "beta": "none"}
likelihood_map = {"bernoulli": "bernoulli", "cbernoulli": "cbernoulli", "homo-gauss": "gaussian",
                  "hetero-gauss": "gaussian", "homo-tgauss": "truncated_gaussian",
                  "hetero-tgauss": "truncated_gaussian", "mse": "gaussian", "beta": "beta"}
twin_output_map = {"bernoulli": False, "cbernoulli": False, "homo-gauss": False, "hetero-gauss": True,
                   "homo-tgauss": False, "hetero-tgauss": True, "mse": False, "beta": True}
# fmt: on

bae_type_classes = {
    "ens": BAE_Ensemble,
    "mcd": BAE_MCDropout,
    "sghmc": BAE_SGHMC,
    "vi": BAE_VI,
    "vae": VAE,
    "ae": BAE_Ensemble,
}

n_bae_samples_map = {
    "ens": 5,
    "mcd": 100,
    "sghmc": 5,
    "vi": 100,
    "vae": 100,
    "ae": 1,
}

# SINGLE SAMPLE
grid = {
    "random_seed": random_seeds,
    "id_dataset": ["FashionMNIST"],
    "skip": [True],
    "latent_factor": [2.0],
    "bae_type": ["ae"],
    "full_likelihood": ["mse"],
    "eval_ood_unc": [False],
}

# # BOTTLENECK
# grid = {
#     "random_seed": random_seeds,
#     "id_dataset": ["FashionMNIST","CIFAR","SVHN","MNIST"],
#     "skip": [True,False],
#     "latent_factor": [0.01,0.1,0.5,1.0,2.0],
#     "bae_type": ["ae","ens","vae"],
#     "full_likelihood": ["mse"],
#     "eval_ood_unc":[False]
# }
#
# # LL
# grid = {
#     "random_seed": random_seeds,
#     "id_dataset": ["FashionMNIST","CIFAR","SVHN","MNIST"],
#     "skip": [True],
#     "latent_factor": [1.0],
#     "bae_type": ["ae","ens","vae","vi","mcd","sghmc"],
#     "full_likelihood": ["mse","hetero-gauss","bernoulli","cbernoulli"],
#     "eval_ood_unc":[False]
# }

# fmt: off
id_n_channels = {"CIFAR": 3, "SVHN": 3, "FashionMNIST": 1, "MNIST": 1}
flattened_dims = {"CIFAR": 3*32*32, "SVHN": 3*32*32, "FashionMNIST": 28*28, "MNIST": 28*28}
input_dims = {"CIFAR": 32, "SVHN": 32, "FashionMNIST": 28, "MNIST": 28}
# fmt: on

for values in tqdm(itertools.product(*grid.values())):
    # setup the grid
    exp_params = dict(zip(grid.keys(), values))
    print(exp_params)

    # unpack exp params
    random_seed = exp_params["random_seed"]
    id_dataset = exp_params["id_dataset"]
    skip = exp_params["skip"]
    latent_factor = exp_params["latent_factor"]
    bae_type = exp_params["bae_type"]
    full_likelihood_i = exp_params["full_likelihood"]

    # whether to evaluate OOD uncertainty
    if full_likelihood == "mse":
        exp_params.update({"eval_ood_unc": True})
    else:
        exp_params.update({"eval_ood_unc": False})

    eval_ood_unc = exp_params["eval_ood_unc"]
    if eval_ood_unc:
        pickle_files = [
            auroc_filename,
            bce_se_filename,
            retained_perf_filename,
            misclas_perf_filename,
        ]
    else:
        pickle_files = [auroc_filename, bce_se_filename]

    twin_output = twin_output_map[full_likelihood_i]
    homoscedestic_mode = homoscedestic_mode_map[full_likelihood_i]
    likelihood = likelihood_map[full_likelihood_i]
    n_bae_samples = n_bae_samples_map[bae_type]

    # === PREPARE DATA ===
    train_loader, test_loader = get_id_set(
        id_dataset=id_dataset, n_channels=id_n_channels[id_dataset]
    )

    # ===============FIT BAE===============
    use_cuda = True
    weight_decay = 0.00000000001
    anchored = True if bae_type == "ens" else False
    bias = False
    se_block = False
    norm = "layer"
    self_att = False
    self_att_transpose_only = False
    num_epochs = 20
    activation = "leakyrelu"
    lr = 0.01

    if id_dataset == "CIFAR" or id_dataset == "SVHN":
        input_dim = list([32, 32])
        input_channel = 3
    else:
        input_dim = list([28, 28])
        input_channel = 1

    latent_dim = int(
        (input_dims[id_dataset] ** 2) * id_n_channels[id_dataset] * latent_factor
    )
    chain_params = [
        {
            "base": "conv2d",
            "input_dim": input_dim,
            "conv_channels": [input_channel, 10, 32],
            "conv_stride": [2, 1],
            "conv_kernel": [2, 2],
            "activation": activation,
            "norm": norm,
            "order": ["base", "norm", "activation"],
            "bias": bias,
            # "dropout": dropout,
            "last_norm": norm,
        },
        {
            "base": "linear",
            "architecture": [100, latent_dim],
            "activation": activation,
            "norm": norm,
            "last_norm": norm,
        },
    ]

    bae_model = bae_type_classes[bae_type](
        chain_params=chain_params,
        last_activation="sigmoid",
        last_norm=norm,
        twin_output=twin_output,
        twin_params={"activation": "selu", "norm": "none"},
        skip=skip,
        use_cuda=use_cuda,
        scaler_enabled=False,
        homoscedestic_mode=homoscedestic_mode,
        likelihood=likelihood,
        weight_decay=weight_decay,
        num_samples=n_bae_samples,
        anchored=anchored,
        learning_rate=lr,
        stochastic_seed=random_seed,
    )

    min_lr, max_lr, half_iter = run_auto_lr_range_v4(
        train_loader,
        bae_model,
        window_size=1,
        num_epochs=10,
        run_full=False,
        plot=False,
        verbose=False,
        save_mecha="copy" if bae_type == "vae" else "file",
    )

    if isinstance(bae_model, BAE_SGHMC):
        bae_model.fit(
            train_loader,
            burn_epoch=int(num_epochs * 2 / 3),
            sghmc_epoch=num_epochs // 3,
            clear_sghmc_params=True,
        )
    else:
        time_method(bae_model.fit, train_loader, num_epochs=num_epochs)

    # === PREDICTIONS ===
    exp_man = ExperimentManager(folder_name="experiments")

    ood_datasets = [
        dt for dt in ["SVHN", "CIFAR", "FashionMNIST", "MNIST"] if dt != id_dataset
    ]
    # ood_datasets = [dt for dt in ["MNIST"] if dt != id_dataset]
    for ood_dataset in ood_datasets:
        exp_params_temp = exp_params.copy()
        exp_params_temp.update({"ood_dataset": ood_dataset})
        ood_loader = get_ood_set(
            ood_dataset=ood_dataset,
            n_channels=id_n_channels[id_dataset],
            resize=[input_dims[id_dataset]] * 2,
        )

        # predict and evaluate
        (bae_id_pred, bae_ood_pred), (
            (eval_auroc, retained_res_all, misclas_res_all)
        ) = evaluate_ood_unc(
            bae_model=bae_model,
            x_id_train=train_loader,
            x_id_test=test_loader,
            x_ood_test=ood_loader,
            exp_name=exp_name,
            exp_params=exp_params_temp,
            eval_ood_unc=eval_ood_unc,
            exp_man=exp_man,
            ret_flatten_nll=False,
            cdf_dists=["norm", "uniform", "ecdf", "expon"],
        )
        print(ood_dataset)
        print(eval_auroc)
