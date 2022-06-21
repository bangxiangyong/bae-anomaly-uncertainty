import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# res_folder = "results/latent"

ll_res_folder = "results/likelihood"
ll_group_keys = ["ss_id", "bae_type", "full_likelihood", "skip"]
predrop_sghmc5 = True
# === BASELINE AUROC ===
# auroc results
auroc_res = pd.concat(
    [
        pd.read_csv(os.path.join(ll_res_folder, csv))
        # for csv in ["STRATH_FORGE_AUROC.csv", "STRATH_FORGE_TGAUSS_AUROC.csv","STRATH_FORGE_NOSKIPALL_AUROC.csv"]
        for csv in ["STRATH_FORGE_AUROC.csv", "STRATH_FORGE_NOSKIPALL_AUROC.csv"]
    ]
)

# drop SGHMC-5 and include the SGHMC-100
if predrop_sghmc5:
    auroc_res = auroc_res.drop(
        index=auroc_res[auroc_res["bae_type"] == "sghmc"].index
    ).reset_index()
    auroc_res = pd.concat(
        [
            auroc_res,
            pd.read_csv(os.path.join(ll_res_folder, "STRATH_FORGE_100SGHMC_AUROC.csv")),
        ]
    )

# combine skip column
auroc_res_mean = auroc_res.groupby(ll_group_keys).mean().reset_index()
auroc_res_std = auroc_res.groupby(ll_group_keys).std().reset_index()

# filter skip
auroc_res_mean["model_name"] = (
    auroc_res_mean["bae_type"]
    + "-"
    + auroc_res_mean["full_likelihood"]
    + auroc_res_mean["ss_id"]
)
auroc_res_mean_noskip = auroc_res_mean[auroc_res_mean["skip"] == False]
auroc_res_mean_skip = auroc_res_mean[auroc_res_mean["skip"] == True]

auroc_res_mean_noskip["E_AUROC-skip"] = list(auroc_res_mean_skip["E_AUROC"])
auroc_res_mean_noskip["V_AUROC-skip"] = list(auroc_res_mean_skip["V_AUROC"])
auroc_res_mean_noskip["E_AUROC-noskip"] = list(auroc_res_mean_noskip["E_AUROC"])
auroc_res_mean_noskip["V_AUROC-noskip"] = list(auroc_res_mean_noskip["V_AUROC"])
auroc_res_mean_noskip = auroc_res_mean_noskip.reset_index(drop=True)
# === FINAL TABLE OF RESULTS ===

# convert to LATEX table row
bae_type_map = {
    "ae": "Det. AE",
    "ens": "BAE, Ensemble",
    "mcd": "BAE, MC-Dropout",
    "vi": "BAE, BayesBB",
    "sghmc": "BAE, SGHMC",
    "vae": "VAE",
}

ll_map = {
    "bernoulli": r"Ber($\hat{x}^*$)",
    "cbernoulli": r"C-Ber($\hat{x}^*$)",
    "mse": r"N($\hat{x}^*$,1)",
    "hetero-gauss": r"N($\hat{x}^*$,$\sigma_i^2$)",
}

# Table 1
table_keys = [
    "bae_type",
    "full_likelihood",
    "E_AUROC-noskip",
    "V_AUROC-noskip",
    "E_AUROC-skip",
    "V_AUROC-skip",
]


def latex_bold_val(x, bold_threshold=0.8):
    formatted = "{:.1f}".format(x * 100)
    if x >= bold_threshold:
        formatted = "\\textbf{" + formatted + "}"
    return formatted


bold_threshold = 0.8

for ss_id in auroc_res_mean_noskip["ss_id"].unique():
    auroc_sensor_mean = auroc_res_mean_noskip[auroc_res_mean_noskip["ss_id"] == ss_id][
        table_keys
    ]

    # rearrange rows
    key_orders = [
        ["ae", "vae", "mcd", "vi", "ens", "sghmc"],
        ["bernoulli", "cbernoulli", "mse", "hetero-gauss"],
    ]

    reorder_auroc_mean = []
    for bae_type in key_orders[0]:
        for full_likelihood in key_orders[1]:
            reorder_auroc_mean.append(
                auroc_sensor_mean[
                    (auroc_sensor_mean["bae_type"] == bae_type)
                    & (auroc_sensor_mean["full_likelihood"] == full_likelihood)
                ]
            )
    reorder_auroc_mean = pd.concat(reorder_auroc_mean).reset_index()

    # write to file
    latex_tb_f = open("latex-table-ssid-" + ss_id + "-skip.txt", "w")
    all_lines = ""
    for i, row in reorder_auroc_mean.round(3).iterrows():
        e_auroc_noskip = latex_bold_val(
            row["E_AUROC-noskip"], bold_threshold=bold_threshold
        )
        v_auroc_noskip = latex_bold_val(
            row["V_AUROC-noskip"], bold_threshold=bold_threshold
        )
        e_auroc_skip = latex_bold_val(
            row["E_AUROC-skip"], bold_threshold=bold_threshold
        )
        v_auroc_skip = latex_bold_val(
            row["V_AUROC-skip"], bold_threshold=bold_threshold
        )

        if row["bae_type"] == "ae":
            v_auroc_noskip = "-"
            v_auroc_skip = "-"

        newline = (
            " & ".join(
                (
                    bae_type_map[row["bae_type"]],
                    ll_map[row["full_likelihood"]],
                    e_auroc_noskip,
                    v_auroc_noskip,
                    e_auroc_skip,
                    v_auroc_skip,
                )
            )
            + " \\\\"
        )
        # add a mid rule after each ae model rows
        if (i + 1) % len(key_orders[1]) == 0:
            newline += (
                " \\midrule" if i != (len(reorder_auroc_mean) - 1) else " \\bottomrule"
            )
        newline += " \n"
        all_lines += newline
    latex_tb_f.write(all_lines)
    latex_tb_f.close()


# bce vs se results
bce_v_se_res = pd.read_csv(os.path.join(ll_res_folder, "STRATH_FORGE_BCE_VS_SE.csv"))

bce_v_se_res_mean = bce_v_se_res.groupby(ll_group_keys).mean()
bce_v_se_res_std = bce_v_se_res.groupby(ll_group_keys).std()

# best results
best_res = (
    auroc_res.groupby(["full_likelihood", "ss_id", "skip", "bae_type"])
    .mean()
    .sort_values(["E_AUROC"])
)
