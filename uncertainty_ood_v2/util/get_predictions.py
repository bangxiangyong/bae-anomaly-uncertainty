from baetorch.baetorch.models_v2.base_layer import flatten_np


def calc_e_nll(bae_pred):
    key = "nll"
    return flatten_np(bae_pred[key].mean(0)).mean(-1)


def calc_var_nll(bae_pred):
    key = "nll"
    return flatten_np(bae_pred[key].var(0)).mean(-1)


def flatten_nll(bae_nll):
    """
    Flattens NLL scores from BAE until there are only 2 dimensions : (BAE samples , total examples).
    """
    while len(bae_nll.shape) > 2:
        bae_nll = bae_nll.mean(-1)
    return bae_nll
