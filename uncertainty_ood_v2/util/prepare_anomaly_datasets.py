import os

from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from typing import Union
from sklearn.preprocessing import MinMaxScaler


def list_anomaly_dataset(base_folder="anomaly_datasets"):
    """
    Returns list of files in folder containing the anomaly datasets.
    This should be used as accompanying the load_anomaly_dataset method.
    """
    mat_file_list = os.listdir(base_folder)
    return mat_file_list


def load_anomaly_dataset(
    mat_file_id: Union[int, str] = 0,
    base_folder="anomaly_datasets",
    train_size=0.8,
    valid_size=0.0,
    random_seed=123,
    minmax_scaling=True,
    minmax_clip=True,
):
    """
    Loads the anomaly detection dataset from base folder.
    Provide the mat_file_id (e.g 0-10) to load the .mat file as indexed when listdir is called on base_folder.
    Otherwise, directly provide the .mat file name to load it.
    Parameters include training_size ratio, random seed for reproducibility of splitting, and option for min-max scaling.

    """

    # prepare and load data path
    if isinstance(mat_file_id, int):
        mat_file_list = list_anomaly_dataset(base_folder)
        mat_file = mat_file_list[mat_file_id]
    elif isinstance(mat_file_id, str):
        mat_file = mat_file_id
    mat = loadmat(os.path.join(base_folder, mat_file))

    # load X and Y
    X = mat["X"]
    y = mat["y"].ravel()

    # get outliers and inliers
    x_ood, x_id = get_outliers_inliers(X, y)

    # update exp params
    # exp_params.update({"dataset": mat_file.split("_")[0] if "_" else mat_file in mat_file})

    # split in-distribution to train and test
    x_id_train, x_id_test = train_test_split(
        x_id, train_size=train_size, shuffle=True, random_state=random_seed
    )

    # check if validation set is required
    if valid_size > 0:
        x_id_train, x_id_valid = train_test_split(
            x_id_train, test_size=valid_size, shuffle=True, random_state=random_seed
        )

    # apply scaling
    if minmax_scaling:
        preproc_scaler = MinMaxScaler(clip=minmax_clip).fit(x_id_train)
        x_id_train = preproc_scaler.transform(x_id_train)
        x_id_test = preproc_scaler.transform(x_id_test)
        x_ood = preproc_scaler.transform(x_ood)

        # scale validation set
        if valid_size > 0:
            x_id_valid = preproc_scaler.transform(x_id_valid)

    # handle returning validation set
    if valid_size > 0:
        return x_id_train, x_id_test, x_ood, x_id_valid
    else:
        return x_id_train, x_id_test, x_ood
