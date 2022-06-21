from scipy.io import arff
import pandas as pd
import os


def load_benchmark_dt(
    base_folder="od_benchmark2", filename="Annthyroid_withoutdupl_norm_07.arff"
):
    data = arff.loadarff(os.path.join(base_folder, filename))
    df = pd.DataFrame(data[0])

    df.loc[df["outlier"] == b"no", "outlier"] = 0
    df.loc[df["outlier"] == b"yes", "outlier"] = 1

    X = df.drop(["id", "outlier"], axis=1).values
    y = df["outlier"].values

    return X, y
