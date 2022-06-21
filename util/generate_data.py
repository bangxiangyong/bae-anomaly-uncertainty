from pyod.utils.data import get_outliers_inliers
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# X_train, Y_train, X_test, Y_test = generate_data(n_train=200,train_only=False, n_features=2)
# n_samples = 500
# X, Y  = datasets.make_circles(n_samples=n_samples, factor=.5,
#                                       noise=.05)
#
# X,Y = datasets.make_moons(n_samples=n_samples, noise=.05)
# blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

# Anisotropicly distributed data
# random_state = 170
# X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
# transformation = [[0.6, -0.6], [-0.4, 0.8]]
# X_aniso = np.dot(X, transformation)
# aniso = (X_aniso, y)


def generate_aniso_(n_samples):
    X, y = datasets.make_blobs(n_samples=n_samples, centers=3)
    print(X.shape)
    transformation = [[0.6, -0.3], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    return X_aniso, y


def generate_moons(train_only=True, n_samples=500, test_size=0.5, outlier_class=0):

    X, Y = datasets.make_moons(n_samples=n_samples, noise=0.05)
    # X, Y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    # X, Y = generate_aniso(n_samples)
    outlier_args = np.argwhere(Y == outlier_class)[:, 0]
    inlier_args = np.argwhere(Y != outlier_class)[:, 0]

    Y[outlier_args] = 1
    Y[inlier_args] = 0

    if train_only:
        return X, Y
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        return X_train, Y_train, X_test, Y_test


def generate_circles(train_only=True, n_samples=500, test_size=0.5, outlier_class=0):

    # X, Y = datasets.make_moons(n_samples=n_samples, noise=.05)
    X, Y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    # X, Y = generate_aniso(n_samples)
    outlier_args = np.argwhere(Y == outlier_class)[:, 0]
    inlier_args = np.argwhere(Y != outlier_class)[:, 0]

    Y[outlier_args] = 1
    Y[inlier_args] = 0

    if train_only:
        return X, Y
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        return X_train, Y_train, X_test, Y_test


def generate_aniso(train_only=True, n_samples=500, test_size=0.5, outlier_class=0):

    X, Y = generate_aniso_(n_samples)
    outlier_args = np.argwhere(Y == outlier_class)[:, 0]
    inlier_args = np.argwhere(Y != outlier_class)[:, 0]

    Y[outlier_args] = 1
    Y[inlier_args] = 0

    if train_only:
        return X, Y
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        return X_train, Y_train, X_test, Y_test


def generate_blobs(
    train_only=True, n_samples=500, test_size=0.5, outlier_class=0, sparse=False
):

    blobs_params = dict(n_samples=n_samples, n_features=2)
    if not sparse:
        X, Y = make_blobs(
            centers=[[5, 5], [-5, -5], [0, 0]],
            cluster_std=[0.5, 0.5, 1],
            **blobs_params
        )
    else:
        X, Y = make_blobs(
            centers=[[2, 2], [-2, -2]], cluster_std=[1.5, 0.3], **blobs_params
        )

    outlier_args = np.argwhere(Y == outlier_class)[:, 0]
    inlier_args = np.argwhere(Y != outlier_class)[:, 0]

    Y[outlier_args] = 1
    Y[inlier_args] = 0

    if train_only:
        return X, Y
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        return X_train, Y_train, X_test, Y_test


# X_train, Y_train, X_test, Y_test = generate_moons(train_only=False, n_samples=500, test_size=0.5)


# plt.figure()
# plt.scatter(x_inliers_train[:,0], x_inliers_train[:,1])
# plt.scatter(x_outliers_train[:,0], x_outliers_train[:,1])
# plt.legend(["INLIER", "OUTLIER"])
