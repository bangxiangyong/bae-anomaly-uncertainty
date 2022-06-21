from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from baetorch.baetorch.util.seed import bae_set_seed
from util.generate_data import generate_moons

bae_set_seed(123)

#generate random data with two features
# X_train, Y_train = generate_data(n_train=200,train_only=True, n_features=2)
X_train, Y_train, X_test, Y_test = generate_moons(train_only=False,
                                                  n_samples=500,
                                                  test_size=0.5,
                                                  outlier_class = 1
                                                  )

# by default the outlier fraction is 0.1 in generate data function
outlier_fraction = 0.01

# store outliers and inliers in different numpy arrays
x_outliers, x_inliers = get_outliers_inliers(X_train,Y_train)
X_train = x_inliers
#============================

# clf = ABOD()
# clf = OCSVM()
clf = IForest()
# clf = KNN()
clf.fit(X_train)

#=========================rescratch=============
def generate_grid2d(X_train, span=1):
    grid = np.mgrid[X_train[:, 0].min() - span:X_train[:, 0].max() + span:100j,
           X_train[:, 1].min() - span:X_train[:, 1].max() + span:100j]
    grid_2d = grid.reshape(2, -1).T
    return grid_2d, grid

# plot figure
def plot_decision_boundary(x_inliers, x_outliers, grid_2d, Z, anomaly_threshold=None, ax=None, figsize=(6,4)):
    grid = grid_2d.T.reshape(2,100,100)
    reshaped_Z = Z.reshape(100, 100)

    if ax is None:
        fig, ax = plt.subplots(1,1)

    contour = plt.contourf(grid[0], grid[1], reshaped_Z, levels=35, cmap="Greys")
    plt.colorbar(contour)

    # plot decision boundary
    if anomaly_threshold is not None:
        a = ax.contour(grid[0], grid[1], reshaped_Z, levels=[anomaly_threshold], linewidths=1.5, colors='red')
        ax.contourf(grid[0], grid[1], reshaped_Z, levels=[Z.min(), anomaly_threshold], colors='tab:blue',alpha=0.5)

    # scatter plot of inliers with white dots
    b = ax.scatter(x_inliers[:, 0], x_inliers[:, 1], c='tab:green', s=20, edgecolor='k')
    # scatter plot of outliers with black dots
    c = ax.scatter(x_outliers[:, 0], x_outliers[:, 1], c='tab:orange', s=20, edgecolor='k')

    if anomaly_threshold is not None:
        plt.legend(
            [a.collections[0], b, c],
            ['Decision Boundary', 'Train (Inliers)', 'Train (Outliers)'],)
    else:
        plt.legend(
            [ b, c],
            [ 'Train (Inliers)', 'Train (Outliers)'],)

def get_anomaly_threshold(raw_train_scores, percentile = 95):
    # higher means more anomalous
    anomaly_threshold = stats.scoreatpercentile(raw_train_scores, percentile)
    return anomaly_threshold

def get_hard_predictions(raw_train_scores, anomaly_threshold):
    hard_pred = np.zeros(len(raw_train_scores))
    hard_pred[np.argwhere(raw_train_scores >= anomaly_threshold)] = 1
    return hard_pred

# get raw train scores
raw_train_scores = clf.decision_function(X_train)
raw_train_scores = np.exp(raw_train_scores)

# get threshold
anomaly_threshold = stats.scoreatpercentile(raw_train_scores,100-(100*outlier_fraction))

# apply threshold to get hard predictions
hard_pred = get_hard_predictions(raw_train_scores, anomaly_threshold)

# visualise grid
grid_2d, grid = generate_grid2d(np.concatenate((x_outliers, x_inliers)), span=0.5)
y_pred_grid = clf.decision_function(grid_2d)
y_pred_grid = np.exp(y_pred_grid)

plot_decision_boundary(x_inliers=x_inliers,
                       x_outliers=x_outliers,
                       grid_2d=grid_2d,
                       Z=y_pred_grid,
                       anomaly_threshold=anomaly_threshold)

plot_decision_boundary(x_inliers=x_inliers,
                       x_outliers=x_outliers,
                       grid_2d=grid_2d,
                       Z=y_pred_grid)

