import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

window_size = 10
dt = pickle.load(open("loss.p","rb"))
dt = pd.DataFrame([dt]).T
rolling_mean = dt.rolling(window=window_size).mean().values.flatten()[window_size:]
rolling_std = dt.rolling(window=window_size).std().values.flatten()[window_size:]
x_axis = np.arange(0,len(rolling_mean))+window_size

plt.figure()
plt.plot(x_axis, rolling_mean)
# plt.fill_between(x_axis, rolling_mean-2*rolling_std,
#                  rolling_mean+2*rolling_std, alpha=0.45
#                  )

diffs = np.diff(rolling_mean)

plt.figure()
plt.plot(x_axis, rolling_mean/rolling_mean[0])

# abs diff
rolling_mean_norm = rolling_mean/rolling_mean[0]
threshold = 0.01
checkpoints = []
min_number = 100
for i in range(len(rolling_mean_norm)):
    if i > min_number+1:
        prev = i-min_number
        # abs_diff = np.abs((rolling_mean_norm[i] - rolling_mean_norm[prev])/rolling_mean_norm[prev])*100
        # abs_diff = np.abs((rolling_mean_norm[i] - rolling_mean_norm[prev])) * 100
        abs_diff = (rolling_mean_norm[prev]-rolling_mean_norm[i] ) * 100

        print(abs_diff)
        if abs_diff >= threshold:
            checkpoints.append(i)
checkpoints = np.array(checkpoints)

plt.figure()
plt.plot(x_axis,rolling_mean_norm)
plt.scatter(checkpoints, checkpoints*0)

# rolling slow and fast
fast_window = 10
slow_window = fast_window*10
n_stop_points = 5

rolling_mean_slow = dt.rolling(window=slow_window).mean().values.flatten()[slow_window:]
rolling_mean_fast = dt.rolling(window=fast_window).mean().values.flatten()[fast_window:]
x_axis_1 = np.arange(0,len(rolling_mean_slow))+slow_window
x_axis_2 = np.arange(0,len(rolling_mean_fast))+fast_window
crossover_points = np.argwhere(rolling_mean_fast[(slow_window-fast_window):]>rolling_mean_slow)[:,0]+(slow_window)
convergence_point = crossover_points[n_stop_points]

plt.figure()
slow_plot, = plt.plot(x_axis_1,rolling_mean_slow)
fast_plot, = plt.plot(x_axis_2,rolling_mean_fast)
crossover_plot = plt.scatter(crossover_points,crossover_points*0+np.min(rolling_mean_slow), marker='x', s=15)
cvg_plot = plt.vlines(convergence_point, np.min(rolling_mean_fast),np.max(rolling_mean_fast), color="tab:red")
plt.axvspan(convergence_point, x_axis_2[-1], color='tab:red', alpha=0.35)

plt.legend([slow_plot,fast_plot, crossover_plot, cvg_plot],["Slow window ("+str(slow_window)+")",
                                            "Fast window ("+str(fast_window)+")",
                                            "Crossover", "Convergence"])

plt.xlabel("Epochs")
plt.ylabel("Loss")


plt.title("Convergence epoch: "+str(convergence_point))

def detect_convergence(losses=np.array([]), fast_window = 10, slow_window = 100, n_stop_points=5):
    """
    Use two windows (fast and slow) to confirm convergence on losses has been achieved.
    As long as slow > fast , assume a downward trend is still ongoing.
    As soon as slow < fast (crossover),  assume convergence is imminent.
    To be even more sure, take the convergence epoch as the epoch after `n_stop_points` has been accumulated.

    Returns
    -------
    convergence_epoch : int
        Epoch where convergence has been confirmed. Returns 0 if convergence has not been reached


    """
    # dt = pickle.load(open("loss.p", "rb"))
    dt = pd.DataFrame([losses]).T
    if len(dt) >= slow_window :
        rolling_mean_slow = dt.rolling(window=slow_window).mean().values.flatten()[slow_window:]
        rolling_mean_fast = dt.rolling(window=fast_window).mean().values.flatten()[fast_window:]
        x_axis_1 = np.arange(0,len(rolling_mean_slow))+slow_window
        x_axis_2 = np.arange(0,len(rolling_mean_fast))+fast_window
        crossover_points = np.argwhere(rolling_mean_fast[(slow_window-fast_window):]>rolling_mean_slow)[:,0]+(slow_window)

        if len(crossover_points) >= n_stop_points:
            convergence_epoch = crossover_points[n_stop_points]
            return convergence_epoch
        else:
            return 0
    else:
        return 0

def plot_convergence(losses=np.array([]), fast_window = 10, slow_window = 100, n_stop_points=5, ax=None):
    """
    Use two windows (fast and slow) to confirm convergence on losses has been achieved.
    As long as slow > fast , assume a downward trend is still ongoing.
    As soon as slow < fast (crossover),  assume convergence is imminent.
    To be even more sure, take the convergence epoch as the epoch after `n_stop_points` has been accumulated.

    Returns
    -------
    convergence_epoch : int
        Epoch where convergence has been confirmed. Returns 0 if convergence has not been reached
    """
    dt = pd.DataFrame([losses]).T

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if len(dt) >= slow_window :
        rolling_mean_slow = dt.rolling(window=slow_window).mean().values.flatten()[slow_window:]
        rolling_mean_fast = dt.rolling(window=fast_window).mean().values.flatten()[fast_window:]
        x_axis_1 = np.arange(0,len(rolling_mean_slow))+slow_window
        x_axis_2 = np.arange(0,len(rolling_mean_fast))+fast_window
        crossover_points = np.argwhere(rolling_mean_fast[(slow_window-fast_window):]>rolling_mean_slow)[:,0]+(slow_window)

        if len(crossover_points) >= n_stop_points:
            convergence_epoch = crossover_points[n_stop_points]
        else:
            convergence_epoch = 0

        slow_plot, = ax.plot(x_axis_1,rolling_mean_slow)
        fast_plot, = ax.plot(x_axis_2,rolling_mean_fast)
        crossover_plot = ax.scatter(crossover_points,crossover_points*0+np.min(rolling_mean_slow), marker='x', s=15)

        if convergence_epoch > 0:
            cvg_plot = ax.vlines(convergence_epoch, np.min(rolling_mean_fast),np.max(rolling_mean_fast), color="tab:red")
            ax.axvspan(convergence_epoch, x_axis_2[-1], color='tab:red', alpha=0.35)

            plt.legend([slow_plot,fast_plot, crossover_plot, cvg_plot],["Slow window ("+str(slow_window)+")",
                                                        "Fast window ("+str(fast_window)+")",
                                                        "Crossover", "Convergence"])
        else:
            if len(crossover_points) > 0:
                plt.legend([slow_plot, fast_plot, crossover_plot], ["Slow window (" + str(slow_window) + ")",
                                                                              "Fast window (" + str(fast_window) + ")",
                                                                              "Crossover"])
            else:
                plt.legend([slow_plot, fast_plot], ["Slow window (" + str(slow_window) + ")",
                                                                    "Fast window (" + str(fast_window) + ")",
                                                                    ])

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Convergence epoch: "+str(convergence_epoch))
    return ax

losses = pickle.load(open("loss.p","rb"))
losses = losses[:90]
convergence = detect_convergence(losses=losses, fast_window = 10, slow_window = 100, n_stop_points=5)
plot_convergence(losses=losses, fast_window = 10, slow_window = 100, n_stop_points=5)