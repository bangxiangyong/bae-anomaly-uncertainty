import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def detect_convergence(
    losses=np.array([]), fast_window=10, slow_window=100, n_stop_points=5
):
    """
    Use two windows (fast and slow) to confirm convergence on losses has been achieved.
    As long as slow > fast , assume a downward trend is still ongoing.
    As soon as slow < fast (crossover),  assume convergence is imminent.
    To be even more sure, take the convergence epoch as the epoch after `n_stop_points` consecutive crossovers has been reached.

    Returns
    -------
    convergence_epoch : int
        Epoch where convergence has been confirmed. Returns 0 if convergence has not been reached
    """

    dt = pd.DataFrame([losses]).T
    if len(dt) >= slow_window:
        rolling_mean_slow = (
            dt.rolling(window=slow_window).mean().values.flatten()[slow_window:]
        )
        rolling_mean_fast = (
            dt.rolling(window=fast_window).mean().values.flatten()[fast_window:]
        )
        crossover_points = np.argwhere(
            rolling_mean_fast[(slow_window - fast_window) :] > rolling_mean_slow
        )[:, 0] + (slow_window)

        if len(crossover_points) > n_stop_points:
            convergence_epoch = crossover_points[n_stop_points]
            return convergence_epoch
        else:
            return 0
    else:
        return 0


def plot_convergence(
    losses=np.array([]), fast_window=10, slow_window=100, n_stop_points=5, ax=None
):
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

    if len(dt) >= slow_window:
        rolling_mean_slow = (
            dt.rolling(window=slow_window).mean().values.flatten()[slow_window:]
        )
        rolling_mean_fast = (
            dt.rolling(window=fast_window).mean().values.flatten()[fast_window:]
        )
        x_axis_1 = np.arange(0, len(rolling_mean_slow)) + slow_window
        x_axis_2 = np.arange(0, len(rolling_mean_fast)) + fast_window
        crossover_points = np.argwhere(
            rolling_mean_fast[(slow_window - fast_window) :] > rolling_mean_slow
        )[:, 0] + (slow_window)

        if len(crossover_points) > n_stop_points:
            convergence_epoch = crossover_points[n_stop_points]
        else:
            convergence_epoch = 0

        (slow_plot,) = ax.plot(x_axis_1, rolling_mean_slow)
        (fast_plot,) = ax.plot(x_axis_2, rolling_mean_fast)
        crossover_plot = ax.scatter(
            crossover_points,
            crossover_points * 0 + np.min(rolling_mean_slow),
            marker="x",
            s=15,
        )

        if convergence_epoch > 0:
            cvg_plot = ax.vlines(
                convergence_epoch,
                np.min(rolling_mean_fast),
                np.max(rolling_mean_fast),
                color="tab:red",
            )
            ax.axvspan(convergence_epoch, x_axis_2[-1], color="tab:red", alpha=0.35)

            plt.legend(
                [slow_plot, fast_plot, crossover_plot, cvg_plot],
                [
                    "Slow window (" + str(slow_window) + ")",
                    "Fast window (" + str(fast_window) + ")",
                    "Crossover",
                    "Convergence",
                ],
            )
        else:
            if len(crossover_points) > 0:
                plt.legend(
                    [slow_plot, fast_plot, crossover_plot],
                    [
                        "Slow window (" + str(slow_window) + ")",
                        "Fast window (" + str(fast_window) + ")",
                        "Crossover",
                    ],
                )
            else:
                plt.legend(
                    [slow_plot, fast_plot],
                    [
                        "Slow window (" + str(slow_window) + ")",
                        "Fast window (" + str(fast_window) + ")",
                    ],
                )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Convergence epoch: " + str(convergence_epoch))
    return ax


def bae_norm_fit_convergence(
    bae_ensemble,
    x,
    num_epoch,
    fast_window=10,
    slow_window=100,
    n_stop_points=5,
    scaler=None,
):
    if isinstance(x, np.ndarray) and scaler is not None:
        bae_ensemble.normalised_fit(scaler.transform(x), num_epochs=num_epoch)
    else:
        bae_ensemble.normalised_fit(x, num_epochs=num_epoch)
    convergence = detect_convergence(
        bae_ensemble.losses,
        fast_window=fast_window,
        slow_window=slow_window,
        n_stop_points=n_stop_points,
    )
    print(
        "LOSS "
        + str(len(bae_ensemble.losses))
        + ":"
        + str(np.mean(bae_ensemble.losses))
    )

    return bae_ensemble, convergence


def bae_fit_convergence(
    bae_ensemble,
    x,
    num_epoch,
    fast_window=10,
    slow_window=100,
    n_stop_points=5,
    scaler=None,
):
    if isinstance(x, np.ndarray) and scaler is not None:
        bae_ensemble.fit(scaler.transform(x), num_epochs=num_epoch)
    else:
        bae_ensemble.fit(x, num_epochs=num_epoch)
    convergence = detect_convergence(
        bae_ensemble.losses,
        fast_window=fast_window,
        slow_window=slow_window,
        n_stop_points=n_stop_points,
    )
    print(
        "LOSS "
        + str(len(bae_ensemble.losses))
        + ":"
        + str(np.mean(bae_ensemble.losses))
    )

    return bae_ensemble, convergence


def bae_fit_convergence_v2(
    bae_ensemble, x, num_epoch, scaler=None, threshold=1.1, verbose=True
):
    """
    Continuously calls fit until converges by checking for ratio of loss for current vs previous epoch.
    """
    # current_loss = np.mean(bae_ensemble.losses)
    num_it = bae_ensemble.num_iterations
    current_loss = np.mean(bae_ensemble.losses[-num_epoch * num_it :])
    if isinstance(x, np.ndarray) and scaler is not None:
        bae_ensemble.fit(scaler.transform(x), num_epochs=num_epoch)
    else:
        bae_ensemble.fit(x, num_epochs=num_epoch)
    # new_loss = np.mean(bae_ensemble.losses)
    new_loss = np.mean(bae_ensemble.losses[-num_epoch * num_it :])
    ratio = current_loss / new_loss
    if ratio <= threshold:
        convergence = 1
    else:
        convergence = 0

    if verbose:
        print(
            "LOSS {:d}: {:.6f} ({:.2f})".format(
                len(bae_ensemble.losses), new_loss, ratio
            )
        )

    return bae_ensemble, convergence


def bae_semi_fit_convergence(
    bae_ensemble,
    x,
    x_outliers,
    num_epoch,
    fast_window=10,
    slow_window=100,
    n_stop_points=5,
    scaler=None,
):
    if isinstance(x, np.ndarray) and scaler is not None:
        bae_ensemble.semisupervised_fit(
            scaler.transform(x),
            x_outliers=scaler.transform(x_outliers),
            num_epochs=num_epoch,
        )
    else:
        bae_ensemble.semisupervised_fit(x, x_outliers, num_epochs=num_epoch)

    convergence = detect_convergence(
        bae_ensemble.losses,
        fast_window=fast_window,
        slow_window=slow_window,
        n_stop_points=n_stop_points,
    )
    print(
        "LOSS "
        + str(len(bae_ensemble.losses))
        + ":"
        + str(np.mean(bae_ensemble.losses))
    )

    return bae_ensemble, convergence


# ---EXAMPLE--
# losses = pickle.load(open("loss.p","rb"))
# losses = losses[:90]
# convergence = detect_convergence(losses=losses, fast_window = 10, slow_window = 100, n_stop_points=5)
# plot_convergence(losses=losses, fast_window = 10, slow_window = 100, n_stop_points=5)
