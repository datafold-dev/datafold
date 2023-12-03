#!/usr/bin/env python

import cProfile
import warnings

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datafold import EDMD, TSCDataFrame, TSCIdentity
from datafold.dynfold.dmd import PartitionedDMD

warnings.filterwarnings("ignore")


def f1(x, t):
    return 1.0 / np.cosh(x + 3) * np.exp(2.3j * t)


def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.exp(2.8j * t)


def f(mu, x, t):
    return mu * f1(x, t) + (1 - mu) * f2(x, t)


n_space = 500
n_time = 160

x = np.linspace(-5, 5, n_space)
t = np.linspace(0, 4 * np.pi, n_time)

xgrid, tgrid = np.meshgrid(x, t)

training_params = np.round(np.linspace(0, 1, 10), 1)

plt.figure(figsize=(8, 2))
plt.scatter(training_params, np.zeros(len(training_params)), label="training")
plt.title("Training parameters")
plt.grid()
plt.xlabel(r"$\mu$")
plt.yticks([], [])


training_snapshots = np.stack([f(x=xgrid, t=tgrid, mu=p) for p in training_params])


X_train_d = TSCDataFrame.from_tensor(training_snapshots, time_values=t)
# TODO: make this a pd.DataFrame (to support later multiple parameters)
P_train_d = pd.DataFrame(training_params, index=X_train_d.ids)


def title(param):
    return rf"$\mu$={param}"


def visualize(X, param, ax, log=False, labels_func=None):
    ax.set_title(title(param))
    if labels_func != None:
        labels_func(ax)
    if log:
        return ax.pcolormesh(X.real.T, norm=colors.LogNorm(vmin=X.min(), vmax=X.max()))
    else:
        return ax.pcolormesh(X.real.T)


def visualize_multiple(
    Xs, params, log=False, figsize=(20, 6), labels_func=None, title=None
):
    if log:
        Xs[Xs == 0] = np.min(Xs[Xs != 0])

    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(nrows=1, ncols=5, sharey=True)

    if labels_func is None:

        def labels_func_default(ax):
            ax.set_yticks([0, n_time // 2, n_time])
            ax.set_yticklabels(["0", r"$\pi$", r"2$\pi$"])

            ax.set_xticks([0, n_space // 2, n_space])
            ax.set_xticklabels(["-5", "0", "5"])

        labels_func = labels_func_default

    im = [
        visualize(X.T, param, ax, log, labels_func)
        for X, param, ax in zip(Xs, params, axes)
    ][-1]

    fig.colorbar(im, ax=axes)

    if fig is not None:
        fig.suptitle(title)


idxes = [0, 2, 4, 6, 8]
visualize_multiple(training_snapshots[idxes], training_params[idxes])


pdmd = EDMD(
    dict_steps=[("_id", TSCIdentity())],
    dmd_model=PartitionedDMD(n_components=20, dmd_kwargs=dict(rank=20)),
)
pdmd = pdmd.fit(X_train_d, P=P_train_d)

profiler = cProfile.Profile()

# Run the profiler on your function
profiler.enable()
X_reconstruct = pdmd.reconstruct(X_train_d, P=P_train_d)
profiler.disable()

profiler.print_stats(sort="cumulative")
exit()

# TODO: include this in a test for partitioned DMD (for a single time series)
# X_reconstruct_test = dfdmd.reconstruct(X_train_d.loc[[0], :], P=P_train_d.loc[[0], :])

visualize_multiple(
    X_train_d.to_tensor("row"),
    training_params,
    figsize=(20, 2.5),
    title="training",
)

visualize_multiple(
    X_reconstruct.to_tensor("row"),
    training_params,
    figsize=(20, 2.5),
    title="DMD (datafold)",
)

similar_testing_params = [1, 3, 5, 7, 9]
testing_params = training_params[similar_testing_params] + np.array(
    [5 * pow(10, -i) for i in range(2, 7)]
)
testing_params_labels = [
    str(training_params[similar_testing_params][i - 2]) + f"+$5*10^{{-{i}}}$"
    for i in range(2, 7)
]

time_step = t[1] - t[0]
N_predict = 40
N_nonpredict = 40

t2 = np.array(
    [4 * np.pi + i * time_step for i in range(-N_nonpredict + 1, N_predict + 1)]
)
xgrid2, tgrid2 = np.meshgrid(x, t2)

testing_snapshots = np.array([f(mu=p, x=xgrid2, t=tgrid2) for p in testing_params])


plt.figure(figsize=(8, 2))
plt.scatter(training_params, np.zeros(len(training_params)), label="Training")
plt.scatter(testing_params, np.zeros(len(testing_params)), label="Testing")
plt.legend()
plt.grid()
plt.title("Training vs testing parameters")
plt.xlabel(r"$\mu$")
plt.yticks([], [])


# TODO: support reconstruct method
# X_reconstruct_test = dfdmd.reconstruct(X_test_d, P=P_test_d)

X_test_d = TSCDataFrame.from_tensor(testing_snapshots, time_values=t2)
P_test_d = pd.DataFrame(testing_params, index=X_test_d.ids)

# X_reconstruct_test = pdmd.predict(
#     X_test_d.initial_states(), P=P_test_d, time_values=X_test_d.time_values()
# )

X_reconstruct_test = pdmd.reconstruct(X_test_d, P=P_test_d)


# this is needed to visualize the time/space in the appropriate way
def labels_func(ax):
    l = X_test_d.shape[0]

    ax.set_yticks([0, l // 2, l])
    ax.set_yticklabels([r"3\pi", r"4$\pi$", r"5$\pi$"])

    ax.set_xticks([0, n_space // 2, n_space])
    ax.set_xticklabels(["-5", "0", "5"])


visualize_multiple(
    X_reconstruct_test.to_tensor("row"),
    testing_params_labels,
    figsize=(20, 2.5),
    labels_func=labels_func,
    title="DMD (datafold)",
)

visualize_multiple(
    testing_snapshots,
    testing_params_labels,
    figsize=(20, 2.5),
    labels_func=labels_func,
    title="ground truth",
)


plt.show()
