import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(dfx, cols=None, axes=None, **kwargs):
    if cols is None:
        cols = dfx.columns

    if axes is None:
        N = len(cols)
        m = 1
        n = int(np.ceil(N / m))

        fig = plt.figure(figsize=(m * 12, n * 4))
        axes = fig.subplots(n, m).flatten()
        ret = fig, axes
    else:
        ret = None

    for i, name in enumerate(cols):
        if name in dfx:
            axes[i].plot(dfx[name].values, **kwargs)
            axes[i].set_title(name)
            axes[i].grid(True)

    return ret


def plot_mpc(mpc, reference, initial_conds, horizon, axes=None):
    reference = reference.iloc[: horizon + 1]

    mpc_ret = mpc.predict(reference, initial_conds)
    mpc_ret.pred["u"] = mpc_ret.control

    cols = ["x", "xdot", "theta", "thetadot", "u"]

    if axes is None:
        fig = plt.figure(figsize=(18, 8))
        axes = fig.subplots(2, 3).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        y1 = reference[col].to_numpy()
        y2 = mpc_ret.pred[col].to_numpy()
        y3 = mpc_ret.actual[col].to_numpy()

        ax.plot(y1, label="ref")
        ax.plot(y2, label="mpc pred")
        ax.plot(y3, linestyle="--", label="mpc actual")
        ax.set_title(col)

    return mpc_ret, axes


def plot_pred(
    trajectories,
    predictor,
    state_cols=None,
    input_cols=None,
    augment_fn=None,
    n=None,
    max_t=None,
    plot_diff=False,
):
    if state_cols is None:
        state_cols = predictor.state_cols
    if input_cols is None:
        input_cols = predictor.input_cols

    if n is None:
        n = len(trajectories)
    m = len(state_cols) + len(input_cols)

    fig = plt.figure(figsize=(6 * m, 3 * n))
    axes = fig.subplots(n, m)

    for i, traj in enumerate(trajectories):
        ic = traj.ic
        dfx = traj.dfx

        if augment_fn is not None:
            ic = augment_fn(ic)
            dfx = augment_fn(dfx)

        if i >= n:
            break

        k = 0
        control = dfx["u"]
        t = dfx["t"]

        if input_cols:
            pred = predictor.predict(ic, control, t).pred
        else:
            pred = predictor.predict(ic, t).pred

        if plot_diff:
            for col in state_cols:
                ax = axes[i, k]

                diff = np.abs((dfx[col].values - pred[col].values.flatten()))
                ax.plot(np.log(diff[:max_t]) / np.log(10))

                ax.set_title(col)
                k += 1

        else:
            for col in state_cols:
                ax = axes[i, k]
                ax.plot(dfx[col].values[:max_t])
                ax.plot(pred[col].values[:max_t])
                ax.set_title(col)
                k += 1

            for col in input_cols:
                ax = axes[i, k]
                ax.plot(dfx[col].values)
                ax.set_title(col)
                k += 1

        axes[i, 0].set_ylabel(f"{i}")

    return fig, axes


def plot_tsc(tsc, axes=None, cols=None, m=2):
    if cols is None:
        cols = set(tsc.columns)
        if "t" in cols:
            cols -= {"t"}
        cols = list(cols)

    N = len(cols)
    n = int(np.ceil(N / m))

    if axes is None:
        fig = plt.figure()
        axes = fig.subplots(n, m).flatten()
        ret = fig, axes
    else:
        ret = None

    for i, col in enumerate(cols):
        axes[i].plot(tsc[col].values)
        axes[i].set_title(col)
        axes[i].grid(True)

    return ret
