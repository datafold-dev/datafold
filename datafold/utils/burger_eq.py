import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from scipy.io import loadmat
from sklearn.base import BaseEstimator

from datafold import (
    EDMD,
    DMDControl,
    TSCColumnTransformer,
    TSCDataFrame,
    TSCIdentity,
    TSCTakensEmbedding,
    TSCTransformerMixin,
)
from datafold.appfold.mpc import LinearKMPC
from datafold.utils._systems import Burger1DPeriodicBoundary

# simulates the setting from https://arxiv.org/pdf/1804.05291.pdf

rng = np.random.default_rng(2)
plot = True

# data generation options
dt = 0.19
sim_length = 200
training_size = 100

# MPC options
Tpred = dt * 20  # prediction horizon
horizon = int(np.round(Tpred // dt))
Tend = 60  # 6
Nsim = int(Tend // dt) + 1

# control options
umin, umax = (-0.1, 0.1)

# START

time_values = np.arange(0, dt * sim_length + 1e-12, dt)

sys = Burger1DPeriodicBoundary(nu=0.01)

# f1 = np.atleast_2d(np.exp(-((15 * (sys.x_nodes - 0.25)) ** 2)))
# f2 = np.atleast_2d(np.exp(-((15 * (sys.x_nodes - 0.75)) ** 2)))

f1 = np.atleast_2d(np.exp(-((15/(2*np.pi) * (sys.x_nodes - 0.5*np.pi)) ** 2)))
f2 = np.atleast_2d(np.exp(-((15/(2*np.pi) * (sys.x_nodes - 1.5*np.pi)) ** 2)))

# f1 = np.atleast_2d(np.exp(-((15 * (np.linspace(0,1, 100) - 0.25)) ** 2)))
# f2 = np.atleast_2d(np.exp(-((15 * (np.linspace(0,1, 100) - 0.75)) ** 2)))

# plt.figure()
# plt.plot(sys.x_nodes, f1.flatten())
# plt.plot(sys.x_nodes, f2.flatten())
# # plt.plot(np.linspace(0,1, 100), f1.flatten())
# # plt.plot(np.linspace(0,1, 100), f2.flatten())
# plt.xlim(0, 2*np.pi)
# # plt.xlim(0, 1)
# plt.show()
# exit()

ic1 = np.exp(-(((sys.x_nodes - 2*np.pi*0.5) * 5/(2*np.pi)) ** 2))
ic2 = np.sin(2 * sys.x_nodes) ** 2
icfunc = lambda a: a * ic1 + (1 - a) * ic2

# sys.predict(np.row_stack([icfunc(0), icfunc(1)]), U=np.zeros((2, 100)), time_values=np.array([0, dt]))

X_tsc = []
U_tsc = []

MODE_DATA = ["generate_save", "load", "matlab"][1]
print(f"{MODE_DATA=}")
if MODE_DATA == "generate_save":
    for i in range(training_size):
        ic = icfunc(rng.uniform(0, 1))

        print(f"{i} / {training_size}")

        rand_vals = rng.uniform(umin, umax, size=(len(time_values), 2))
        # rand_vals = np.zeros((len(time_values), 2))

        U1rand = lambda t: np.atleast_2d(interp1d(time_values, rand_vals[:, 0], kind="previous")(t)).T
        U2rand = lambda t: np.atleast_2d(interp1d(time_values, rand_vals[:, 1], kind="previous")(t)).T

        def U(t, x):
            return U1rand(t) * f1 + U2rand(t) * f2

        X_predict, Ufull = sys.predict(
            ic, U=U, time_values=time_values, require_last_control_state=False
        )

        # U = TSCDataFrame.from_same_indices_as(Ufull, rand_vals, except_columns=["u1", "u2"])
        # drop last control input, as it is not required
        U = TSCDataFrame.from_array(
            rand_vals[:-1, :],
            time_values=Ufull.time_values(),
            feature_names=["u1", "u2"],
        )

        X_tsc.append(X_predict)
        U_tsc.append(U)

    X_tsc = TSCDataFrame.from_frame_list(X_tsc)
    U_tsc = TSCDataFrame.from_frame_list(U_tsc)

    X_tsc.to_csv("X_tsc.csv")
    U_tsc.to_csv("U_tsc.csv")

elif MODE_DATA == "load":
    X_tsc = TSCDataFrame.from_csv("X_tsc.csv")
    U_tsc = TSCDataFrame.from_csv("U_tsc.csv")

elif MODE_DATA == "matlab":
    mat = loadmat("BurgersTrajectoryData.mat")
    training_size, sim_length, X, Y, U = (
        int(mat["Ntraj"]),
        int(mat["SimLength"]),
        mat["X"],
        mat["Y"],
        mat["U"],
    )

    X_tsc_own = TSCDataFrame.from_csv("X_tsc.csv")

    X_tsc = np.zeros((training_size, sim_length + 1, 100))

    X_tsc[:, :-1, :] = np.reshape(X.T, (training_size, sim_length, 100))
    X_tsc[:, -1, :] = np.reshape(Y.T, (training_size, sim_length, 100))[:, -1, :]

    X_tsc = TSCDataFrame.from_tensor(
        X_tsc, time_values=time_values, columns=[f"x{i}" for i in range(100)]
    )

    U_tsc = np.reshape(U.T, (training_size, sim_length, 2))
    U_tsc = TSCDataFrame.from_tensor(
        U_tsc, time_values=time_values[:-1], columns=[f"u{i+1}" for i in range(2)]
    )

    f = plt.figure()

    tsid = 0
    (model_line,) = plt.plot(sys.x_nodes, X_tsc.loc[pd.IndexSlice[tsid, :], :].iloc[0].to_numpy(), label="model matlab")
    (ref_line,) = plt.plot(sys.x_nodes, X_tsc_own.loc[pd.IndexSlice[tsid, :], :].iloc[0].to_numpy(), label="model own")
    plt.legend()

    def func(i):
        model_line.set_ydata(X_tsc.loc[pd.IndexSlice[tsid, :], :].iloc[i, :].to_numpy())
        ref_line.set_ydata(X_tsc_own.loc[pd.IndexSlice[tsid, :], :].iloc[i, :].to_numpy())
        return (ref_line,)

    anim = FuncAnimation(f, func=func, frames=X_tsc.shape[0])
    plt.show()
    exit()

print(f"{X_tsc.head(5)}")
print(f"{U_tsc.head(5)}")

plt_trajectory = False
if plt_trajectory:
    f, ax = plt.subplots(nrows=2)
    tsid = 0
    (ref_line,) = ax[0].plot(sys.x_nodes,
                           X_tsc.loc[pd.IndexSlice[tsid, :], :].iloc[0].to_numpy(),
                           label="model")

    def Ufunc(u):
        return u[0] * f1 + u[1] * f2

    (control_line,) = ax[1].plot(sys.x_nodes,
                           Ufunc(U_tsc.loc[pd.IndexSlice[tsid, :], :].iloc[0].to_numpy()).ravel(),
                           label="model")
    plt.legend()

    def func(i):
        ref_line.set_ydata(X_tsc.loc[pd.IndexSlice[tsid, :], :].iloc[i, :].to_numpy())
        control_line.set_ydata(Ufunc(U_tsc.loc[pd.IndexSlice[tsid, :], :].iloc[i].to_numpy()).ravel())
        return (ref_line, control_line, )


    anim = FuncAnimation(f, func=func, frames=U_tsc.shape[0])
    plt.show()
    exit()

def subselect_measurements(tscdf):
    # subselect measurements to every 10th node
    return tscdf.iloc[:, 9::10]

X_tsc_reduced = subselect_measurements(X_tsc)

assert isinstance(X_tsc_reduced, TSCDataFrame)

# move time values in U_tsc


def shift_time_index_U(_X, _U):
    new_index = _X.groupby("ID").tail(_X.n_timesteps - 1).index
    return _U.set_index(new_index)


X_tsc_reduced = pd.concat(
    [X_tsc_reduced, shift_time_index_U(X_tsc_reduced, U_tsc)], axis=1
).fillna(0)


class L2Norm(BaseEstimator, TSCTransformerMixin):
    def fit(self, X):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["l2norm"]

    def transform(self, X: TSCDataFrame, y=None):
        return TSCDataFrame.from_same_indices_as(
            X,
            # np.linalg.norm(X.to_numpy(), axis=1),
            np.sum(np.square(np.abs(X.to_numpy())), axis=1) / X.shape[1],
            except_columns=self.get_feature_names_out(),
        )


# class ReorderColumns(BaseEstimator, TSCTransformerMixin):
#     def fit(self, X):
#         return self
#
#     def get_feature_names_out(self, input_features=None):
#         firsts = np.logical_and(input_features.str.contains("x"), input_features.str.contains("d"))
#         second = np.logical_and(input_features.str.contains("x"), ~input_features.str.contains("d"))
#         third = input_features.str.contains("u")
#         fourth = input_features.str.contains("l2norm")
#         fifth = input_features.str.contains("const")
#
#         return input_features[firsts].append(input_features[second]).append(input_features[third]).append(input_features[fourth]).append(input_features[fifth])
#
#
#     def transform(self, X):
#         return X.loc[:, self.get_feature_names_out(X.columns)]


l2norm = ("l2", L2Norm(), lambda df: df.columns.str.startswith("x"))
_id = ("id", "passthrough", lambda df: df.columns)
l2norm = (
    "l2norm",
    TSCColumnTransformer([l2norm, _id], verbose_feature_names_out=False),
)

delay1 = (
    "delay_x",
    TSCTakensEmbedding(delays=4),
    lambda df: df.columns.str.startswith("x"),
)
delay2 = (
    "delay_u",
    TSCTakensEmbedding(delays=3),
    lambda df: df.columns.str.startswith("u"),
)
_l2norm = ("l2norm", "passthrough", ["l2norm"])
tde = (
    "tde",
    TSCColumnTransformer([delay1, delay2, _l2norm], verbose_feature_names_out=False),
)

_id = ("_id", TSCIdentity(include_const=True))

# _reorder = ("reorder", ReorderColumns())

edmd = EDMD([l2norm, tde, _id], dmd_model=DMDControl(), include_id_state=False)

# import tempfile
# from sklearn.utils import estimator_html_repr
# import webbrowser
# with tempfile.NamedTemporaryFile("w", suffix=".html") as fp:
#     fp.write(estimator_html_repr(tde[1]))
#     fp.flush()
#     webbrowser.open_new_tab(fp.name)
#     input("Press Enter to continue...")
# exit()

# The last sample is not required (as there is no prediction from the last state)
# U_tsc = U_tsc.tsc.drop_last_n_samples(1)

edmd.fit(X_tsc_reduced, U=U_tsc, dict_preserves_id_states=True)

X_dict = edmd.transform(X_tsc_reduced.head(10))

print(f"{X_dict.columns=}")
print(f"{len(X_dict.columns)=}")
print(X_dict.head(5))

kmpc = LinearKMPC(
    edmd=edmd,
    horizon=horizon,
    state_bounds=None,
    input_bounds=np.array([[-0.1, 0.1], [-0.1, 0.1]]),
    qois=X_tsc_reduced.columns[X_tsc_reduced.columns.str.startswith("x")],
    cost_running=1,
    cost_terminal=1,
    cost_input=1,
)

# perform simulation for the initial time embedding
X_init, _ = sys.predict(
    icfunc(0.2),
    U=np.zeros((4, sys.n_control_in_)),
    time_values=np.arange(0, edmd.n_samples_ic_ * dt, dt),
    require_last_control_state=True,
)

start_time = X_init.time_values()[-1]
time_values_ref = np.arange(0, start_time + Tend, dt)
X_ref = np.zeros(len(time_values_ref))
X_ref[time_values_ref <= 20] = 0.5
X_ref[np.logical_and(time_values_ref > 20, time_values_ref < 40)] = 0.6
X_ref[time_values_ref > 40] = 0.5
X_ref = np.outer(X_ref, np.ones(X_tsc.shape[1]))
X_ref = TSCDataFrame.from_array(
    X_ref, time_values=time_values_ref, feature_names=X_tsc.columns
)

X_ref_reduced = subselect_measurements(X_ref)

U_ic = TSCDataFrame.from_array(
    np.zeros((5, 2)), time_values=X_init.time_values(), feature_names=["u1", "u2"]
)

edmd_state = pd.concat([subselect_measurements(X_init), U_ic], axis=1)
model_state = X_init.iloc[[-1], :]

X_model_evolution = X_init
U_evolution = U_ic

X_model_unctr_evolution = X_init.copy()

X_error_evolution = X_init - X_ref.iloc[[0], :].to_numpy()

for i in range(X_init.shape[0], Nsim):
    print(i)

    # start horizon from next state (we can't change the current state)
    reference = X_ref_reduced.iloc[i:i + horizon, :]

    t = X_model_evolution.time_values()[-1]
    t_new = X_model_evolution.time_values()[-1] + dt

    if reference.shape[0] != kmpc.horizon:
        # stop if the rest of reference signal is smaller than horizon
        break

    # initvals=U_evolution.iloc[-1, :].to_numpy() if i > 1 else None
    U = kmpc.control_sequence(edmd_state, reference=reference)

    # use only the first control input and optimize again for the next window
    Ufull = U.iloc[0, 0] * f1 + U.iloc[0, 1] * f2

    X_model, _ = sys.predict(X_model_evolution.iloc[[-1], :], U=Ufull, time_values=np.array([t, t_new]))
    X_model_evolution = pd.concat([X_model_evolution, X_model.iloc[[1], :]], axis=0)

    diff = X_model.iloc[[1], :] - X_ref.iloc[[i], :].to_numpy()
    X_error_evolution = pd.concat([X_error_evolution, diff])

    X_model_unctr, _ = sys.predict(X_model_unctr_evolution.iloc[[-1], :], U=np.zeros_like(sys.x_nodes)[np.newaxis, :], time_values=np.array([t, t_new]))
    X_model_unctr_evolution = pd.concat([X_model_unctr_evolution, X_model_unctr.iloc[[1], :]], axis=0)

    U_evolution = pd.concat([U_evolution, U.iloc[[0], :]], axis=0)

    # prepare new edmd state
    X_model_last = subselect_measurements(X_model_evolution.iloc[-edmd.n_samples_ic_ :, :])
    U_last = U_evolution.iloc[-edmd.n_samples_ic_ :-1, :]
    U_last_shifted = shift_time_index_U(X_model_last, shift_time_index_U(X_model_last, U_last))
    edmd_state = pd.concat([X_model_last, U_last_shifted], axis=1).fillna(0)

if True:

    f, ax = plt.subplots(nrows=3, sharex=True)

    (model_line,) = ax[0].plot(sys.x_nodes, X_model_evolution.iloc[0], label="model")
    (model_uctr_line,) = ax[0].plot(sys.x_nodes, X_model_unctr_evolution.iloc[0], label="model uncontrolled")
    (ref_line,) = ax[0].plot(sys.x_nodes, X_ref.iloc[0], label="reference")
    plt.legend()

    Ufunc = lambda u, x: (u[0] * f1 + u[1] * f2).ravel()
    (control_line,) = ax[1].plot(
        sys.x_nodes,
        Ufunc(U_evolution.iloc[0, :].to_numpy(), None),
        label="control",
    )
    plt.legend()

    (error_line,) = ax[2].plot(sys.x_nodes, X_error_evolution.iloc[0, :].to_numpy(), label="difference")
    plt.legend()

    def func(i):
        model_line.set_ydata(X_model_evolution.iloc[i, :].to_numpy())
        model_uctr_line.set_ydata(X_model_unctr_evolution.iloc[i, :].to_numpy())
        ref_line.set_ydata(X_ref.iloc[i, :].to_numpy())
        control_line.set_ydata(Ufunc(U_evolution.iloc[i].to_numpy(), None))
        error_line.set_ydata(X_error_evolution.iloc[i, :].to_numpy())

        return (
            model_line,
            model_uctr_line,
            ref_line,
            error_line
        )

    anim = FuncAnimation(f, func=func, frames=U_evolution.shape[0])
    plt.show()
