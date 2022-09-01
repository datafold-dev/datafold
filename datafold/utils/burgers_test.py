import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
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
from datafold.appfold.kmpc import LinearKMPC
from datafold.utils._systems import Burger

# simulates the setting from https://arxiv.org/pdf/1804.05291.pdf

rng = np.random.default_rng(2)
plot = False

# data generation options
dt = 0.01
sim_length = 200
training_size = 100

# MPC options
Tpred = 0.1  # prediction horizon
horizon = int(Tpred // dt)
Tend = 6
Nsim = int(Tend // (dt * horizon))

# control options
umin, umax = (-0.1, 0.1)

# START

time_values = np.arange(0, dt * sim_length + 1e-15, dt)

sys = Burger()

f1 = lambda x: np.atleast_2d(np.exp(-((15 * (x - 0.25)) ** 2)))
f2 = lambda x: np.atleast_2d(np.exp(-((15 * (x - 0.75)) ** 2)))

ic1 = np.exp(-(((sys.x_nodes - 0.5) * 5) ** 2))
ic2 = np.sin(4 * np.pi * sys.x_nodes) ** 2
icfunc = lambda a: a * ic1 + (1 - a) * ic2

X_tsc = []
U_tsc = []

MODE_DATA = ["generate_save", "load", "matlab"][2]
if MODE_DATA == "generate_save":
    for i in range(training_size):
        ic = icfunc(rng.uniform(0, 1))

        print(i)

        rand_vals = rng.uniform(umin, umax, size=(len(time_values), 2))
        # rand_vals = np.zeros((len(time_values), 2))
        U1rand = lambda t: np.atleast_2d(np.interp(t, time_values, rand_vals[:, 0])).T
        U2rand = lambda t: np.atleast_2d(np.interp(t, time_values, rand_vals[:, 1])).T

        def U(t, x):
            if x.shape[1] == 1:
                x = x.T
            return U1rand(t) * f1(x) + U2rand(t) * f2(x)

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

    # f = plt.figure()
    #
    # tsid = 5
    # (model_line,) = plt.plot(sys.x_nodes, X_tsc.loc[pd.IndexSlice[tsid, :], :].iloc[0].to_numpy(), label="model matlab")
    # (ref_line,) = plt.plot(sys.x_nodes, X_tsc_own.loc[pd.IndexSlice[tsid, :], :].iloc[0].to_numpy(), label="model own")
    # plt.legend()
    #
    # def func(i):
    #     model_line.set_ydata(X_tsc.loc[pd.IndexSlice[tsid, :], :].iloc[i, :].to_numpy())
    #     ref_line.set_ydata(X_tsc_own.loc[pd.IndexSlice[tsid, :], :].iloc[i, :].to_numpy())
    #     return (model_line, ref_line,)
    #
    # anim = FuncAnimation(f, func=func, frames=X_tsc.shape[0], interval=500)
    # plt.show()
    # exit()

print(f"{X_tsc.head(5)}")
print(f"{U_tsc.head(5)}")

# subselect measurements to every 10th node
X_tsc_reduced = X_tsc.iloc[:, 0::10]

assert isinstance(X_tsc_reduced, TSCDataFrame)

# move time values in U_tsc


def shift_index_U(_X, _U):
    new_index = _X.groupby("ID").tail(_X.n_timesteps - 1).index
    return _U.set_index(new_index)


X_tsc_reduced = pd.concat(
    [X_tsc_reduced, shift_index_U(X_tsc_reduced, U_tsc)], axis=1
).fillna(0)


class L2Norm(BaseEstimator, TSCTransformerMixin):
    def fit(self, X):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["l2norm"]

    def transform(self, X: TSCDataFrame, y=None):
        return TSCDataFrame.from_same_indices_as(
            X,
            np.linalg.norm(X.to_numpy(), axis=1),
            except_columns=self.get_feature_names_out(),
        )


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

edmd.fit(X_tsc_reduced, U=U_tsc)
X_dict = edmd.transform(X_tsc_reduced.head(10))

print(f"{X_dict.columns=}")
print(f"{len(X_dict.columns)=}")
print(X_dict.head(5))

# TODO: use cvxpy instead of quadprog
kmpc = LinearKMPC(
    predictor=edmd,
    horizon=horizon,
    state_bounds=np.array([[np.inf, np.inf]]),
    input_bounds=np.array([[-0.1, 0.1], [-0.1, 0.1]]),
    qois=list(np.arange(10)),
    cost_running=1,
    cost_terminal=1,
    cost_input=1,
)


# perform simulation for the initial time embedding
X_init, _ = sys.predict(
    icfunc(0.2),
    U=np.zeros((5, sys.n_control_in_)),
    time_values=np.arange(0, 5 * dt, dt),
    require_last_control_state=True,
)

start_time = X_init.time_values()[-1]
time_values_ref = np.arange(start_time, start_time + Tend, dt)
X_ref = np.zeros(len(time_values_ref))
X_ref[time_values_ref <= 2] = 0.5
X_ref[np.logical_and(time_values_ref > 2, time_values_ref < 4)] = 1
X_ref[time_values_ref > 4] = 0.5
X_ref = np.outer(X_ref, np.ones(X_tsc.shape[1]))
X_ref = TSCDataFrame.from_array(
    X_ref, time_values=time_values_ref, feature_names=X_tsc.columns
)

X_ref_reduced = X_ref.iloc[:, ::10]

U_ic = TSCDataFrame.from_array(
    np.zeros((5, 2)), time_values=X_init.time_values(), feature_names=["u1", "u2"]
)

edmd_state = pd.concat([X_init.iloc[:, ::10], U_ic], axis=1)
model_state = X_init.iloc[[-1], :]

X_model_evolution = [X_init.tsc.drop_last_n_samples(1)]
U_evolution = [U_ic.tsc.drop_last_n_samples(1)]

for i in range(Nsim):
    print(i)

    start_i = i * horizon
    end_i = (i + 1) * horizon + 1
    ref = X_ref_reduced.iloc[start_i:end_i, :]
    time_values_predict = ref.time_values()

    U = kmpc.generate_control_signal(edmd_state, reference=ref)
    U = TSCDataFrame.from_array(
        U, time_values=time_values_predict[:-1], feature_names=edmd.control_names_in_
    )

    U1 = lambda t: np.atleast_2d(
        np.interp(t, time_values_predict[:-1], U.iloc[:, 0].to_numpy())
    ).T
    U2 = lambda t: np.atleast_2d(
        np.interp(t, time_values_predict[:-1], U.iloc[:, 1].to_numpy())
    ).T

    def Ufunc(t, x):
        if x.shape[1] == 1:
            x = x.T
        return U1(t) * f1(x) + U2(t) * f2(x)

    X_model, _ = sys.predict(model_state, U=Ufunc, time_values=time_values_predict)

    # prepare new edmd state
    X_model_last = X_model.iloc[-edmd.n_samples_ic_ :, ::10]
    U_last = U.iloc[-edmd.n_samples_ic_ + 1 :, :]
    U_last_shifted = shift_index_U(X_model_last, shift_index_U(X_model_last, U_last))
    edmd_state = pd.concat([X_model_last, U_last_shifted], axis=1).fillna(0)

    # set new model state from where to predict
    model_state = X_model.iloc[[-1], :]

    # remove last state, bc. it is also used as the initial condition for the next (avoid duplicate states)
    X_model_evolution.append(X_model.tsc.drop_last_n_samples(1))
    U_evolution.append(U)

X_model_evolution = pd.concat(X_model_evolution, axis=0)
U_evolution = pd.concat(U_evolution, axis=0)
# TODO: measure output error (state-reference) over time!

if True:

    f, ax = plt.subplots(nrows=2)

    (model_line,) = ax[0].plot(sys.x_nodes, X_model_evolution.iloc[0], label="model")
    (ref_line,) = ax[0].plot(sys.x_nodes, X_ref.iloc[0], label="reference")

    Ufunc = lambda u, x: (u[0] * f1(x) + u[1] * f2(x)).ravel()
    (control_line,) = ax[1].plot(
        sys.x_nodes,
        Ufunc(
            U_evolution.iloc[0, :].to_numpy(), X_model_evolution.iloc[0, :].to_numpy()
        ),
        label="control",
    )

    plt.legend()

    def func(i):
        model_line.set_ydata(X_model_evolution.iloc[i, :])
        ref_line.set_ydata(X_ref.iloc[i, :])
        control_line.set_ydata(
            Ufunc(
                U_evolution.iloc[i].to_numpy(), X_model_evolution.iloc[i, :].to_numpy()
            )
        )
        return (
            model_line,
            ref_line,
        )

    anim = FuncAnimation(f, func=func, frames=X_model_evolution.shape[0])
    plt.show()
