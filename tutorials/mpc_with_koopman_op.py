import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.linalg import logm
from sklearn.base import BaseEstimator
from tqdm import tqdm

from datafold import EDMD, InitialCondition, TSCDataFrame, TSCTransformerMixin
from datafold.dynfold.dmd import DMDControl
from datafold.utils._systems import VanDerPol

# Note that it will not always converge when choosing another
rng = np.random.default_rng(55)


n_timeseries = 20  ## number of resets
n_timesteps = 200  ## how long we simulate the system for
dt = 0.01

vdp = VanDerPol(control_coord="y")

time_values = np.arange(0, n_timesteps * dt, dt)

X_ic = rng.uniform(-3.0, 3.0, size=(n_timeseries, 2))
idx = pd.MultiIndex.from_arrays([np.arange(n_timeseries), np.zeros(n_timeseries)])
X_ic = TSCDataFrame(X_ic, index=idx, columns=vdp.feature_names_in_)

U_tsc = rng.uniform(-3.0, 3.0, size=(n_timeseries, 1, 1))
U_tsc = np.tile(U_tsc, (1, n_timesteps - 1, 1))
U_tsc = TSCDataFrame.from_tensor(
    U_tsc,
    time_series_ids=X_ic.ids,
    columns=vdp.control_names_in_,
    time_values=time_values[:-1],
)

X_tsc, U_tsc = vdp.predict(X_ic, U=U_tsc)

for i in X_tsc.ids:
    idx = pd.IndexSlice[i, :]
    plt.plot(X_tsc.loc[idx, "x1"].to_numpy(), X_tsc.loc[idx, "x2"].to_numpy())


class VdPDictionary(BaseEstimator, TSCTransformerMixin):
    def get_feature_names_out(self, input_features=None):
        return ["x1^2", "x1^2 * x2"]

    def fit(self, X, y=None):
        self._setup_feature_attrs_fit(X)
        return self

    def transform(self, X: TSCDataFrame):
        X = X.copy()
        X["x1^2"] = np.square(X.loc[:, "x1"].to_numpy())
        X["x1^2 * x2"] = X["x1^2"].to_numpy() * X["x2"].to_numpy()
        return X.drop(["x1", "x2"], axis=1)


edmd = EDMD(
    dict_steps=[("vdpdict", VdPDictionary())],
    dmd_model=DMDControl(),
    include_id_state=True,
)
edmd.fit(X_tsc, U=U_tsc)

# Q = np.diag([1, 1, 0., 0.])
# R = np.diag([1.0]) * 1e-2

# Ad = edmd.dmd_model.sys_matrix_
# Bd = edmd.dmd_model.control_matrix_
# Pd = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
# Flqr = np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad

from datafold.appfold.mpc import LQR

n_timesteps_oos = 700  # simulation time
time_values_oos = np.arange(0, n_timesteps_oos * X_tsc.delta_time, X_tsc.delta_time)

X_ic_oos = rng.uniform(-3, 3, size=(1, 2))
X_ic_oos = InitialCondition.from_array(
    X_ic_oos, feature_names=edmd.feature_names_in_, time_value=0
)

target_state = InitialCondition.from_array(
    np.array([0, 0]), feature_names=edmd.feature_names_in_, time_value=0
)

lqr = LQR(edmd=edmd, cost_running=np.array([1, 1, 0, 0]), cost_input=1e-2)
lqr.preset_target_state(target_state)

# objects to fill in following loop
X_oos = TSCDataFrame.from_array(
    np.zeros((n_timesteps_oos, 2)),
    feature_names=vdp.feature_names_in_,
    time_values=time_values_oos,
)
U_oos = TSCDataFrame.from_array(
    np.zeros((n_timesteps_oos - 1, 1)),
    feature_names=vdp.control_names_in_,
    time_values=time_values_oos[:-1],
)

X_oos.iloc[0, :] = X_ic_oos.to_numpy()

for i in tqdm(range(1, n_timesteps_oos)):
    state = X_oos.iloc[[i - 1], :]
    U_oos.iloc[i - 1, :] = lqr.control_sequence(X=state)
    new_state, _ = vdp.predict(
        state, U=U_oos.iloc[[i - 1], :], time_values=time_values_oos[i - 1 : i + 1]
    )
    X_oos.iloc[i, :] = new_state.iloc[[1], :].to_numpy()


trajectory_uncontrolled, _ = vdp.predict(
    X_ic_oos, U=np.zeros((n_timesteps_oos - 1)), time_values=time_values_oos
)

plt.figure()
plt.plot(X_oos.to_numpy(), c="black", label="discrete")
plt.xlabel("t")
plt.legend()
plt.ylabel("x1, x2")

plt.figure()
plt.plot(
    X_oos.loc[:, "x1"].to_numpy(),
    X_oos.loc[:, "x2"].to_numpy(),
    c="red",
    label="controlled system",
)
plt.quiver(
    *X_oos.to_numpy()[:-1, :].T,
    *np.column_stack(
        [np.zeros_like(U_oos.to_numpy()), U_oos.to_numpy() / X_oos.delta_time]
    ).T,
    color="blue"
)
plt.plot(X_oos.iloc[0, 0], X_oos.iloc[0, 1], "o", c="red")
plt.plot(
    trajectory_uncontrolled.loc[:, "x1"].to_numpy(),
    trajectory_uncontrolled.loc[:, "x2"].to_numpy(),
    c="black",
    label="uncontrolled system",
)
plt.plot(
    trajectory_uncontrolled.iloc[0, 0],
    trajectory_uncontrolled.iloc[0, 1],
    "o",
    c="black",
    label="initial state",
)
plt.plot(
    target_state.iloc[0, 0],
    target_state.iloc[0, 1],
    "*",
    c="black",
    label="target state state",
)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

plt.figure()
plt.plot(np.linalg.norm(X_oos.to_numpy(), axis=1))
plt.axhline(np.linalg.norm(target_state.iloc[:2]), c="red")
plt.xlabel("t")
plt.ylabel("norm")
plt.show()
