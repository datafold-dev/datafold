import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.io

from datafold import (
    EDMD,
    DMDControl,
    InitialCondition,
    TSCColumnTransformer,
    TSCDataFrame,
    TSCIdentity,
    TSCRadialBasis,
    TSCTakensEmbedding,
)
from datafold.appfold.mpc import LinearKMPC
from datafold.pcfold.kernels import ThinplateKernel
from datafold.utils._systems import Motor

# System
sys = Motor(ivp_kwargs={"atol": 1e-2, "rtol": 1e-5, "method": "RK23"})
select = "x2"


# Samling
dt = 0.01
Ntraj = 200
Nsim = 1000

nD = 1
ny = 1
rng = np.random.default_rng(1)

time_values = np.arange(0, Nsim * dt, dt)


# A function to required to augment the state with control input
def shift_time_index_U(_X, _U):
    new_index = _X.groupby("ID").tail(_X.n_timesteps - 1).index
    return _U.set_index(new_index)


X_ic = InitialCondition.from_array(
    rng.uniform(size=(Ntraj, sys.n_features_in_)) * 2 - 1,
    time_value=0.0,
    feature_names=sys.feature_names_in_,
)

U = 2 * rng.uniform(size=(Ntraj, Nsim - 1, 1)) - 1
U = TSCDataFrame.from_tensor(
    U, time_values=time_values[:-1], columns=sys.control_names_in_
)

X_ic_xtest = TSCDataFrame.from_array(
    np.array([-0.5332, -0.9401]), feature_names=X_ic.columns
)
U_ic_xtest = TSCDataFrame.from_array(np.array([0.2400]), feature_names=U.columns)

# TODO: need to fix the arguments here... is Nsim really required?
X_tsc, U_tsc = sys.predict_vectorize(X_ic, U=U, nsim=Nsim, dt=dt)
X_tsc = pd.concat(
    [X_tsc.loc[:, [select]], shift_time_index_U(X_tsc, U_tsc)], axis=1
).fillna(0)


# Set up EDMD
delay = ("delay", TSCTakensEmbedding(delays=1), [select])
_id = ("id", TSCIdentity(), ["u"])

t1 = (
    "t1",
    TSCColumnTransformer(
        transformers=[delay, _id], verbose_feature_names_out=False
    ),
)
rbf = ("rbf", TSCRadialBasis(kernel=ThinplateKernel(), center_type="fit_params"))
edmd = EDMD([t1, rbf], dmd_model=DMDControl(), include_id_state=True)

centers = rng.uniform(size=(100, 3)) * 2 - 1

# Fit EDMD model
edmd = edmd.fit(X_tsc, U=U_tsc, rbf__centers=centers)

# Specify a test sampling to see the predictive power of EDMD
Tmax = 1
Nsim = int(Tmax / dt)


# TODO: need to rename this, why is it there?
def myprbs(N, duty_cycle):
    cond = rng.uniform(0, 1, size=(N, 1)) > (1 - duty_cycle)
    return cond.astype(float)

U_test = TSCDataFrame.from_array(
    2 * myprbs(Nsim, 0.5) - 1,
    time_values=np.arange(0, Nsim * dt, dt),
    feature_names=U_tsc.columns,
)

X_ic_orig = TSCDataFrame.from_array(
    rng.uniform(0, 1, size=(1, 2)) - 0.5, feature_names=X_tsc.columns[:2]
)
U_ic_edmd = TSCDataFrame.from_array(
    2 * rng.uniform(0, 1, size=(1, 1)) - 1, feature_names=U_tsc.columns
)
X_ic_edmd, U_ic_edmd = sys.predict_vectorize(X_ic_orig, U=U_ic_edmd, dt=dt, nsim=2)
X_ic_edmd = pd.concat(
    [X_ic_edmd.loc[:, [select]], shift_time_index_U(X_ic_edmd, U_ic_edmd)], axis=1
).fillna(0)

X_true_test, _ = sys.predict_vectorize(X_ic_orig, U=U_test, dt=dt, nsim=U_test.shape[0])

X_edmd_pred = edmd.predict(X_ic_edmd, U=U_test)

f, ax = plt.subplots()
ax.set_title("control input")
ax.set_xlabel("time [s]")
ax.set_ylabel("U")
ax.step(U_test.time_values(), U_test.to_numpy())

f, ax = plt.subplots()
ax.plot(
    X_true_test.time_values(), X_true_test.loc[:, select].to_numpy(), label="original"
)
ax.plot(X_edmd_pred.time_values(), X_edmd_pred.loc[:, select].to_numpy(), label="EDMD")
ax.legend()


Tmax = 3  # simulation length
Nsim = Tmax / dt

MODE = ["step", "cos"][1]
# 'step' or 'cos'

if MODE == "step":
    ymin = -0.6
    ymax = 0.6
    x0 = TSCDataFrame.from_array(np.array([[0, 0.6]]), feature_names=X_tsc.columns)

    values = (0.3 * (-1 + 2 * (np.arange(1, Nsim) > Nsim / 2)))[:, np.newaxis]
    reference = TSCDataFrame.from_array(
        values, time_values=np.arange(dt * 2, dt * (Nsim + 1), dt), feature_names=["x2"]
    )  # reference
elif MODE == "cos":
    ymin = -0.4
    ymax = 0.4
    x0 = TSCDataFrame.from_array(np.array([[-0.1, 0.1]]), feature_names=X_tsc.columns)
    values = (
        0.5 * np.cos(2 * np.pi * np.arange(1, Nsim) / Nsim)[:, np.newaxis]
    )  # reference
    reference = TSCDataFrame.from_array(
        values, time_values=np.arange(dt * 2, dt * (Nsim + 1), dt), feature_names=["x2"]
    )  # reference
else:
    raise RuntimeError("")

Q = 1
R = 0.01

Tpred = 1
Np = int(np.round(Tpred / dt))

kmpc = LinearKMPC(
    edmd=edmd,
    horizon=Np,
    input_bounds=np.array([[-1, 1]]),
    state_bounds=np.array([[ymin, ymax]]),
    qois=[select],
    cost_running=Q,
    cost_terminal=Q,
    cost_input=R,
)

U_eval = TSCDataFrame.from_array(
    np.zeros(edmd.n_samples_ic_ - 1), feature_names=U.columns
)
X_ic_predict, _ = sys.predict_vectorize(x0, U=U_eval, nsim=2, dt=dt)
X_state_evol = pd.concat(
    [X_ic_predict, shift_time_index_U(X_ic_predict, U_eval)], axis=1
).fillna(0)

for i in range(reference.shape[0] - 5):

    if np.mod(i, 10) == 0:
        print(f"{i} / {reference.shape[0]}")

    _ref = reference.iloc[i : i + Np, :]

    if _ref.shape[0] != Np:
        # TODO: move this to a function in TSCAccessor...
        n_attach = Np - _ref.shape[0]
        last_state, last_time = (
            _ref.iloc[[-1], :].to_numpy(),
            _ref.time_values()[-1] + dt,
        )
        attach_states = np.tile(last_state, (n_attach, 1))
        attach_time = np.arange(last_time, last_time + n_attach * dt - 1e-15, dt)
        tsc_attach = TSCDataFrame.from_array(
            attach_states, time_values=attach_time, feature_names=_ref.columns
        )
        _ref = pd.concat([_ref, tsc_attach], axis=0)

    ukmpc = kmpc.control_sequence(
        X=X_state_evol.iloc[-2:, :].loc[:, edmd.feature_names_in_], reference=_ref
    )
    U_eval = pd.concat([U_eval, ukmpc.iloc[[0], :]])

    state, _ = sys.predict_vectorize(
        X_state_evol.iloc[[-1], :].loc[:, ["x1", "x2"]],
        U=U_eval.iloc[[-1], :],
        nsim=2,
        dt=dt,
    )
    X_state_evol = pd.concat(
        [X_state_evol.loc[:, ["x1", "x2"]], state.iloc[[-1], :]], axis=0
    )
    X_state_evol = pd.concat(
        [X_state_evol, shift_time_index_U(X_state_evol, U_eval)], axis=1
    ).fillna(0)

f, ax = plt.subplots()
teval = X_state_evol.time_values()
ax.plot(teval, np.ones_like(teval) * ymin, c="black")
ax.plot(teval, np.ones_like(teval) * ymax, c="black")
ax.plot(teval, X_state_evol.loc[:, "x2"].to_numpy())
ax.plot(reference.time_values(), reference.to_numpy())
ax.set_ylim([ymin * 0.99, ymax * 1.01])
plt.show()
