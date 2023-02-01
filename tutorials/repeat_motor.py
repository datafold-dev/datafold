import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from datafold.utils._systems import DCMotor

# System
sys = DCMotor()
output = "x2"

# define sampling parameter
rng = np.random.default_rng(1)

dt = 0.01
n_timeseries = 200
n_timesteps = 1000


from scipy.interpolate import interp1d
interp1d([0, .5, 1.5, 2, 2.5], [-.3, .4, -.6, -.2, .2], kind="previous")


time_values = np.arange(0, n_timesteps * dt, dt)

# A function to required to augment the state with control input
def shift_time_index_U(_X, _U):
    new_index = _X.groupby("ID").tail(_X.n_timesteps - 1).index
    return _U.set_index(new_index)

X_ic = InitialCondition.from_array(
    rng.uniform(size=(n_timeseries, sys.n_features_in_)) * 2 - 1,
    time_value=0.0,
    feature_names=sys.feature_names_in_,
)

U = TSCDataFrame.from_tensor(
    rng.uniform(size=(n_timeseries, n_timesteps - 1, 1), low=-1, high=1),
    time_values=time_values[:-1],
    columns=sys.control_names_in_
)

X_ic_xtest = TSCDataFrame.from_array(
    np.array([-0.5332, -0.9401]), feature_names=X_ic.columns
)
U_ic_xtest = TSCDataFrame.from_array(np.array([0.2400]), feature_names=U.columns)

X_tsc, U_tsc = sys.predict_vectorize(X_ic, U=U)
X_tsc = X_tsc.loc[:, [output]].tsc.augment_control_input(U_tsc).fillna(0)

# use the columns transformer to apply the time delay embedding on the output variable (x2) and the identity on the control input
delay = ("delayembedding", TSCColumnTransformer(transformers=[("delay", TSCTakensEmbedding(delays=1), [output]), ("id", TSCIdentity(), ["u"])], verbose_feature_names_out=False),)
rbf = ("rbf", TSCRadialBasis(kernel=ThinplateKernel(), center_type="fit_params"))
rbf_centers = rng.uniform(size=(100, 3)) * 2 - 1
edmd = EDMD([delay, rbf], dmd_model=DMDControl(), include_id_state=True)

# Fit EDMD model
edmd = edmd.fit(X_tsc, U=U_tsc, rbf__centers=rbf_centers)

# Specify a test sampling to see the predictive power of EDMD
time_horizon = 1
n_timesteps = int(time_horizon / dt)

# TODO: need to rename this, why is it there?
def myprbs(N, duty_cycle):
    cond = rng.uniform(0, 1, size=(N, 1)) > (1 - duty_cycle)
    return cond.astype(float)


def random_control(duty_cycle):
    # generates random control input with values -1 or 1
    # the duty cycle influences which value shows up more often
    cond = rng.uniform(0, 1, size=(n_timesteps, 1)) > (1 - duty_cycle)
    return 2 * cond.astype(float) - 1

# control input for the test time series
U_test = TSCDataFrame.from_array(
    2 * myprbs(n_timesteps, 0.5) - 1,
    time_values=np.arange(0, n_timesteps * dt, dt),
    feature_names=U_tsc.columns,
)

# initial condition
X_ic_oos = TSCDataFrame.from_array(
    rng.uniform(0, 1, size=(1, 2)) - 0.5, feature_names=X_tsc.columns[:2]
)

# evaluate the true model
X_model_pred, _ = sys.predict_vectorize(X_ic_oos, U=U_test)

# create the initial condition for EDMD -- because of the delay embedding we need
X_edmd = X_model_pred.iloc[:2, :].loc[:, [output]]
U_edmd = U_test.iloc[[0], :]
X_ic_edmd = pd.concat([X_edmd, shift_time_index_U(X_edmd, U_edmd)], axis=1).fillna(0)
X_edmd_pred = edmd.predict(X_ic_edmd, U=U_test)

f, ax = plt.subplots()
ax.set_title("control input")
ax.set_xlabel("time [s]")
ax.set_ylabel("U")
ax.step(U_test.time_values(), U_test.to_numpy())

f, ax = plt.subplots()
ax.plot(X_model_pred.time_values(), X_model_pred.loc[:, output].to_numpy(), label="original")
ax.plot(X_edmd_pred.time_values(), X_edmd_pred.loc[:, output].to_numpy(), label="EDMD")
ax.legend()

time_horizon = 3  # simulation length
n_timesteps = time_horizon / dt

MODE = ["step", "cos"][1]
# 'step' or 'cos'

if MODE == "step":
    ymin = -0.6
    ymax = 0.6
    x0 = TSCDataFrame.from_array(np.array([[0, 0.6]]), feature_names=X_tsc.columns)

    values = (0.3 * (-1 + 2 * (np.arange(1, n_timesteps) > n_timesteps / 2)))[:, np.newaxis]
    reference = TSCDataFrame.from_array(
        values, time_values=np.arange(dt * 2, dt * (n_timesteps + 1), dt), feature_names=["x2"]
    )  # reference
elif MODE == "cos":
    ymin = -0.4
    ymax = 0.4
    x0 = TSCDataFrame.from_array(np.array([[-0.1, 0.1]]), feature_names=X_tsc.columns)
    values = (
        0.5 * np.cos(2 * np.pi * np.arange(1, n_timesteps) / n_timesteps)[:, np.newaxis]
    )  # reference
    reference = TSCDataFrame.from_array(
        values, time_values=np.arange(dt * 2, dt * (n_timesteps + 1), dt), feature_names=["x2"]
    )  # reference
else:
    raise RuntimeError("")

Q = 1
R = 0.01

mpc_horizon = 1
mpc_n_timesteps = int(np.round(mpc_horizon / dt))

kmpc = LinearKMPC(
    edmd=edmd,
    horizon=mpc_n_timesteps,
    input_bounds=np.array([[-1, 1]]),
    state_bounds=np.array([[ymin, ymax]]),
    qois=[output],
    cost_running=Q,
    cost_terminal=Q,
    cost_input=R,
)

# 1.
U_seq = TSCDataFrame.from_array(0.0, feature_names=U.columns, time_values=0.0)

sys_ic, _ = sys.predict_vectorize(x0, U=U_seq, time_values=[0, dt])
X_ic = sys_ic.copy().tsc.augment_control_input(U_seq).fillna(0)
X_ic = X_ic.loc[:, edmd.feature_names_in_]

X_seq, _ = kmpc.control_system(sys=sys.predict_vectorize, sys_ic=sys_ic, X_ref=reference, X_ic=X_ic, augment_control=True)

f, ax = plt.subplots()
teval = X_seq.time_values()
ax.plot(teval, np.ones_like(teval) * ymin, c="black")
ax.plot(teval, np.ones_like(teval) * ymax, c="black")
ax.plot(teval, X_seq.loc[:, [output]].to_numpy())
ax.plot(reference.time_values(), reference.to_numpy())
ax.set_ylim([ymin * 0.99, ymax * 1.01])
plt.show()
