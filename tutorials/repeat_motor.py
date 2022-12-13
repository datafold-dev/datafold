import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy
from datafold.utils._systems import Motor

import pandas as pd

from datafold import EDMD, DMDControl, InitialCondition, TSCDataFrame, TSCRadialBasis, TSCTakensEmbedding, TSCColumnTransformer, TSCIdentity
from datafold.dynfold.transform import TSCFeatureSelect
from datafold.pcfold.kernels import ThinplateKernel

dt = 0.01
Ntraj = 200
Nsim = 1000

nD = 1
ny = 1
rng = np.random.default_rng(554)

time_values = np.arange(0, Nsim * dt, dt)


def shift_time_index_U(_X, _U):
    # TODO: create an accessor function to accomblish this?
    new_index = _X.groupby("ID").tail(_X.n_timesteps - 1).index
    return _U.set_index(new_index)


def load_matlab_data():
    C = scipy.io.loadmat("file.mat")
    X, Y, U = C["X"], C["Y"], C["U"]

    X_tsc = TSCDataFrame.from_shift_matrices(
        X, Y, snapshot_orientation="col", columns=["x1", "x2", "x3"]
    )

    U_tsc = TSCDataFrame(
        U.T,
        index=pd.MultiIndex.from_arrays(
            [X_tsc.ids, np.zeros(X_tsc.n_timeseries, dtype=int)]
        ),
        columns=["u"],
    )

    return X_tsc, U_tsc


# sampling
own_impl = True
if own_impl:
    sys = Motor(ivp_kwargs={"atol": 1E-2, "rtol": 1E-5, "method": "RK23"})

    X_ic = InitialCondition.from_array(
        rng.uniform(size=(Ntraj, sys.n_features_in_)) * 2 - 1,
        time_value=0.0,
        feature_names=sys.feature_names_in_,
    )

    U = 2 * rng.uniform(size=(Ntraj, Nsim - 1, 1)) - 1
    U = TSCDataFrame.from_tensor(
        U, time_values=time_values[:-1], columns=sys.control_names_in_
    )

    X_ic_xtest = TSCDataFrame.from_array(np.array([-0.5332, -0.9401]), feature_names=X_ic.columns)
    U_ic_xtest = TSCDataFrame.from_array(np.array([0.2400]), feature_names=U.columns)
    sys.predict_vectorize(X_ic_xtest, U_ic_xtest, nsim=1, dt=dt)


    select = "x2"

    X_tsc, U_tsc = sys.predict_vectorize(X_ic, U=U, nsim=Nsim, dt=dt)
    X_tsc = pd.concat([X_tsc.loc[:, [select]], shift_time_index_U(X_tsc, U_tsc)], axis=1).fillna(0)

    delay = ("delay", TSCTakensEmbedding(delays=1), [select])
    _id = ("id", TSCIdentity(), ["u"])

    t1 = ("t1", TSCColumnTransformer(transformers=[delay, _id], verbose_feature_names_out=False))
    rbf = ("rbf", TSCRadialBasis(kernel=ThinplateKernel(), center_type="fit_params"))
    edmd = EDMD([t1, rbf], dmd_model=DMDControl(), include_id_state=True)
else:
    X_tsc, U_tsc = load_matlab_data()
    rbf = ("rbf", TSCRadialBasis(kernel=ThinplateKernel(), center_type="fit_params"))

    edmd = EDMD([rbf], dmd_model=DMDControl(), include_id_state=True)

print("start fitting")
centers = rng.uniform(size=(100, 3)) * 2 - 1
edmd = edmd.fit(X_tsc, U=U_tsc, rbf__centers=centers)
print("successful")

Tmax = 1
Nsim = int(Tmax / dt)


def myprbs(N, duty_cycle):
    cond = rng.uniform(0, 1, size=(N, 1)) > (1-duty_cycle)
    return cond.astype(float)


U_test = TSCDataFrame.from_array(2*myprbs(Nsim, 0.5) - 1, time_values=np.arange(0, Nsim*dt, dt), feature_names=U_tsc.columns)  # uprbs

X_ic_orig = TSCDataFrame.from_array(rng.uniform(0, 1, size=(1,2))-0.5, feature_names=X_tsc.columns)
U_ic_edmd = TSCDataFrame.from_array(2*rng.uniform(0,1, size=(1,1)) - 1, feature_names=U_tsc.columns)
X_ic_edmd, U_ic_edmd = sys.predict_vectorize(X_ic_orig, U=U_ic_edmd, dt=dt, nsim=2)
X_ic_edmd = pd.concat([X_ic_edmd.loc[:, [select]], shift_time_index_U(X_ic_edmd, U_ic_edmd)], axis=1).fillna(0)

X_true_test, _ = sys.predict_vectorize(X_ic_orig, U=U_test, dt=dt, nsim=U_test.shape[0])

X_edmd_pred = edmd.predict(X_ic_edmd, U=U_test)

f, ax = plt.subplots()
ax.set_title("control input")
ax.set_xlabel("time [s]")
ax.set_ylabel("U")
ax.step(U_test.time_values(), U_test.to_numpy())

f, ax = plt.subplots()
ax.plot(X_true_test.time_values(), X_true_test.loc[:, select].to_numpy(), label="original")
ax.plot(X_edmd_pred.time_values(), X_edmd_pred.loc[:, select].to_numpy(), label="EDMD")
ax.legend()
plt.show()


Tmax = 3 # Simlation legth
Nsim = Tmax/dt

MODE = 'step'; # 'step' or 'cos'

if MODE == "step":
    ymin = -0.6
    ymax = 0.6
    x0 = [0, 0.6]
    yrr = 0.3*( -1 + 2*(np.arange(1, Nsim) > Nsim/2)) # reference
elif MODE == 'cos':
    ymin = -0.4
    ymax = 0.4
    x0 = [-0.1, 0.1]
    yrr = 0.5*np.cos(2*np.pi*np.arange(1, Nsim) / Nsim) # reference
else:
    raise RuntimeError("")
