import numpy as np
import scipy
import scipy.io

from datafold.utils._systems import Motor

dt = 0.01
Ntraj = 200
Nsim = 1000

nD = 1
ny = 1
rng = np.random.default_rng(1)

time_values = np.arange(0, Nsim * dt, dt)

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
from datafold.dynfold.transform import TSCFeatureSelect
from datafold.pcfold.kernels import ThinplateKernel


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


def shift_time_index_U(_X, _U):
    # TODO: create an accessor function to accomblish this?
    new_index = _X.groupby("ID").tail(_X.n_timesteps - 1).index
    return _U.set_index(new_index)


# sampling
own_impl = True
if own_impl:
    sys = Motor(ivp_kwargs={"atol": 1e-2, "rtol": 1e-5, "method": "RK23"})

    X_ic = InitialCondition.from_array(
        rng.uniform(size=(Ntraj, sys.n_features_in_)) * 2 - 1,
        time_value=0.0,
        feature_names=sys.feature_names_in_,
    )

    U = 2 * rng.uniform(size=(Ntraj, Nsim - 1, 1)) - 1
    U = TSCDataFrame.from_tensor(
        U, time_values=time_values[:-1], columns=sys.control_names_in_
    )

    X_tsc, U_tsc = sys.predict_vectorize(X_ic, U=U, nsim=Nsim, dt=dt)
    X_tsc = X_tsc.loc[:, ["x1"]]
    X_tsc = pd.concat([X_tsc, shift_time_index_U(X_tsc, U_tsc)], axis=1).fillna(0)

    delay = ("delay", TSCTakensEmbedding(delays=1), ["x1"])
    _id = ("id", TSCIdentity(), ["u"])

    t1 = (
        "t1",
        TSCColumnTransformer(
            transformers=[delay, _id], verbose_feature_names_out=False
        ),
    )
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
Nsim = Tmax / dt


def myprbs(N, duty_cycle):
    cond = rng.uniform(0, 1, size=(N, 1)) > 1 - duty_cycle
    return cond.astype(float)


uprbs = 2 * myprbs(Nsim, 0.5) - 1
u_dt = lambda i: uprbs[i + 1]

X_ic = rng.uniform(0, 1, size=(1, 2)) - 0.5
U_ic = 2 * rng.uniform(0, 1, size=(2, 1)) - 1
sys.predict_vectorize(X_ic, U=U_ic, nsim=3)
