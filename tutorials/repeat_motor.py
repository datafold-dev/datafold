import numpy as np

from datafold.utils._systems import Motor

dt = 0.01
Ntraj = 10
Nsim = 1000

nD = 1
ny = 1
rng = np.random.default_rng(1)

time_values = np.arange(0, Nsim*dt, dt)

from datafold import InitialCondition, TSCDataFrame, TSCRadialBasis, EDMD, DMDControl
from datafold.pcfold.kernels import Thinplate
import pandas as pd

# sampling
own_impl = False
if own_impl:
    sys = Motor()

    X_ic = InitialCondition.from_array(rng.uniform(size=(Ntraj, sys.n_features_in_))*2 - 1, time_value=0., feature_names=sys.feature_names_in_)

    U = 2*rng.uniform(size=(Ntraj, Nsim-1, 1))-1
    U = TSCDataFrame.from_tensor(U, time_values=time_values[:-1], columns=sys.control_names_in_)

    print("solve")
    X_pred, _ = sys.predict(X_ic, U=U)
    print("finish")
    print(X_pred)
else:
    import scipy
    C = scipy.io.loadmat("file.mat")
    X, Y, U = C["X"], C["Y"], C["U"]
    X_tsc = TSCDataFrame.from_shift_matrices(X, Y, snapshot_orientation="col", columns=["x1", "x2", "x3"])
    U_tsc = TSCDataFrame(U.T, index=pd.MultiIndex.from_arrays([X_tsc.ids, np.zeros(X_tsc.n_timeseries, dtype=int)]), columns=["u"])


rbf = ("rbf", TSCRadialBasis(kernel=Thinplate(), center_type="fit_params"))
edmd = EDMD([rbf], dmd_model=DMDControl(), include_id_state=True)

print("start fitting")
centers = rng.uniform(size=(100, 3))*2 - 1
edmd = edmd.fit(X_tsc, U=U_tsc, rbf__centers=centers)
print("successful")

# X_ic = rng.uniform((1,2)) - 0.5

for i in range(100):
    pass
