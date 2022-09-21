import numpy as np
from datafold.utils._systems import Motor

dt = 0.01
Ntraj = 10
Nsim = 1000

nD = 1
ny = 1

time_values = np.arange(0, Nsim*dt, dt)

# sampling

sys = Motor()

rng = np.random.default_rng(1)

from datafold import InitialCondition, TSCDataFrame

X_ic = InitialCondition.from_array(rng.uniform(size=(Ntraj, sys.n_features_in_))*2 - 1, time_value=0., feature_names=sys.feature_names_in_)

U = 2*rng.uniform(size=(Ntraj, Nsim-1, 1))-1
U = TSCDataFrame.from_tensor(U, time_values=time_values[:-1], columns=sys.control_names_in_)

print("solve")
X_pred, _ = sys.predict(X_ic, U=U)
print("finish")
print(X_pred)
