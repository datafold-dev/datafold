import matplotlib.pyplot as plt
import numpy as np

from datafold import TSCDataFrame
from datafold.utils._systems import InvertedPendulum

# Data generation parameters
training_size = 20
time_values = np.arange(0, 10 + 1e-15, 0.01)

Xlist, Ulist = [], []

# specify a seedtraining_size = 20
rng = np.random.default_rng(42)

for i in range(training_size):
    control_amplitude = 0.1 + 0.9 * rng.uniform(0, 1)
    control_frequency = np.pi + 2 * np.pi * rng.uniform(0, 1)
    control_phase = 2 * np.pi * rng.uniform(0, 1)
    control_func = lambda t, y: control_amplitude * np.sin(
        control_frequency * t + control_phase
    )
    invertedPendulum = InvertedPendulum()
    X, U = invertedPendulum.predict(
        # Sample from a single initial condition
        X=np.array([0, 0, 0, 0]),
        U=control_func,
        time_values=time_values,
    )

    Xlist.append(X)
    Ulist.append(U)

X_tsc = TSCDataFrame.from_frame_list(Xlist)
U_tsc = TSCDataFrame.from_frame_list(Ulist)

X_tsc


print(X_tsc.delta_time)
print(X_tsc)

X_tsc.loc[:, ["x", "xdot"]].iloc[:100, :].plot()
plt.show()
