import numpy as np

from datafold import TSCDataFrame
from datafold.utils._systems import InvertedPendulum

# Data generation parameters
training_size = 1
time_values = np.arange(0, 10 + 1e-15, 0.01)

# Sample from a single initial condition (but use randomly sampled control signals below)
ic = np.array([0, 0, np.pi / 2, 0])

X_tsc, U_tsc = [], []  # lists to collect sampled time series

# specify a seedtraining_size = 20
rng = np.random.default_rng(1)

for i in range(training_size):

    control_amplitude = 0.1 + 0.9 * np.random.random()
    control_frequency = np.pi + 2 * np.pi * np.random.random()
    control_phase = 2 * np.pi * np.random.random()
    Ufunc = lambda t, y: control_amplitude * np.sin(
        control_frequency * t + control_phase
    )

    invertedPendulum = InvertedPendulum()

    # X are the states and U is the control input acting on the state's evolution
    X, U = invertedPendulum.predict(
        X=ic,
        Ufunc=Ufunc,
        time_values=time_values,
    )

    X_tsc.append(X)
    U_tsc.append(U)

# Turn lists into time series collection data structure
X_tsc = TSCDataFrame.from_frame_list(X_tsc)
U_tsc = TSCDataFrame.from_frame_list(U_tsc)

invertedPendulum.animate(X_tsc, U_tsc)

print(X_tsc)
# print(U_tsc)
