
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datafold import TSCDataFrame
from datafold.utils._systems import Burger

# simulates the setting from https://arxiv.org/pdf/1804.05291.pdf

rng = np.random.default_rng(2)

time_values = np.linspace(0, 2, 100)

umin, umax = (-0.1, 0.1)
plot = False

f1 = lambda x: np.atleast_2d(np.exp(-(15 * (x - 0.25)) ** 2))
f2 = lambda x: np.atleast_2d(np.exp(-(15 * (x - 0.75)) ** 2))

training_size = 2

sys = Burger()

X_tsc = []

for i in range(training_size):
    a = rng.uniform(0, 1)
    ic1 = np.exp(-(((sys.x_nodes) - .5) * 5) ** 2)
    ic2 = np.sin(4 * np.pi * sys.x_nodes) ** 2
    ic = a * ic1 + (1 - a) * ic2

    rand_vals = rng.uniform(umin, umax, size=(len(time_values), 2))
    # rand_vals = np.zeros((len(time_values), 2))
    U1rand = lambda t: np.atleast_2d(np.interp(t, time_values, rand_vals[:, 0])).T
    U2rand = lambda t: np.atleast_2d(np.interp(t, time_values, rand_vals[:, 1])).T

    def U(t, x):
        if x.shape[1] == 1:
            x = x.T
        return U1rand(t) * f1(x) + U2rand(t) * f2(x)

    X_predict, _ = sys.predict(ic, U=U, time_values=time_values)

    X_tsc.append(X_predict)

X_tsc = TSCDataFrame.from_frame_list(X_tsc)

print(X_tsc.head(10))


# TODO: subselect measurements in Koopman operator (every 10th)
# TODO: add time delay embedding (5 delays) of BOTH state and control
# TODO: add constant function to observables
# TODO: add L2 norm as a observable (altogether there should be 60)
# TODO: use LinearKMPC the controller is used for a rather short time period only (0.1 seconds)
# TODO: measure output error (state-reference) over time!



if True:
    values = X_tsc.loc[pd.IndexSlice[0, :], :].to_numpy()

    f = plt.figure()
    (line,) = plt.plot(sys.x_nodes, ic)


    def func(i):
        line.set_data(sys.x_nodes, values[i, :])
        return (line,)

    anim = FuncAnimation(f, func=func, frames=values.shape[0])
    plt.show()
