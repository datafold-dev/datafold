
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
U_tsc = []

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

    X_predict, Ufull = sys.predict(ic, U=U, time_values=time_values, require_last_control_state=True)

    # drop last control input, as it is not required
    U = TSCDataFrame.from_same_indices_as(Ufull, rand_vals, except_columns=["u1", "u2"])

    X_tsc.append(X_predict)
    U_tsc.append(U)

X_tsc = TSCDataFrame.from_frame_list(X_tsc)
U_tsc = TSCDataFrame.from_frame_list(U_tsc)

print(f"{X_tsc.head(5)}")
print(f"{U_tsc.head(5)}")

# subselect measurements to every 10th node
X_tsc = X_tsc.iloc[:, 0::10]

X_tsc = pd.concat([X_tsc, U_tsc], axis=1)

from sklearn.base import BaseEstimator
from datafold import TSCTakensEmbedding, TSCIdentity, TSCColumnTransformer, EDMD, DMDControl, TSCTransformerMixin

class L2Norm(BaseEstimator, TSCTransformerMixin):
    def fit(self, X):
        return self

    def get_feature_names_out(self, input_features=None):
        return ["l2norm"]

    def transform(self, X: TSCDataFrame, y=None):
        return TSCDataFrame.from_same_indices_as(X, np.linalg.norm(X.to_numpy(), axis=1), except_columns=self.get_feature_names_out())


l2norm = ("l2", L2Norm(), lambda df: df.columns.str.startswith("x"))
_id = ("id", "passthrough", lambda df: df.columns)
l2norm = ("l2norm", TSCColumnTransformer([l2norm, _id], verbose_feature_names_out=False))

delay1 = ("delay_x", TSCTakensEmbedding(delays=4), lambda df: df.columns.str.startswith("x"))
delay2 = ("delay_u", TSCTakensEmbedding(delays=3), lambda df: df.columns.str.startswith("u"))
_l2norm = ("l2norm", "passthrough", ["l2norm"])
tde = ("tde", TSCColumnTransformer([delay1, delay2, _l2norm], verbose_feature_names_out=False))

_id = ("_id", TSCIdentity(include_const=True))

edmd = EDMD([l2norm, tde, _id], dmd_model=DMDControl(), include_id_state=False)

# import tempfile
# from sklearn.utils import estimator_html_repr
# import webbrowser
# with tempfile.NamedTemporaryFile("w", suffix=".html") as fp:
#     fp.write(estimator_html_repr(tde[1]))
#     fp.flush()
#     webbrowser.open_new_tab(fp.name)
#     input("Press Enter to continue...")
# exit()

# The last sample is not required (as there is no prediction from the last state)
U_tsc = U_tsc.tsc.drop_last_n_samples(1)

edmd.fit(X_tsc, U=U_tsc)
X_dict = edmd.transform(X_tsc.head(10))

print(f"{X_dict.columns=}")
print(f"{len(X_dict.columns)=}")
print(X_dict.head(5))

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
