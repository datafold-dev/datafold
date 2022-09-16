#!/usr/bin/env python
# coding: utf-8

# # Model-Based Control using Koopman Operators
# 
# The Koopman operator $\mathcal{K}$ is an infinite dimensional linear operator that directly acts on the functions of state 
# $$
#     \mathcal{K}g = g \circ F,
# $$
# where $\circ$ is the composition operator such that
# $$
#     \mathcal{K}g(x(t_i)) = g(F(x(t_i))) = g(x(t_{i+1})).
# $$
# That is, the Koopman operator $\mathcal{K}$ takes *any* observation of state $g(x(t_i))$ at time $t_i$ and evolves the function of state subject to its dynamics forward in time *linearly*. Thus, a nonlinear dynamical system can be represented by a linear Koopman operator in a lifted function space where the observations of state evolve linearly. The contribution in robotics is that now a highly nonlinear robotic system can be represented as a linear dynamical system which contains all the nonlinear information (as opposed to a Taylor expansion of the dynamics centered at an equilibrium state).
# 

# ## Computing the Koopman operator with Control inputs
# The Koopman operator $\mathcal{K}$ is infeasible to compute in the infinite dimensional space. A finite subspace approximation to the operator $\mathfrak{K} \in \mathbb{R}^c \times \mathbb{R}^c$ acting on $\mathcal{C} \subset \mathbb{C}$ is used instead. Here, we define a subset of function observables (or observations of state)
# $z(x) : \mathbb{R}^n \in \mathbb{R}^c \subset \mathcal{C}$. The output dimension of $z(x)$ will now define the finite subspace that spance $\mathcal{C} \subset \mathbb{C}$.
# The operator $\mathfrak{K}$ acting on $z(x(t_i))$ is then represented in discrete time as
# $$
#     z(x(t_{i+1})) = \mathfrak{K} z(x(t_i)) + r(x(t_i))
# $$
# where $r(x) \in \mathbb{C}$ is the residual function error. Adding control to the Koopman operator representation is done by augmenting the state $x(t)$ with the input control to the system $u(t)$ where now the function observables become $z(x, u)$ and the approximate Koopman operator is given by 
# $$
#     z(x(t_{i+1}), u(t_{i+1})) = \mathfrak{K} z(x(t_i), u(t_i)) + r(x(t_i), u(t_i))
# $$
# In principle, as $c \to \infty$, the residual error goes to zero; however, it is sometimes possible to find $c < \infty$ such that $r(x) = 0$. This is known as a Koopman invariant subspace, but for our purposes we won't go into too much detail.
# 
# The continuous time Koopman operator is calculated by taking the matrix logarithm as $t_{i+1} - t_i \to 0$ where we overload the notation for the Koopman operator:
# $$
# \dot{z}(x(t), u(t)) = \mathfrak{K} z(x(t), u(t)) + r(x(t), u(t)).
# $$
# 
# We are first going to write a class for the Van der Pol dynamical system which has the following differential equation:
# $$
#     \frac{d}{dt} \begin{bmatrix} x_1 \\ x_2  \end{bmatrix} = \begin{bmatrix} x_2 \\ -x_1 + \epsilon (1 - x_1^2)x_2 + u \end{bmatrix}
# $$

# In[2]:

import autograd.numpy as np
import pandas as pd
from autograd import jacobian
from scipy.linalg import logm
import scipy.linalg
from datafold.utils._systems import VanDerPol as VDP2
from datafold import TSCDataFrame, EDMD
from datafold.dynfold.dmd import DMDControl
rng = np.random.default_rng(5)




# For the Koopman operator system, we are going to use the following basis functions:

# In[3]:




num_trials = 10  ## number of resets
horizon = 200  ## how long we simulate the system for
M = num_trials * horizon  ## M sized data


vdp2 = VDP2(n_control=1)

time_values = np.arange(0, horizon*0.01, 0.01)

X = rng.uniform(-3., 3., size=(num_trials, 2))

idx = pd.MultiIndex.from_arrays([np.arange(num_trials), np.zeros(num_trials)])
X_ic = TSCDataFrame(X, index=idx, columns=["x1", "x2"])

U_tsc = rng.uniform(-3., 3., size=(num_trials, 1, 1))
U_tsc = np.tile(U_tsc, (1, horizon-1, 1))
U_tsc = TSCDataFrame.from_tensor(U_tsc, time_series_ids=X_ic.ids, columns=vdp2.control_names_in_, time_values=time_values[:-1])

X_tsc, U_tsc = vdp2.predict(X_ic, U=U_tsc)

# time_values = np.arange(0, 1E-15+horizon*0.01, 0.01)
# X_tsc = TSCDataFrame.from_tensor(states, columns=["x1", "x2"], time_values=time_values)
# U_tsc = TSCDataFrame.from_tensor(control, columns=["u1", "u2"], time_values=time_values)
# U_tsc = U_tsc.loc[:, ["u1"]].tsc.drop_last_n_samples(1)

import matplotlib.pyplot as plt

for i in X_tsc.ids:
    idx = pd.IndexSlice[i, :]
    plt.plot(X_tsc.loc[idx, "x1"].to_numpy(), X_tsc.loc[idx, "x2"].to_numpy())


# dmdc = DMDControl().fit(X_tsc, U=U_tsc)

# dictionary
# [x[0], x[1], x[0]**2, (x[0]**2)*x[1]

from sklearn.base import BaseEstimator
from datafold import TSCTransformerMixin

class VdPDictionary(BaseEstimator, TSCTransformerMixin):
    def get_feature_names_out(self, input_features=None):
        return ["x1^2", "x1^2 * x2"]

    def fit(self, X, y=None):
        self._setup_feature_attrs_fit(X)
        return self

    def transform(self, X: TSCDataFrame):
        X = X.copy()
        X["x1^2"] = np.square(X.loc[:, "x1"].to_numpy())
        X["x1^2 * x2"] = X["x1^2"].to_numpy() * X["x2"].to_numpy()
        return X.drop(["x1", "x2"], axis=1)


vdp_dictionary = ("dict", VdPDictionary())

edmd = EDMD(dict_steps=[vdp_dictionary], dmd_model=DMDControl(), include_id_state=True)
edmd.fit(X_tsc, U=U_tsc)

Q = np.diag([1, 1, 0., 0.])
R = np.diag([1.0]) * 1e-2

Ad = edmd.dmd_model.sys_matrix_
Bd = edmd.dmd_model.control_matrix_
Pd = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R) * X_tsc.delta_time
Flqr = np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad


# In[8]:

from datafold import InitialCondition

X_ic_oos = rng.uniform(-3, 3, size=(1,2))
X_ic_oos = InitialCondition.from_array(X_ic_oos, feature_names=edmd.feature_names_in_, time_value=0)

target_point = InitialCondition.from_array(np.array([0, 0]), feature_names=edmd.feature_names_in_, time_value=0)
target_point = edmd.transform(target_point).to_numpy()

horizon = 500 # simulation time

time_values_oos = np.arange(0, -1E-15+horizon*X_tsc.delta_time, X_tsc.delta_time)

# objects to fill in following loop
trajectory = TSCDataFrame.from_array(np.zeros((horizon, 2)), feature_names=vdp2.feature_names_in_, time_values=time_values_oos)
u_discrete = TSCDataFrame.from_array(np.zeros((horizon-1, 1)), feature_names=vdp2.control_names_in_, time_values=time_values_oos[:-1])

trajectory.iloc[0, :] = X_ic_oos.to_numpy()

# state = trajectory_cont[0]
# vdp.state = state
# for t in range(1, horizon+1):
#     u = - Flqr @ (z(state, u_def)[:num_x_obs] - target_z)
#     u_discrete[t-1] = u
#     state = vdp.step(u / vdp.dt)
#     trajectory_discrete[t, :] = state

for i in range(1, horizon):
    print(i)
    state = trajectory.iloc[[i-1], :]
    u_discrete.iloc[i-1, :] = - Flqr @ (edmd.transform(state).to_numpy() - target_point).T

    new_state, _ = vdp2.predict(state, U=u_discrete.iloc[[i-1], :], time_values=time_values_oos[i-1:i+1])
    trajectory.iloc[i, :] = new_state.iloc[[1], :].to_numpy()

print("got here")
trajectory_uncontrolled, _ = vdp2.predict(X_ic_oos, U=np.zeros((horizon-1)), time_values=time_values_oos)

# Here we visualize the resulting trajectory from applying a model-based controller using the Koopman operator representation of the dynamical system.

# In[9]:
plt.figure()
plt.plot(trajectory.to_numpy(), c="black", label="discrete")
plt.xlabel('t')
plt.legend()
plt.ylabel('x1, x2')

plt.figure()
plt.plot(trajectory.loc[:, "x1"].to_numpy(), trajectory.loc[:, "x2"].to_numpy(), c="red")
plt.quiver(*trajectory.to_numpy()[:-1, :].T, *np.column_stack([np.zeros_like(u_discrete.to_numpy()), u_discrete.to_numpy()]).T, color="blue")
plt.plot(trajectory.iloc[0, 0], trajectory.iloc[0, 1], "o", c="red")
plt.plot(trajectory_uncontrolled.loc[:, "x1"].to_numpy(), trajectory_uncontrolled.loc[:, "x2"].to_numpy(), c="black")
plt.plot(trajectory_uncontrolled.iloc[0, 0], trajectory_uncontrolled.iloc[0, 1], "o", c="black")
plt.plot(target_point[0, 0], target_point[0, 1], "*", c="black")
plt.title("discrete case")
plt.xlabel('x1')
plt.ylabel('x2')

plt.figure()
plt.plot(np.linalg.norm(trajectory.to_numpy(), axis=1))
plt.axhline(np.linalg.norm(target_point[:2]), c="red")
plt.xlabel('t')
plt.ylabel('norm (discrete')
plt.show()
