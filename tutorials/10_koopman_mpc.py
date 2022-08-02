#!/usr/bin/env python
# coding: utf-8

# (Visit the
# [documentation](https://datafold-dev.gitlab.io/datafold/tutorial_index.html) page
# to view the executed notebook.)
# 
# # Koopman Operator for Model Predictive Control
# In this tutorial we introduce the usage of the Dynamic Mode Decomposition (DMD) method with controlled systems. Furthermore, we demonstrate how the Extended-DMD can be applied to model predictive control.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# In[2]:


from datafold import EDMD, DMDControl, gDMDAffine, TSCIdentity, TSCRadialBasis, TSCDataFrame, InverseQuadraticKernel

from datafold.appfold.kmpc import AffineKgMPC, LinearKMPC
from datafold.utils._systems import InvertedPendulum


# ## Data generation
# 
# ### Inverted pendulum physics
# A test model implemented in datafold is used to generate example data in this tutorial. A pendulum and a moving cart are connected by a swivel, which allows the pendulum to freely rotate. The cart wheels spin on a rail, and the entire system is powered by a DC motor. The displacement of the cart $x$ and the angular rotation of the pendulum $\theta$ describe the movement of the pendulum (note that the time derivatives, $\dot x$ and $\dot \theta$, are also needed to describe the full state), and the voltage to the motor $u$ can be controlled.
# 
# ### Creating the training set
# The training set will consists of 20 different trajectories of 10 seconds each, discretized at 0.01 s time steps. The initial conditions of all trajectories is the same, and the control input is a sinusoidal signal of varying amplitude, frequency and phase. The 20 timeseries are concatenated into datafold's data structure for handling time series collections, `TSCDataFrame`.

# In[3]:


# Data generation parameters
training_size = 20
time_values = np.arange(0, 10+1E-15, 0.01)

# Sample from a single initial condition
ic = np.array([0, 0, 0, 0])


# In[4]:


Xlist, Ulist = [], []

# specify a seedtraining_size = 20
#rng = np.random.default_rng(42)

np.random.seed(42)

for i in range(training_size):
    #control_amplitude = 0.1 + 0.9 * rng.uniform(0, 1)
    #control_frequency = np.pi + 2 * np.pi * rng.uniform(0, 1)
    #control_phase = 2 * np.pi * rng.uniform(0, 1)
    #control_func = lambda t, y: control_amplitude * np.sin(
    #    control_frequency * t + control_phase
    #)
    
    control_amplitude = 0.1 + 0.9 * np.random.random()
    control_frequency = np.pi + 2 * np.pi * np.random.random()
    control_phase = 2 * np.pi * np.random.random()
    U_func = lambda t, y: control_amplitude * np.sin(
        control_frequency * t + control_phase
    )
    
    invertedPendulum = InvertedPendulum()
    
    # X are the states and U is the control input acting on the state's evolution
    X, U = invertedPendulum.predict(
        X=ic,
        U=U_func,
        time_values=time_values,
    )

    Xlist.append(X)
    Ulist.append(U)

X_tsc = TSCDataFrame.from_frame_list(Xlist)
U_tsc = TSCDataFrame.from_frame_list(Ulist);


# We can look at the sampled time series data (`X_tsc`) on the state space and the control (`U_tsc`) that acted on the system evolution. We store the last time series as a system reference to use throughout the tutorial.

# In[5]:


X_last = X_tsc.loc[[19], :]
X_tsc


# In[6]:


U_last = U_tsc.loc[[19], :]
U_tsc


# In[7]:


# TODO: remove x-ticks and insert vertical lines to not confuse...
plt.figure(figsize=(16, 7))

plt.subplot(211)
plt.title(r"Training data - cart position $x$")
plt.plot(X_tsc["theta"].to_numpy())

plt.subplot(212)
plt.title(r"Training data - control voltage $u$")
plt.plot(U_tsc["u"].to_numpy());


# ## Identify the system 
# 
# ### Linear control DMD
# In this section we demonstrate how to use the `DMDControl` class to create a Dynamic Mode Decomposition predictor for controlled systems using the data above.
# 
# The `DMDControl` class implements the `TSCPredictMixin` which is based on the `scikit-learn` estimator-style interface. After we initialize the model, we can use the `.fit` method to train the model. Internally, the `DMDControl` computes two matrices $A$ and $B$, which best satisfy $\mathbf{x}_{k+1} = A\mathbf{x}_k + B\mathbf{u}_k$ ($\mathbf{x}$ refering to the state vector of the system and $\mathbf{u}$ to the control input). This model assumes that the system is linear in both the state and  control.  

# In[8]:


dmdc = DMDControl()
dmdc.fit(X_tsc, U=U_tsc);


# In[9]:


plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title("state matrix $A$")
plt.imshow(dmdc.sys_matrix_)
plt.colorbar()

plt.subplot(122)
plt.title("control matrix $B$")
plt.imshow(dmdc.control_matrix_)
plt.colorbar();


# The `.predict` method can be used to estimate the trajectory of the system from a given initial condition and for a given control input.

# In[10]:


prediction = dmdc.predict(ic, U=U_last)


# In[16]:


plt.figure(figsize=(16, 3))

plt.subplot(121)
plt.title(r"Linear DMD prediction - cart position $x$");
plt.plot(time_values, prediction["x"].to_numpy(), label="prediction")
plt.plot(time_values, X["x"].to_numpy(), label="actual")
plt.legend();

plt.subplot(122)
plt.title(r"EDMD(100 random rbf) prediction - pendulum angle $\theta$");
plt.plot(time_values, prediction["theta"].to_numpy(), label="prediction")
plt.plot(time_values, X_last["theta"].to_numpy(), label="actual")
plt.legend();


# ### Using DMDControl within the Extended Dynamic Mode Decomposition (EDMD) model framework
# 
# A generic `EDMD` class is specified with
# 
# * `dict_steps` (dictionary): corresponding to a set of functions with the goal to approximately linearize the system dynamics
# * `dmd_model` (final estimator): corresponding to a DMD model to performs a linear system identification in the lifted space
# 
# (we also highlight the dedicated `EDMD` tutorial)
# 
# For the last parameter, it is also possible to set a DMD variant with control input (such as `DMDControl`) and hence use `EDMD` for controlled systems.
# 
# Here we use a dictionary of 100 randomly selected radial basis functions (RBF) specified with an inverse quadratic kernel. The functions are centered at random positions on the state space. 
# 
# #### Select RBF centers

# In[32]:


n_rbf_centers = 200
eps = 1

rbf = TSCRadialBasis(
    kernel=InverseQuadraticKernel(epsilon=eps), center_type="fit_params"
)
center_ids = sorted(
    np.random.choice(
        range(0, 1000 * training_size), size=n_rbf_centers, replace=False
    )
)
centers = X_tsc.iloc[center_ids].to_numpy()


# #### Set up `EDMD` with RBF dictionary and `DMDControl`
# 
# Note that using `TSCRadialBasis` transformer with `center_type='fit_params'` requires passing the `centers` parameter to the `.fit` method of transformer using the `fit_params` as detailed in the documentation.

# In[33]:


edmdrbf = EDMD(
    dict_steps=[("rbf", rbf)], 
    dmd_model=DMDControl(), 
    include_id_state=True
)

edmdrbf.fit(
    X_tsc,
    U=U_tsc,
    rbf__centers=centers,
)

rbfprediction = edmdrbf.predict(ic, U=U_last)


# In[34]:


plt.figure(figsize=(16, 3))
plt.subplot(121)
plt.title(r"EDMD(100 random rbf) prediction - cart position $x$")
plt.plot(time_values, rbfprediction["x"].to_numpy(), label="prediction")
plt.plot(time_values, X_last["x"].to_numpy(), label="actual")
plt.legend()

plt.subplot(122)
plt.title(r"EDMD(100 random rbf) prediction - pendulum angle $\theta$");
plt.plot(time_values, rbfprediction["theta"].to_numpy(), label="prediction")
plt.plot(time_values, X_last["theta"].to_numpy(), label="actual")
plt.legend();


# In the figure above we can see the the EDMD using the linear DMD estimator predicts the displacement even better than the DMD, but the prediction for $\theta$ is good only on a short time scale.

# ## Koopman Model Predictive Control
# 
# Model Predictive Control (MPC) is a method for estimating a control signal to provide to a system to achieve certain behaviour in the future. 
# 
# An important parameter for MPC is the prediction horizon, which denotes how far in the future is the control signal computed. Here we set it to 100 timesteps (corresponding to one second). The optimality of the signal is computed using a cost function of the referenced states (`qois`) and the control signal itself. Here we provide both $\mathbf{x}$ and $\theta$ as referenced states, but set the running cost weight of $\theta$ to 0 to showcase the functionality of the interface. Additionaly, the cost function enables penalizing control inputs. To match the test data as well as possible, this is set to a small value.
# 
# #### Linear MPC
# Here we show an implementation of a Koopman MPC (KMPC) where the model part is based on the Koopman operator. The `LinearKMPC` class implements such a controller based on `EDMD` (as used above with `dmd_model=DMDControl()`). The key benefit is that the model is a linear system in the lifted space and the optimal control can be directly computed using a quadratic programming optimizer.

# In[35]:


horizon = 100  # in time steps

kmpc = LinearKMPC(
    predictor=edmdrbf,
    horizon=horizon,
    state_bounds=np.array([[1, -1], [6.28, 0]]),
    input_bounds=np.array([[5, -5]]),
    qois=["x", "theta"],
    cost_running=np.array([100, 0]),
    cost_terminal=1,
    cost_input=0.001,
)


# To generate the control signal, a full initial state is required, in addition to a reference to track. Here we use a reference produced by a known control signal in the training data, to be able to later compare it to the optimal control signal computed by the controller.

# In[36]:


reference = X_last[["x", "theta"]].iloc[: horizon + 1]
ukmpc = kmpc.generate_control_signal(rbfprediction.initial_states(), reference)


# Further more, we compute what trajectory the model has predicted based on the control signal computed by the controller and what is the real response of the system.

# In[37]:


kmpcpred = edmdrbf.predict(rbfprediction.initial_states(), U=ukmpc)

ukmpc_interp = interp1d(
    rbfprediction.time_values()[:horizon],
    ukmpc,
    axis=0,
    fill_value=0,
    bounds_error=False,
)
ukmpc_func = lambda t, y: ukmpc_interp(t).T
kmpctraj, _ = invertedPendulum.predict(
    X=np.array([0,0,0,0]),
    U=ukmpc_func,
    time_values=time_values,
)


# In[ ]:


kmpctraj


# We now plot the data that we obtained from the `KoopmanMPC` class and compare the predicted trajectories for both position $x$ and angle $\theta$. In addition, we also compare the estimated control signal from the `KoopmanMPC` class and compare it with the the reference control signal above.

# In[ ]:


plt.figure(figsize=(16, 3.5))

plt.subplot(131)
plt.title(r"Comparison : Cart Position $x$")
plt.plot(time_values[:horizon+1], X_last["x"].to_numpy()[:horizon+1], label="reference")
plt.plot(time_values[:horizon+1], kmpcpred["x"].to_numpy()[:horizon+1], label="prediction")
plt.plot(time_values[:horizon+1], kmpctraj["x"].to_numpy()[:horizon+1], label="actual")
plt.legend()

plt.subplot(132)
plt.title(r"Comparison : Pendulum Angle $\theta$")
plt.plot(time_values[:horizon+1], X_last["theta"].to_numpy()[:horizon+1], label="reference")
plt.plot(time_values[:horizon+1], kmpcpred["theta"].to_numpy()[:horizon+1], label="prediction")
plt.plot(time_values[:horizon+1], kmpctraj["theta"].to_numpy()[:horizon+1], label="actual")
plt.legend()

plt.subplot(133)
plt.title(r"Comparison : Control Signal $u$");
plt.plot(time_values[:horizon+1], U_last.to_numpy()[:horizon+1], label="correct")
plt.plot(time_values[:horizon], ukmpc, label="controller")
plt.legend();


# #### Affine KMPC
# 
# Similarly to above, the `AffineKgMPC` class models the system based on `EDMDControl` class used with `dmd_model=gDMDAffine()`. This class however is much less efficient, since not only is the prediction slower, but the structure of the affine systems requires a slower quasi-Newton optimizer. Some differences in the interface are also present:
# 
# 1. The `AffineKgMPC` class does not support applying bounds on the state, only on the input
# 
# 1. Due to the above, the `qoi` parameter is not implemented independently. The effect of tracking only a subset of the state can be achieved by setting `cost_state` of the untracked parameters to 0.
# 
# 1. Unlike in the `LinearKMPC` implementation, the cost of the state is the same over the whole horizon, and is not split between `cost_running` and `cost_terminal`
# 
# 1. Time steps at which to generate the control signal are required, either as index if `reference` is a `TSCDataFrame`, or as a separte array as shown. Those can be again non-uniform.

# In[ ]:


horizon = 100

akgmpc = AffineKgMPC(
    predictor=egdmdarbf,
    horizon=horizon,
    input_bounds=np.array([[5, -5]]),
    cost_state=np.array([1, 0, 0, 0]),
    cost_input=0.001,
)


# In[ ]:


reference = X_tsc.loc[[19], :]

# TODO: why is generate_control_signal reference transposed?
reference_values = reference.to_numpy()[: horizon + 1, :].T

time_values_horizon = reference.time_values()[: horizon + 1]

ukmpc = akgmpc.generate_control_signal(
    rbfprediction.initial_states(),
    reference_values,
    time_values=time_values_horizon,
)


# In[ ]:


# TODO: this needs to get fixed
ukmpc.shape == time_values_horizon.shape


# In[ ]:


kmpcpred = egdmdarbf.predict(rbfprediction.initial_states(), U=ukmpc)

ukmpc_interp = interp1d(
    rbfprediction.time_values()[:horizon],
    ukmpc,
    axis=0,
    fill_value=0,
    bounds_error=False,
)
ukmpc_func = lambda t, y: ukmpc_interp(t).T

kmpctraj, _ = invertedPendulum.predict(
    X=ic,
    U=ukmpc_func,
    time_step=X_tsc.delta_time,
    num_steps=horizon,
    
)
kmpctraj


# In[ ]:


plt.figure(figsize=(16, 3.5))

plt.subplot(131)
plt.title(r"Comparison : Cart Position $x$")
plt.plot(X_tsc.loc[[19], "x"].to_numpy()[: horizon + 1], label="reference")
plt.plot(kmpcpred["x"].to_numpy(), label="prediction")
plt.plot(kmpctraj["x"].to_numpy(), label="actual")
plt.legend()

plt.subplot(132)
plt.title(r"Comparison : Pendulum Angle $\theta$")
plt.plot(X_tsc.loc[[19], "theta"].to_numpy()[: horizon + 1], label="reference")
plt.plot(kmpcpred["theta"].to_numpy(), label="prediction")
plt.plot(kmpctraj["theta"].to_numpy(), label="actual")
plt.legend()

plt.subplot(133)
plt.title(r"Comparison : Control Signal $u$")
plt.plot(U_tsc.loc[[19], "u"].to_numpy()[: horizon + 1], label="correct")
plt.plot(ukmpc, label="controller")
plt.legend();


# In[ ]:




