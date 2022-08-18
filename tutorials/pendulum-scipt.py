import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from datafold import (
    EDMD,
    DMDControl,
    InverseQuadraticKernel,
    TSCDataFrame,
    TSCIdentity,
    TSCPrincipalComponent,
    TSCRadialBasis,
    TSCTransformerMixin,
)
from datafold.appfold.kmpc import AffineKgMPC, LinearKMPC
from datafold.utils._systems import InvertedPendulum

include_dmdcontrol = False

# Data generation parameters
training_size = 20
time_values = np.arange(0, 10, 0.05)

# Sample from a single initial condition (but use randomly sampled control signals below)
ics = np.array([[0, 0, np.pi + 0.1, 0], [0, 0, np.pi + 1.0, 0], [0, 0, 0.1, 0]])

X_tsc, U_tsc = [], []  # lists to collect sampled time series

# specify a seed
rng = np.random.default_rng(2)

for ic in range(ics.shape[0]):
    for i in range(training_size):
        control_fraction = 1
        control_amplitude = rng.uniform(0.1, 1)
        control_frequency = rng.uniform(np.pi / 2, 2 * np.pi)
        control_phase = rng.uniform(0, 2 * np.pi)
        Ufunc = lambda t, y: control_fraction * (
            control_amplitude * np.sin(control_frequency * t + control_phase)
        )

        invertedPendulum = InvertedPendulum(pendulum_mass=1)

        # X are the states and U is the control input acting on the state's evolution
        X, U = invertedPendulum.predict(
            X=ics[ic, :],
            Ufunc=Ufunc,
            time_values=time_values,
        )

        X_tsc.append(X)
        U_tsc.append(U)


# Turn lists into time series collection data structure
X_tsc = TSCDataFrame.from_frame_list(X_tsc)
U_tsc = TSCDataFrame.from_frame_list(U_tsc)

print(f"{X_tsc.shape[0]=}")
print(f"{X_tsc.n_timesteps=}")

X_tsc_theta = X_tsc.copy()
# X_tsc = InvertedPendulum.theta_to_trigonometric(X_tsc)
# X_back = InvertedPendulum.trigonometric_to_theta(X_tsc)
# X_theta_req = np.mod(X_tsc_theta["theta"].to_numpy(), 2*np.pi)
# min_diff = np.min(X_back["theta"].to_numpy() - X_theta_req)
# max_diff = np.max(X_back["theta"].to_numpy() - X_theta_req)
# print(f"{min_diff=} -- {max_diff=}")

plt.figure(figsize=(16, 7))

plt.subplot(311)
plt.title(
    r"Training data - cart position $\theta$ $\mathbf{x}$ (each time series is separated by verical lines)"
)
plt.plot(X_tsc_theta["theta"].to_numpy())
[plt.axvline(i, c="black") for i in np.arange(0, X_tsc.shape[0], X_tsc.n_timesteps)]
plt.xticks([])


plt.subplot(312)
plt.title(
    r"Training data - control voltage $\mathbf{u}$ (each time series is separated by verical lines)"
)

# plt.plot(InvertedPendulum.trigonometric_to_theta(X_tsc)["theta"].to_numpy())
# [plt.axvline(i, c="black") for i in np.arange(0, X_tsc.shape[0], X_tsc.n_timesteps)]
# plt.xticks([])

plt.subplot(313)
plt.plot(U_tsc["u"].to_numpy())
[plt.axvline(i, c="black") for i in np.arange(0, U_tsc.shape[0], U_tsc.n_timesteps)]
plt.xticks([])

last_id = X_tsc.ids[-5]

X_last = X_tsc.loc[[last_id], :]
U_last = U_tsc.loc[[last_id], :]

if include_dmdcontrol:
    dmdc = DMDControl()
    dmdc.fit(X_tsc, U=U_tsc)
    prediction = dmdc.predict(X_last.head(1), U=U_last)

    plt.figure(figsize=(16, 3))
    plt.subplot(121)
    plt.title(r"Linear DMD prediction - cart position $x$")
    plt.plot(time_values, prediction["x"].to_numpy(), label="prediction")
    plt.plot(time_values, X_last["x"].to_numpy(), label="actual")
    plt.legend()

    plt.subplot(122)
    plt.title(r"EDMD(100 random rbf) prediction - pendulum angle $\theta$")
    plt.plot(
        time_values,
        InvertedPendulum.trigonometric_to_theta(prediction)["theta"].to_numpy(),
        label="prediction",
    )
    plt.plot(
        time_values,
        InvertedPendulum.trigonometric_to_theta(X_last)["theta"].to_numpy(),
        label="actual",
    )
    plt.legend()

# anim1 = InvertedPendulum().animate(X_last, U_last)
# anim2 = InvertedPendulum().animate(prediction, U_last)

# angle_replace = Degree2sincos(feature_names=["theta"])

n_rbf_centers = 1000
eps = 3

rbf = (
    "rbf",
    TSCRadialBasis(
        kernel=InverseQuadraticKernel(epsilon=eps),
        center_type="random",
        n_samples=n_rbf_centers,
    ),
)

from datafold import DiffusionMaps, GaussianKernel, TSCTakensEmbedding

delays = 35
delay = ("delay", TSCTakensEmbedding(delays=delays))
# transform_U = TSCTakensEmbedding(delays=delays)
transform_U = TSCIdentity()

U_tsc = transform_U.fit_transform(U_tsc)

dmap = (
    "dmap",
    DiffusionMaps(
        kernel=GaussianKernel(epsilon=lambda d: np.median(d) / 3), n_eigenpairs=70
    ),
)

_id = ("id", TSCIdentity())
pca = ("pca", TSCPrincipalComponent(n_components=144))

_dict = [delay]

edmd = EDMD(dict_steps=_dict, dmd_model=DMDControl(), include_id_state=False)

edmd.fit(
    X_tsc,
    U=U_tsc,
)


rbfprediction = edmd.predict(
    X_last.head(edmd.n_samples_ic_), U=transform_U.transform(U_last)
)

plt.figure(figsize=(16, 3))
plt.subplot(121)
plt.title(r"EDMD(100 random rbf) prediction - cart position $x$")
plt.plot(
    rbfprediction.time_values(),
    rbfprediction["x"].to_numpy(),
    c="red",
    label="prediction",
)
plt.plot(
    rbfprediction.time_values()[0],
    rbfprediction["x"].to_numpy()[0],
    c="red",
    marker="o",
)
plt.plot(time_values, X_last["x"].to_numpy(), c="black", label="actual")
plt.legend()

plt.subplot(122)
plt.title(r"EDMD(100 random rbf) prediction - pendulum angle $\theta$")
plt.plot(
    rbfprediction.time_values(),
    rbfprediction["theta"].to_numpy(),
    c="red",
    label="prediction",
)
plt.plot(
    rbfprediction.time_values()[0],
    rbfprediction["theta"].to_numpy()[0],
    c="red",
    marker="o",
)
plt.plot(time_values, X_last["theta"].to_numpy(), c="black", label="actual")
plt.legend()

# anim3 = InvertedPendulum().animate(X_last, U_last)
# anim4 = InvertedPendulum().animate(rbfprediction, U_last)


horizon = 100  # in time steps

kmpc = LinearKMPC(
    predictor=edmd,
    horizon=horizon,
    state_bounds=np.array([[1, -1], [6.28, 0]]),
    input_bounds=np.array([[5, -5]]),
    qois=["x", "theta"],
    cost_running=np.array([100, 0]),
    cost_terminal=1,
    cost_input=0.001,
)

plt.show()

print("successful")
