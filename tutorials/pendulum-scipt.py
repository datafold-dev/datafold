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
from datafold.appfold.mpc import LinearKMPC
from datafold.utils._systems import InvertedPendulum

include_dmdcontrol = False

# Data generation parameters
training_size = 5
time_values = np.arange(0, 13, 0.01)

# specify a seed
rng = np.random.default_rng(4)

invertedPendulum = InvertedPendulum(pendulum_mass=2)


def generate_data(n_timeseries, ics):
    X_tsc, U_tsc = [], []  # lists to collect sampled time series
    for ic in range(ics.shape[0]):
        for i in range(n_timeseries):
            control_fraction = 1
            control_amplitude = rng.uniform(0.1, 1)
            control_frequency = rng.uniform(np.pi / 2, 2 * np.pi)
            control_phase = rng.uniform(0, 2 * np.pi)
            Ufunc = lambda t, y: control_fraction * (
                control_amplitude * np.sin(control_frequency * t + control_phase)
            )

            # X are the states and U is the control input acting on the state's evolution
            X, U = invertedPendulum.predict(
                X=ics[ic, :],
                U=Ufunc,
                time_values=time_values,
            )

            X_tsc.append(X)
            U_tsc.append(U)

    X_tsc = TSCDataFrame.from_frame_list(X_tsc)
    U_tsc = TSCDataFrame.from_frame_list(U_tsc)

    return X_tsc, U_tsc


# Turn lists into time series collection data structure

# Sample from a single initial condition (but use randomly sampled control signals below)

ics_train = np.zeros((4, 4))

for i in range(4):
    ics_train[i] = [0, 0, np.pi + rng.uniform(-1, 1), 0]
# ics_train = np.array([[0, 0, np.pi + 0.1, 0], [0, 0, np.pi + 1.0, 0], [0, 0, 0.1, 0]])

ics_test = np.array([[0, 0, np.pi + rng.uniform(-1, 1), 0]])

X_tsc, U_tsc = generate_data(training_size, ics_train)
X_oos, U_oos = generate_data(1, ics_test[[0]])

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

# last_id = X_tsc.ids[2]
# X_oos = X_tsc.loc[[last_id], :]
# U_oos = U_tsc.loc[[last_id], :]

if include_dmdcontrol:
    dmdc = DMDControl()
    dmdc.fit(X_tsc, U=U_tsc)
    prediction = dmdc.predict(X_oos.head(1), U=U_oos)

    plt.figure(figsize=(16, 3))
    plt.subplot(121)
    plt.title(r"Linear DMD prediction - cart position $x$")
    plt.plot(time_values, prediction["x"].to_numpy(), label="prediction")
    plt.plot(time_values, X_oos["x"].to_numpy(), label="actual")
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
        InvertedPendulum.trigonometric_to_theta(X_oos)["theta"].to_numpy(),
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

delays = 100
delay = ("delay", TSCTakensEmbedding(delays=delays))
# transform_U = TSCTakensEmbedding(delays=delays)
transform_U = TSCIdentity()

U_tsc = transform_U.fit_transform(U_tsc)

dmap = (
    "dmap",
    DiffusionMaps(
        kernel=GaussianKernel(epsilon=lambda d: np.median(d) / 2), n_eigenpairs=100
    ),
)

_id = ("id", TSCIdentity(include_const=True))

pca = ("pca", TSCPrincipalComponent(n_components=60))

_dict = [delay, pca, _id]

edmd = EDMD(dict_steps=_dict, dmd_model=DMDControl(), include_id_state=False)

edmd.fit(
    X_tsc,
    U=U_tsc,
)

edmdpredict = edmd.predict(
    X_oos.head(edmd.n_samples_ic_), U=transform_U.transform(U_oos)
)

plt.figure(figsize=(16, 3))
plt.subplot(121)
plt.title(r"EDMD(100 random rbf) prediction - cart position $x$")
plt.plot(
    edmdpredict.time_values(),
    edmdpredict["x"].to_numpy(),
    c="red",
    label="prediction",
)
plt.plot(
    edmdpredict.time_values()[0],
    edmdpredict["x"].to_numpy()[0],
    c="red",
    marker="o",
)
plt.plot(time_values, X_oos["x"].to_numpy(), c="black", label="actual")
plt.legend()

plt.subplot(122)
plt.title(r"EDMD(100 random rbf) prediction - pendulum angle $\theta$")
plt.plot(
    edmdpredict.time_values(),
    edmdpredict["theta"].to_numpy(),
    c="red",
    label="prediction",
)
plt.plot(
    edmdpredict.time_values()[0],
    edmdpredict["theta"].to_numpy()[0],
    c="red",
    marker="o",
)
plt.plot(time_values, X_oos["theta"].to_numpy(), c="black", label="actual")
plt.legend()

# anim3 = InvertedPendulum().animate(X_last, U_last)
# anim4 = InvertedPendulum().animate(rbfprediction, U_last)

horizon = 300  # in time steps

kmpc = LinearKMPC(
    predictor=edmd,
    horizon=horizon,
    state_bounds=np.array([[-5, 5], [0, 6.28]]),
    input_bounds=np.array([[-99, 99]]),
    qois=["x", "theta"],
    cost_running=np.array([100, 0]),
    cost_terminal=0.1,
    cost_input=0.1,
)

reference = X_oos[["x", "theta"]].iloc[
    edmd.n_samples_ic_ - 1 : edmd.n_samples_ic_ + horizon
]
reference_u = U_oos[["u"]].iloc[edmd.n_samples_ic_ - 1 : edmd.n_samples_ic_ + horizon]
# reference = TSCDataFrame.from_same_indices_as(reference, values=np.zeros_like(reference.to_numpy()))

ukmpc = kmpc.optimal_control_sequence(
    X_oos.initial_states(edmd.n_samples_ic_), reference
)

kmpcpred = edmd.predict(X_oos.initial_states(edmd.n_samples_ic_), U=ukmpc)

ukmpc_interp = interp1d(
    kmpcpred.time_values()[:-1],
    ukmpc[:, 0],
    axis=0,
    fill_value=0,
    bounds_error=False,
)

ukmpc_func = lambda t, y: ukmpc_interp(t).T
kmpctraj, _ = invertedPendulum.predict(
    X=X_oos.initial_states(edmd.n_samples_ic_).to_numpy()[-1, :],
    U=ukmpc_func,
    time_values=kmpcpred.time_values()[:horizon],
)


plt.figure(figsize=(16, 3.5))

plt.subplot(131)
plt.title(r"Comparison : Cart Position $x$")
plt.plot(reference.time_values(), reference["x"].to_numpy(), label="reference")
plt.plot(
    kmpcpred.time_values(),
    kmpcpred["x"].to_numpy(),
    label="prediction",
)
plt.plot(kmpctraj.time_values(), kmpctraj["x"].to_numpy(), label="actual")
plt.legend()

plt.subplot(132)
plt.title(r"Comparison : Pendulum Angle $\theta$")
plt.plot(
    reference.time_values(),
    reference["theta"].to_numpy(),
    label="reference",
)
plt.plot(
    kmpcpred.time_values(),
    kmpcpred["theta"].to_numpy(),
    label="prediction",
)
plt.plot(
    kmpctraj.time_values(),
    kmpctraj["theta"].to_numpy(),
    label="actual",
)
plt.legend()

plt.subplot(133)
plt.title(r"Comparison : Control Signal $u$")
plt.plot(reference_u.time_values(), reference_u.to_numpy(), label="correct")
plt.plot(reference_u.time_values()[:-1], ukmpc[:, 0], label="controller")
plt.legend()

plt.show()

print("successful")
