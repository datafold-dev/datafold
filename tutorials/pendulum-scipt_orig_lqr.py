import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from datafold import DiffusionMaps, GaussianKernel, TSCTakensEmbedding, TSCApplyLambdas, TSCColumnTransformer
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
from tqdm import tqdm
from datafold.appfold.mpc import LinearKMPC, LQR
from datafold.utils._systems import InvertedPendulum, InvertedPendulum2

include_dmdcontrol = False

# TODO: include state bounds of CART properly in KMPC (they are essentially ignored currently)

# Data generation parameters
training_size = 20
time_values = np.arange(0, 10, 0.02)

# specify a seed
rng = np.random.default_rng(4)

invertedPendulum = InvertedPendulum2()

def generate_data(n_timeseries, ics):
    X_tsc, U_tsc = [], []  # lists to collect sampled time series
    for ic in tqdm(range(ics.shape[0])):
        for i in range(n_timeseries):
            control_fraction = 1
            control_amplitude = rng.uniform(0.1, 1)
            control_frequency = rng.uniform(np.pi / 2, 2 * np.pi)
            control_phase = rng.uniform(0, 2 * np.pi)
            Ufunc = lambda t, y: control_fraction * (
                control_amplitude * np.sin(control_frequency * t + control_phase)
            )

            # vals = rng.uniform(-1, 1, size=(len(time_values)))
            # vals = rng.normal(0, 1, size=(len(time_values)))
            # Ufunc = lambda t, y: np.interp(t, time_values, vals)

            # X are the states and U is the control input acting on the state's evolution
            X, U = invertedPendulum.predict(
                X=ics[ic, :],
                U=Ufunc,
                time_values=time_values,
                require_last_control_state=False
            )

            X_tsc.append(X)
            U_tsc.append(U)

    X_tsc = TSCDataFrame.from_frame_list(X_tsc)
    U_tsc = TSCDataFrame.from_frame_list(U_tsc)

    return X_tsc, U_tsc


# Turn lists into time series collection data structure

# Sample from a single initial condition (but use randomly sampled control signals below)

n_ics = 10
ics_train = np.zeros((n_ics, 4))

for i in range(n_ics):
    ics_train[i] = [0, 0, rng.uniform(-0.1, 0.1), 0]
# ics_train = np.array([[0, 0, np.pi + 0.1, 0], [0, 0, np.pi + 1.0, 0], [0, 0, 0.1, 0]])

ics_test = np.array([[0, 0, rng.uniform(-0.1, 0.1), 0]])

X_tsc, U_tsc = generate_data(training_size, ics_train)
X_oos, U_oos = generate_data(1, ics_test[[0]])

# X_tsc = pd.concat([X_tsc, U_tsc], axis=1)
# X_oos = pd.concat([X_oos, U_oos], axis=1)
#
# U_tsc = U_tsc.tsc.drop_last_n_samples(1)
# U_oos = U_oos.tsc.drop_last_n_samples(1)


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

# anim1 = InvertedPendulum().animate(X_last, U_last)
# anim2 = InvertedPendulum().animate(prediction, U_last)

# angle_replace = Degree2sincos(feature_names=["theta"])

n_rbf_centers = 1000
rbf = (
    "rbf",
    TSCRadialBasis(
        kernel=InverseQuadraticKernel(epsilon=1),
        center_type="random",
        n_samples=n_rbf_centers,
    ),
)

delays = 25
delay = ("delay", TSCTakensEmbedding(delays=delays))

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

ldas_sin = ("sin", TSCApplyLambdas(lambdas=[lambda x: np.sin(x)]), ["theta"])
ldas_cos = ("cos", TSCApplyLambdas(lambdas=[lambda x: np.cos(x)]), ["theta"])
ldas_cosdot = ("cos_dot", TSCApplyLambdas(lambdas=[lambda x: np.cos(x)]), ["thetadot"])
ldas_sindot = ("sin_dot", TSCApplyLambdas(lambdas=[lambda x: np.sin(x)]), ["thetadot"])
_id_ = ("id", TSCIdentity(include_const=True), lambda df: df.columns)
trigon = ("trigon", TSCColumnTransformer(transformers=[ldas_cos, ldas_sin, ldas_cosdot, ldas_sindot, _id_]))

_dict = [delay, rbf, _id]

edmd = EDMD(dict_steps=_dict, dmd_model=DMDControl(), include_id_state=False)

edmd.fit(
    X_tsc,
    U=U_tsc,
)

print("SAMPLE X_DICT")
print(edmd.transform(X_oos.head(edmd.n_samples_ic_ + 10)))
print("SAMPLE END")

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

const_values = np.zeros((edmd.n_samples_ic_, edmd.n_features_in_))
target_state = TSCDataFrame.from_array(const_values, time_values=np.arange(0, edmd.n_samples_ic_*edmd.dt_, edmd.dt_), feature_names=edmd.feature_names_in_)

horizon = 20

kmpc = LQR(edmd=edmd, cost_running=1, cost_input=0.01)
kmpc.preset_target_state(target_state=target_state)

X_ic = np.array([0, 0, rng.uniform(-0.1, 0.1), 0])  # TODO: make IC sampling function
n_mpc_steps = 100

time_values_mpc = np.arange(0, n_mpc_steps*edmd.dt_, edmd.dt_)

# fill with nans to check if something is not filled out..

U = TSCDataFrame.from_single_timeseries(pd.DataFrame(np.nan, index=time_values_mpc[:-1], columns=edmd.control_names_in_))
U.iloc[0:edmd.n_samples_ic_-1, :] = 0 # no control for initially

X_controlled = TSCDataFrame.from_single_timeseries(pd.DataFrame(np.nan, index=time_values_mpc[:-1], columns=invertedPendulum.feature_names_in_))
X_controlled.loc[(0, 0), :] = X_ic

if edmd.n_samples_ic_ > 1:
    X_warm_up, _ = invertedPendulum.predict(X_ic, U=U.iloc[0:edmd.n_samples_ic_ - 1, :])
    X_controlled.loc[X_warm_up.index] = X_warm_up


for i in tqdm(range(edmd.n_samples_ic_-1, n_mpc_steps)):
    time_values_horizon = time_values_mpc[i+1:i+horizon+1]

    if len(time_values_horizon) < horizon:
        break

    ukmpc = kmpc.control_sequence(
        X=X_controlled.iloc[i-edmd.n_samples_ic_+1:i+1, :]
    )

    control_action = ukmpc.iloc[[0], :]
    print(control_action)
    t_now, t_next = time_values_mpc[i:i + 2]

    U.loc[control_action.index] = control_action

    _X_next, _ = invertedPendulum.predict(X_controlled.loc[[(0, t_now)], :], U=control_action, time_values=np.array([t_now, t_next]))
    X_controlled.loc[[(0, t_next)], :] = _X_next.iloc[[-1], :]

# X_reference_no_control, _ = invertedPendulum.predict(X_ic, U=TSCDataFrame.from_single_timeseries(pd.DataFrame(0, index=time_values_mpc[:-1], columns=invertedPendulum.control_names_in_)))

anim = invertedPendulum.animate(X_controlled, U=U)
# anim = invertedPendulum.animate(X_reference_no_control, None)

plt.show()
exit()

# anim = invertedPendulum.animate(kmpctraj, None)

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
    reference_no_control.time_values(),
    reference_no_control["theta"].to_numpy(),
    linestyle="--",
    label="no control",
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
plt.plot(ukmpc.time_values(), ukmpc.iloc[:, 0], label="controller")
plt.legend()

print("successful")
plt.show()
