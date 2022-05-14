import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd

from datafold.appfold.edmd import EDMDControl
from datafold.appfold.kmpc import AffineKgMPC, LinearKMPC
from datafold.dynfold.dmd import (
    ControlledAffineDynamicalSystem,
    ControlledLinearDynamicalSystem,
    gDMDAffine,
)
from datafold.dynfold.transform import TSCIdentity
from datafold.pcfold import InitialCondition, TSCDataFrame
from datafold.utils._systems import InvertedPendulum


class LinearKMPCTest(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(LinearKMPCTest, self).__init__(*args, **kwargs)
        self._state_columns = ["x", "xdot", "theta", "thetadot"]
        self._control_columns = ["u"]
        self.X, self.t, self.u, self.dfx, self.dfu = self._generate_data()

    def _generate_data(self):

        # Simulation model parameters

        dt = 0.01
        num_steps = 1000
        size = 20
        model = InvertedPendulum()
        np.random.seed(42)

        # Data structures

        Xlist = []
        Ulist = []

        for _ in range(size):
            model.reset()
            control_amplitude = 0.1 + 0.9 * np.random.random()
            control_frequency = np.pi + 2 * np.pi * np.random.random()
            control_phase = 2 * np.pi * np.random.random()
            control_function = lambda t, y: control_amplitude * np.sin(
                control_frequency * t + control_phase
            )

            trajectory = model.predict(
                time_step=dt,
                num_steps=num_steps,
                control_func=control_function,
            )
            assert (
                model.sol.success
            ), f"Divergent solution for amplitude={control_amplitude}, frequency={control_frequency}"
            t = model.sol.t
            dfx = pd.DataFrame(data=trajectory.T, index=t, columns=self._state_columns)
            dfx["u"] = 0.0
            Xlist.append(dfx)

            control_input = control_function(t, trajectory)
            dfu = pd.DataFrame(data=control_input, index=t, columns=("u",))
            for col in self._state_columns:
                dfu[col] = 0.0
            dfu = dfu[self._state_columns + self._control_columns]
            Ulist.append(dfu)

            X_tsc = TSCDataFrame.from_frame_list(Xlist)[self._state_columns]
            X_tsc[self._control_columns] = TSCDataFrame.from_frame_list(Ulist)[
                self._control_columns
            ]
            X_tsc

            return X_tsc, t, control_input, dfx, dfu

    def _execute_mock_test(self, state_size, input_size, n_timesteps, seed=42):
        gen = np.random.default_rng(seed)

        A = gen.uniform(size=(state_size, state_size)) * 2 - 1.0
        x0 = gen.uniform(size=state_size)
        B = gen.uniform(size=(state_size, input_size)) * 2 - 1.0
        t = np.linspace(0, n_timesteps - 1, n_timesteps)
        u = gen.uniform(size=(1, input_size))
        u = (u.T + np.sin(u.T + u.T * np.atleast_2d(t))).T
        df = (
            ControlledLinearDynamicalSystem()
            .setup_matrix_system(A, B)
            .evolve_system(x0, u)
        )
        from unittest.mock import Mock

        edmdmock = Mock()
        edmdmock.sys_matrix = A
        edmdmock.control_matrix = B
        edmdmock.state_columns = [f"x{i}" for i in range(state_size)]
        edmdmock.control_columns = [f"u{i}" for i in range(input_size)]
        edmdmock.transform = lambda x: x

        kmpcperfect = LinearKMPC(
            predictor=edmdmock,
            horizon=n_timesteps - 1,
            state_bounds=np.array([[5, -5]] * state_size),
            input_bounds=np.array([[5, -5]] * input_size),
            cost_running=np.array([1] * state_size),
            cost_terminal=1,
            cost_input=0,
        )
        pred = kmpcperfect.generate_control_signal(x0, df)
        nptest.assert_allclose(pred, u[:-1, :])

    def test_kmpc_mock_edmd_1d(self):
        self._execute_mock_test(2, 1, 50)

    def test_kmpc_mock_edmd_2d(self):
        self._execute_mock_test(2, 2, 50)

    def test_kmpc_generate_control_signal(self):
        horizon = 100

        edmdcontrol = EDMDControl(
            dict_steps=[
                ("id", TSCIdentity()),
            ],
            include_id_state=False,
        ).fit(
            self.X,
            split_by="name",
            state=self._state_columns,
            control=self._control_columns,
        )

        kmpc = LinearKMPC(
            predictor=edmdcontrol,
            horizon=horizon,
            state_bounds=np.array([[1, -1], [6.28, 0]]),
            input_bounds=np.array([[5, -5]]),
            qois=["x", "theta"],
            cost_running=np.array([100, 0]),
            cost_terminal=1,
            cost_input=1,
        )

        reference = self.dfx[["x", "theta"]].iloc[: horizon + 1]
        initial_conditions = InitialCondition.from_array(
            np.array([0, 0, np.pi, 0]), columns=["x", "xdot", "theta", "thetadot"]
        )
        U = kmpc.generate_control_signal(
            initial_conditions=initial_conditions, reference=reference
        )

        assert U.any() is not None
        assert U.shape == (horizon, len(self._control_columns))


class AffineKMPCTest(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(AffineKMPCTest, self).__init__(*args, **kwargs)
        self._state_columns = ["x", "xdot", "theta", "thetadot"]
        self._control_columns = ["u"]
        self.X, self.t, self.u, self.dfx, self.dfu = self._generate_data()

    def _generate_data(self):

        # Simulation model parameters

        dt = 0.01
        num_steps = 1000
        size = 20
        model = InvertedPendulum()
        np.random.seed(42)

        # Data structures

        Xlist = []
        Ulist = []

        for _ in range(size):
            model.reset()
            control_amplitude = 0.1 + 0.9 * np.random.random()
            control_frequency = np.pi + 2 * np.pi * np.random.random()
            control_phase = 2 * np.pi * np.random.random()
            control_function = lambda t, y: control_amplitude * np.sin(
                control_frequency * t + control_phase
            )

            trajectory = model.predict(
                time_step=dt,
                num_steps=num_steps,
                control_func=control_function,
            )
            assert (
                model.sol.success
            ), f"Divergent solution for amplitude={control_amplitude}, frequency={control_frequency}"
            t = model.sol.t
            dfx = pd.DataFrame(data=trajectory.T, index=t, columns=self._state_columns)
            dfx["u"] = 0.0
            Xlist.append(dfx)

            control_input = control_function(t, trajectory)
            dfu = pd.DataFrame(data=control_input, index=t, columns=("u",))
            for col in self._state_columns:
                dfu[col] = 0.0
            dfu = dfu[self._state_columns + self._control_columns]
            Ulist.append(dfu)

            X_tsc = TSCDataFrame.from_frame_list(Xlist)[self._state_columns]
            X_tsc[self._control_columns] = TSCDataFrame.from_frame_list(Ulist)[
                self._control_columns
            ]
            X_tsc

            return X_tsc, t, control_input, dfx, dfu

    def _execute_mock_test(
        self, state_size, input_size, n_timesteps, AffineMPCtype, seed=42
    ):
        gen = np.random.default_rng(seed)
        x0 = gen.uniform(size=state_size)
        A = gen.uniform(-0.4, 0.5, size=(state_size, state_size))
        np.fill_diagonal(A, gen.uniform(-0.6, -0.5, size=state_size))
        Bi = np.stack(
            [gen.uniform(size=(state_size, state_size)) for i in range(input_size)],
            2,
        )

        t = np.linspace(0, n_timesteps - 1, n_timesteps) * 0.1
        u = 0.5 + gen.uniform(-0.1, 0.1, size=(1, input_size))
        u = (u.T + np.sin(u.T + u.T * np.atleast_2d(t))).T
        sys = ControlledAffineDynamicalSystem().setup_matrix_system(A, Bi)
        df = sys.evolve_system(x0, u, t)

        from unittest.mock import Mock

        edmdmock = Mock()
        edmdmock.sys_matrix = A
        edmdmock.control_matrix = Bi
        edmdmock.state_columns = [f"x{i}" for i in range(state_size)]
        edmdmock.control_columns = [f"u{i}" for i in range(input_size)]
        edmdmock.transform = lambda x: x

        kmpcperfect = AffineMPCtype(
            predictor=edmdmock,
            horizon=n_timesteps - 1,
            input_bounds=np.array([[5, -5]] * input_size),
            cost_state=np.array([1] * state_size),
            cost_input=0,
        )
        pred = kmpcperfect.generate_control_signal(x0, df)
        dfpred = sys.evolve_system(x0, np.pad(pred, ((0, 1), (0, 0))), t)
        nptest.assert_allclose(dfpred.values, df.values, rtol=0.1, atol=0.1)

    def test_kgmpc_mock_edmd_1d(self):
        self._execute_mock_test(2, 1, 10, AffineKgMPC)

    def test_kgmpc_mock_edmd_2d(self):
        self._execute_mock_test(2, 2, 10, AffineKgMPC)

    def test_kmpc_generate_control_signal(self):
        horizon = 100

        edmdcontrol = EDMDControl(
            dmd_model=gDMDAffine(),
            dict_steps=[
                ("id", TSCIdentity()),
            ],
            include_id_state=False,
        ).fit(
            self.X,
            split_by="name",
            state=self._state_columns,
            control=self._control_columns,
        )

        kmpc = AffineKgMPC(
            predictor=edmdcontrol,
            horizon=horizon,
            input_bounds=np.array([[5, -5]]),
            cost_state=np.array([1, 1, 0, 0]),
            cost_input=1,
        )

        reference = TSCDataFrame(self.dfx[self._state_columns].iloc[: horizon + 1])
        initial_conditions = InitialCondition.from_array(
            np.array([0, 0, np.pi, 0]), columns=["x", "xdot", "theta", "thetadot"]
        )
        U = kmpc.generate_control_signal(
            initial_conditions=initial_conditions, reference=reference
        )

        assert U.any() is not None
        assert U.shape == (horizon, len(self._control_columns))
