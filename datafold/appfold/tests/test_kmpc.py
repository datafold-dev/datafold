import unittest
from unittest.mock import Mock

import numpy as np
import numpy.testing as nptest
import pandas as pd

from datafold.appfold.edmd import EDMD
from datafold.appfold.kmpc import AffineKgMPC, LinearKMPC
from datafold.dynfold.dmd import DMDControl, gDMDAffine
from datafold.dynfold.dynsystem import LinearDynamicalSystem
from datafold.dynfold.transform import TSCIdentity
from datafold.pcfold import InitialCondition, TSCDataFrame
from datafold.utils._systems import InvertedPendulum


class LinearKMPCTest(unittest.TestCase):
    def setUp(self) -> None:
        self._state_columns = ["x", "xdot", "theta", "thetadot"]
        self._control_columns = ["u"]
        self.X, self.t, self.u, self.dfx, self.dfu = self._generate_data()

    def _generate_data(self, seed=42):

        # Simulation model parameters

        dt = 0.01
        num_steps = 1000
        size = 20
        model = InvertedPendulum()
        gen = np.random.default_rng(seed)

        # Data structures

        Xlist = []
        Ulist = []

        for _ in range(size):
            model.reset()
            control_amplitude = 0.1 + 0.9 * gen.random()
            control_frequency = np.pi + 2 * np.pi * gen.random()
            control_phase = 2 * np.pi * gen.random()
            control_function = lambda t, y: control_amplitude * np.sin(
                control_frequency * t + control_phase
            )

            trajectory = model.predict(
                time_step=dt,
                num_steps=num_steps,
                control_func=control_function,
            )
            assert model.sol.success, (
                f"Divergent solution for amplitude={control_amplitude}, "
                f"frequency={control_frequency}"
            )
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
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="matrix", is_controlled=True
            )
            .setup_matrix_system(A, control_matrix=B)
            .evolve_system(
                x0, control_input=u, time_values=np.arange(u.shape[0]), time_delta=1
            )
        )

        edmdmock = Mock()
        edmdmock.dmd_model.sys_matrix_ = A
        edmdmock.dmd_model.control_matrix_ = B
        edmdmock.feature_names_in_ = [f"x{i}" for i in range(state_size)]
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

        edmdcontrol = EDMD(
            dict_steps=[
                ("id", TSCIdentity()),
            ],
            include_id_state=False,
            dmd_model=DMDControl(),
        ).fit(
            self.X[self._state_columns],
            self.X[self._control_columns],
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
            np.array([0, 0, np.pi, 0]), time_value=0, feature_names=["x", "xdot", "theta", "thetadot"]
        )
        U = kmpc.generate_control_signal(
            initial_conditions=initial_conditions, reference=reference
        )

        assert U.any() is not None
        assert U.shape == (horizon, len(self._control_columns))


class AffineKMPCTest(unittest.TestCase):
    def setUp(self) -> None:
        self._state_columns = ["x", "xdot", "theta", "thetadot"]
        self._control_columns = ["u"]
        self.X, self.t, self.u, self.dfx, self.dfu = self._generate_data()

    def _generate_data(self, seed=42):

        # Simulation model parameters

        dt = 0.01
        num_steps = 1000
        size = 20
        model = InvertedPendulum()
        gen = np.random.default_rng(seed)

        # Data structures
        Xlist = []
        Ulist = []

        for _ in range(size):
            model.reset()
            control_amplitude = 0.1 + 0.9 * gen.random()
            control_frequency = np.pi + 2 * np.pi * gen.random()
            control_phase = 2 * np.pi * gen.random()
            control_function = lambda t, y: control_amplitude * np.sin(
                control_frequency * t + control_phase
            )

            trajectory = model.predict(
                time_step=dt,
                num_steps=num_steps,
                control_func=control_function,
            )

            try:
                self.assertTrue(model.sol.success)
            except AssertionError as e:
                print(
                    f"Divergent solution for amplitude={control_amplitude}, "
                    f"frequency={control_frequency}"
                )
                raise e

            t = model.sol.t
            dfx = pd.DataFrame(data=trajectory.T, index=t, columns=self._state_columns)
            dfx["u"] = 0.0
            Xlist.append(dfx)

            control_input = control_function(t, trajectory)
            dfu = pd.DataFrame(data=control_input, index=t, columns=("u",))
            Ulist.append(dfu)

            X_tsc = TSCDataFrame.from_frame_list(Xlist)
            X_tsc[self._control_columns] = TSCDataFrame.from_frame_list(Ulist)

            return X_tsc, t, control_input, dfx, dfu

    def _execute_mock_test(self, state_size, input_size, n_timesteps, seed=42):
        gen = np.random.default_rng(seed)
        x0 = gen.uniform(size=state_size)
        A = gen.uniform(-0.4, 0.5, size=(state_size, state_size))
        np.fill_diagonal(A, gen.uniform(-0.6, -0.5, size=state_size))
        Bi = np.stack(
            [gen.uniform(size=(state_size, state_size)) for _ in range(input_size)],
            2,
        )

        t = np.linspace(0, n_timesteps - 1, n_timesteps) * 0.1
        u = 0.5 + gen.uniform(-0.1, 0.1, size=(1, input_size))
        u = (u.T + np.sin(u.T + u.T * np.atleast_2d(t))).T
        sys = LinearDynamicalSystem(
            sys_type="differential",
            sys_mode="matrix",
            is_controlled=True,
            is_control_affine=True,
        ).setup_matrix_system(A, control_matrix=Bi)
        expected = sys.evolve_system(
            x0, control_input=u, time_values=t, time_delta=t[1] - t[0]
        )

        edmdmock = Mock()
        edmdmock.dmd_model.sys_matrix_ = A
        edmdmock.dmd_model.control_matrix_ = Bi
        edmdmock.feature_names_in_ = [f"x{i}" for i in range(state_size)]
        edmdmock.transform = lambda x: x

        kmpcperfect = AffineKgMPC(
            predictor=edmdmock,
            horizon=n_timesteps - 1,
            input_bounds=np.array([[5, -5]] * input_size),
            cost_state=np.array([1] * state_size),
            cost_input=0,
        )
        pred = kmpcperfect.generate_control_signal(x0, expected)
        actual = sys.evolve_system(
            x0,
            control_input=np.pad(pred, ((0, 1), (0, 0))),
            time_values=t,
            time_delta=t[1] - t[0],
        )
        nptest.assert_allclose(
            actual.to_numpy(), expected.to_numpy(), rtol=0.1, atol=0.1
        )

    def test_kgmpc_mock_edmd_1d(self):
        self._execute_mock_test(state_size=2, input_size=1, n_timesteps=10)

    def test_kgmpc_mock_edmd_2d(self):
        self._execute_mock_test(state_size=2, input_size=2, n_timesteps=10)

    def test_kmpc_generate_control_signal(self):
        horizon = 100

        edmdcontrol = EDMD(
            dict_steps=[
                ("id", TSCIdentity()),
            ],
            dmd_model=gDMDAffine(),
            include_id_state=False,
        ).fit(
            self.X[self._state_columns],
            self.X[self._control_columns],
        )

        kmpc = AffineKgMPC(
            predictor=edmdcontrol,
            horizon=horizon,
            input_bounds=np.array([[5, -5]]),
            cost_state=np.array([1, 1, 0, 0]),
            cost_input=1,
        )

        reference = TSCDataFrame.from_frame_list(
            [self.dfx[self._state_columns].iloc[: horizon + 1]]
        )
        initial_conditions = InitialCondition.from_array(
            np.array([0, 0, np.pi, 0]), time_value=0, feature_names=["x", "xdot", "theta", "thetadot"]
        )
        U = kmpc.generate_control_signal(
            initial_conditions=initial_conditions, reference=reference
        )

        self.assertTrue(U is not None)
        self.assertEqual(U.shape, (horizon, len(self._control_columns)))
