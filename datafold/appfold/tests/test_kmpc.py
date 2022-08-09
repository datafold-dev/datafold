import unittest
from unittest.mock import Mock

import numpy as np
import numpy.testing as nptest

from datafold.appfold.edmd import EDMD
from datafold.appfold.kmpc import AffineKgMPC, LinearKMPC
from datafold.appfold.tests.test_edmd import EDMDTest
from datafold.dynfold.dmd import DMDControl, gDMDAffine
from datafold.dynfold.dynsystem import LinearDynamicalSystem
from datafold.dynfold.transform import TSCIdentity
from datafold.pcfold import InitialCondition


class LinearKMPCTest(unittest.TestCase):
    def setUp(self) -> None:
        self.X_tsc, self.U_tsc = EDMDTest.setup_inverted_pendulum(
            sim_time_step=0.01, sim_num_steps=1000, training_size=20
        )

    def _execute_mock_test(self, state_size, input_size, n_timesteps, seed=42):
        gen = np.random.default_rng(seed)

        A = gen.uniform(size=(state_size, state_size)) * 2 - 1.0
        x0 = gen.uniform(size=state_size)
        B = gen.uniform(size=(state_size, input_size)) * 2 - 1.0
        t = np.linspace(0, n_timesteps, n_timesteps)
        u = gen.uniform(size=(1, input_size))
        u = (u.T + np.sin(u.T + u.T * np.atleast_2d(t))).T
        df = (
            LinearDynamicalSystem(
                sys_type="flowmap", sys_mode="matrix", is_controlled=True
            )
            .setup_matrix_system(A, control_matrix=B)
            .evolve_system(
                x0, control_input=u, time_values=np.arange(u.shape[0] + 1), time_delta=1
            )
        )

        edmdmock = Mock()
        edmdmock.dmd_model.sys_matrix_ = A
        edmdmock.dmd_model.control_matrix_ = B
        edmdmock.feature_names_in_ = [f"x{i}" for i in range(state_size)]
        edmdmock.transform = lambda x: x

        kmpcperfect = LinearKMPC(
            predictor=edmdmock,
            horizon=n_timesteps,
            state_bounds=np.array([[5, -5]] * state_size),
            input_bounds=np.array([[5, -5]] * input_size),
            cost_running=np.array([1] * state_size),
            cost_terminal=1,
            cost_input=0,
        )
        pred = kmpcperfect.generate_control_signal(x0, df)
        nptest.assert_allclose(pred, u)

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
            self.X_tsc,
            U=self.U_tsc,
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

        reference = self.X_tsc[["x", "theta"]].iloc[: horizon + 1]
        initial_conditions = InitialCondition.from_array(
            np.array([0, 0, np.pi, 0]),
            time_value=0,
            feature_names=["x", "xdot", "theta", "thetadot"],
        )
        U = kmpc.generate_control_signal(
            initial_conditions=initial_conditions, reference=reference
        )

        self.assertEqual(U.shape, (horizon, self.U_tsc.shape[1]))


class AffineKMPCTest(unittest.TestCase):
    def _execute_mock_test(self, state_size, input_size, n_timesteps, seed=42):
        gen = np.random.default_rng(seed)
        x0 = gen.uniform(size=state_size)
        A = gen.uniform(-0.4, 0.5, size=(state_size, state_size))
        np.fill_diagonal(A, gen.uniform(-0.6, -0.5, size=state_size))
        Bi = np.stack(
            [gen.uniform(size=(state_size, state_size)) for _ in range(input_size)],
            2,
        )

        t = np.arange(0, 1.01, 0.1)
        u = 0.5 + gen.uniform(-0.1, 0.1, size=(len(t), input_size))
        u = u + np.sin(u + u * np.atleast_2d(t).T)

        sys = LinearDynamicalSystem(
            sys_type="differential",
            sys_mode="matrix",
            is_controlled=True,
            is_control_affine=True,
        ).setup_matrix_system(A, control_matrix=Bi)

        sys._requires_last_control_state = True

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
            horizon=n_timesteps,
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

        X_tsc, U_tsc = EDMDTest.setup_inverted_pendulum(
            sim_time_step=0.01,
            sim_num_steps=1000,
            training_size=20,
            include_last_control_state=True,
        )

        edmdcontrol = EDMD(
            dict_steps=[
                ("id", TSCIdentity()),
            ],
            dmd_model=gDMDAffine(),
            include_id_state=False,
        ).fit(
            X_tsc,
            U=U_tsc,
        )

        kmpc = AffineKgMPC(
            predictor=edmdcontrol,
            horizon=horizon,
            input_bounds=np.array([[5, -5]]),
            cost_state=np.array([1, 1, 0, 0]),
            cost_input=1,
        )

        reference = X_tsc.iloc[: horizon + 1]
        initial_conditions = InitialCondition.from_array(
            np.array([0, 0, np.pi, 0]),
            time_value=0,
            feature_names=["x", "xdot", "theta", "thetadot"],
        )
        actual = kmpc.generate_control_signal(
            initial_conditions=initial_conditions, reference=reference
        )

        self.assertTrue(actual is not None)
        self.assertEqual(actual.shape, (horizon, U_tsc.shape[1]))
