import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd

from datafold.appfold.edmd import EDMDControl
from datafold.appfold.kmpc import KoopmanMPC
from datafold.dynfold.dmd import ControlledLinearDynamicalSystem
from datafold.dynfold.transform import TSCIdentity
from datafold.pcfold import InitialCondition, TSCDataFrame
from datafold.utils._systems import InvertedPendulum


class KMPCTest(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(KMPCTest, self).__init__(*args, **kwargs)
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

    def test_kmpc_mock_edmd(self):
        gen = np.random.default_rng(42)
        state_size = 2
        input_size = 1
        n_timesteps = 50
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
        edmdmock.state_columns = ["0", "1"]
        edmdmock.control_columns = ["u"]
        edmdmock.transform = lambda x: x

        kmpcperfect = KoopmanMPC(
            predictor=edmdmock,
            horizon=n_timesteps - 1,
            state_bounds=np.array([[5, -5], [5, -5]]),
            input_bounds=np.array([[5, -5]]),
            cost_running=np.array([1, 1]),
            cost_terminal=1,
            cost_input=5,
        )
        pred = kmpcperfect.generate_control_signal(x0, df)
        nptest.assert_allclose(pred, u.ravel()[:-1])

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

        kmpc = KoopmanMPC(
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
        U = kmpc.generate_control_signal(ic=initial_conditions, reference=reference)

        assert U.any() is not None
        assert len(U) == horizon


if __name__ == "__main__":
    test = KMPCTest()
