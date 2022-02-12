import unittest

import numpy as np
import pandas as pd

from datafold.appfold.edmd import EDMDControl
from datafold.appfold.kmpc import KoopmanMPC
from datafold.dynfold.transform import TSCIdentity
from datafold.pcfold import InitialCondition, TSCDataFrame
from datafold.utils.kmpc import InvertedPendulum


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

            trajectory = model.trajectory(dt, num_steps, control_function)
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

    def test_edmd_dictionary(self):
        # Using Identity as feature dictionary

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

        initial_conditions = InitialCondition.from_array(
            np.array([0, 0, np.pi, 0]), columns=self._state_columns
        )
        edmd_predictions = edmdcontrol.predict(
            X=initial_conditions, time_values=self.t, control_input=self.u
        )

        assert edmd_predictions["x"].values.all() != None
        assert len(edmd_predictions["x"].values) == len(self.dfx["x"].values)

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
    test.test_edmd_dictionary()
    test.test_kmpc_generate_control_signal()
