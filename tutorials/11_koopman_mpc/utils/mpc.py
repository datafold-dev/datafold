from pprint import pformat
from typing import Optional

import numpy as np
import pandas
from tqdm import tqdm

from datafold.appfold.mpc import LinearKMPC
from datafold.pcfold import TSCDataFrame

from .model import Augmenter, Model, Predictor


class MPCConfig:
    def __init__(self, state_columns, qois_columns, time_step, **kwargs):
        self.state_columns = state_columns
        self.qois_columns = qois_columns

        self.state_bounds = kwargs.get("state_bounds", np.array([[2.5, -2.5], [1, -1]]))
        self.input_bounds = kwargs.get("input_bounds", np.array([[5, -5]]))

        self.cost_running = kwargs.get("cost_running", np.array([100, 0]))
        self.cost_terminal = kwargs.get("cost_terminal", 1)
        self.cost_input = kwargs.get("cost_input", 1)

        self.time_step = time_step
        self.horizon = kwargs.get("horizon", 100)

    def copy(self):
        return MPCConfig(**self.__dict__)

    def __repr__(self):
        name = self.__class__.__name__
        data = self.__dict__
        data = pformat(data)
        return f"<{name}>: {data}"


class MPC:
    def __init__(
        self,
        predictor: Predictor,
        config: MPCConfig,
        augmenter: Optional[Augmenter] = None,
    ):
        self.predictor = predictor

        self.config = config
        self._augmenter = augmenter

        self._mpc = self._init_mpc()

    def _init_mpc(self):
        return LinearKMPC(
            predictor=self.predictor._predictor,
            horizon=self.config.horizon,
            state_bounds=self.config.state_bounds,
            input_bounds=self.config.input_bounds,
            qois=self.config.qois_columns,
            # qois=['x', 'xdot', 'sin_th1'],
            cost_running=self.config.cost_running,
            cost_terminal=self.config.cost_terminal,
            cost_input=self.config.cost_input,
        )

    def _augment(self, state):
        if self._augmenter is not None:
            return self._augmenter.augment(state)
        return state

    def _deaugment(self, state):
        if self._augmenter is not None:
            return self._augmenter.deaugment(state)
        return state

    def predict(self, reference, initial_conds):
        horizon = self.config.horizon
        state_columns = self.config.state_columns
        qois_columns = self.config.qois_columns

        ic_a = self._augment(initial_conds)[state_columns]
        ref_a = self._augment(reference)
        ref_a = ref_a[qois_columns].iloc[: horizon + 1]

        t = reference["t"].iloc[: horizon + 1].values

        mpc_control_a = self._mpc.optimal_control_sequence(ic_a, ref_a).reshape(-1)
        mpc_pred = self.predictor.predict(ic_a, mpc_control_a, None)
        mpc_cost = self._mpc.compute_cost(mpc_control_a, ref_a, ic_a)

        mpc_control = self._deaugment({"u": mpc_control_a})["u"]

        return MPCResult(
            initial_conds, t, reference, mpc_control, mpc_pred, mpc_cost, self.config
        )

    def control_loop(self, ic, target_state, n_steps, model: Model, step_size):
        state = ic

        horizon = self.config.horizon
        dt = self.config.time_step

        cols = target_state.columns
        ref = np.ones((horizon + 1, len(cols))) * target_state[cols].values
        ref = pandas.DataFrame(ref, columns=cols)
        ref["t"] = np.arange(horizon + 1) * dt

        pred = None
        data = []
        control = np.zeros((n_steps, step_size))
        for i in tqdm(range(n_steps)):
            ret = self.predict(ref, state)
            control[i, :] = ret.control[:step_size]

            model.reset(ic)
            pred = model.predict(dt, (i + 1) * step_size, control[: i + 1, :].flatten())

            state = pred.iloc[[-1]].copy()
            state = state.set_index("t")
            state.index = [0]
            state = TSCDataFrame.from_single_timeseries(state, ts_id=0)

            data.append((ret, pred))

        return pred, control, data


class MPCResult:
    def __init__(self, ic, t, ref, control, pred, cost, config: MPCConfig):
        self.config = config
        self.ic = ic
        self.ref = ref
        self.t = t

        self.control = control
        self.pred = pred
        self.cost = cost

    def actual_trajectory(self, model: Model):
        horizon = self.config.horizon

        model.reset(self.ic)
        mpc_trajectory = model.predict(
            t_step=self.config.time_step, n_steps=horizon, control=self.control
        )

        return mpc_trajectory
