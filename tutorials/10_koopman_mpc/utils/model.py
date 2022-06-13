from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas
from pandas import DataFrame
from tqdm import tqdm

from datafold.appfold import EDMDControl
from datafold.pcfold import TSCDataFrame


class Augmenter:
    def augment(self, state):
        return state

    def deaugment(self, state):
        return state


class ScalingAugmenter(Augmenter):
    def __init__(self, offset, scale):
        self.offset = offset
        self.scale = scale

    @classmethod
    def new(cls, dfx: DataFrame):
        offset, scale = cls._init_params(dfx)
        return cls(offset, scale)

    @property
    def keys(self):
        return list(self.offset.keys())

    def augment(self, state):
        # return (state - self.offset) / self.scale

        if isinstance(state, (DataFrame, pandas.Series)):
            return (state - self.offset) / self.scale
        elif isinstance(state, dict):
            ret = {}
            for k in state:
                if k in self.keys:
                    ret[k] = (state[k] - self.offset[k]) / self.scale[k]
                else:
                    ret[k] = state[k]

            return ret
        raise TypeError

    def deaugment(self, state):
        if isinstance(state, DataFrame):
            return (state * self.scale) + self.offset
        elif isinstance(state, dict):
            ret = {}
            for k in state:
                if k in self.keys:
                    ret[k] = (state[k] * self.scale[k]) + self.offset[k]
                else:
                    ret[k] = state[k]

            return ret
        raise TypeError

    @classmethod
    def _init_params(cls, dfx: DataFrame):
        offset = dfx.mean()
        scale = dfx.std()

        # scale = (dfx.max() - dfx.min()) / 2

        offset["t"] = 0
        # offset['u'] = 0
        scale["t"] = 1
        # scale['u'] = 1

        return offset, scale


class Model:
    def reset(self, ic=None):
        raise NotImplementedError

    def predict(self, t_step, n_steps, control: np.ndarray) -> DataFrame:
        raise NotImplementedError

    @staticmethod
    def _control_func(time_values, control):
        def f(t, _):
            return np.interp(t, time_values, control)

        return f


class Predictor:
    def __init__(self, state_cols, input_cols):
        self.state_cols = state_cols
        self.input_cols = input_cols

        self._predictor = self._init_predictor()

    def _init_predictor(self) -> EDMDControl:
        raise NotImplementedError

    def fit(self, X_tsc: TSCDataFrame):
        raise NotImplementedError

    def predict(self, initial_conds, control_input, t):
        raise NotImplementedError


class PredictResult:
    def __init__(self, control, ic, pred, state_cols, input_cols, traj=None):
        self.control = control
        self.ic = ic

        self.pred = pred
        self.traj = traj
        self.state_cols = state_cols
        self.input_cols = input_cols

    def error(self, actual):
        cols = self.state_cols
        pred = self.pred.copy()
        pred.index = actual.index

        diff = pred[cols] - actual[cols]
        diff2 = diff**2

        m = diff2.mean()
        s = diff2.std()

        return m, s

    def error_v2(self, dfx):
        cols = self.state_cols

        diff = self.pred[cols].values - dfx[cols].values
        err = diff**2

        err = np.hstack([err, np.mean(err, axis=1).reshape(-1, 1)])
        return err

    def metric(self, dfx, thresholds=(1e-3, 1e-6)):
        err = self.error_v2(dfx)
        err = np.concatenate((err, np.ones((1, err.shape[1])) * np.inf), axis=0)

        err = err.reshape(*err.shape, 1)
        thresholds = np.array(thresholds).reshape(1, 1, -1)

        mask = err > thresholds
        idx = np.argmax(mask, axis=0)
        return idx

    def plot(self, axes=None, cols=None, ranges=None):
        if cols is None:
            cols = self.state_cols

        if axes is None:
            m = len(cols)
            fig = plt.figure(figsize=(6 * m, 3))
            axes = fig.subplots(1, m).flatten()
        else:
            fig = None

        for i, col in enumerate(cols):
            ax = axes[i]

            if self.traj is not None and col in self.traj.dfx:
                ax.plot(self.traj.dfx[col].values, label="traj")
            else:
                ax.plot([])

            if col in self.pred:
                ax.plot(self.pred[col].values, label="pred")

            if ranges is not None and col in ranges:
                axes[i].axhline(ranges[col][0], color="k", linestyle="--", alpha=0.6)
                axes[i].axhline(ranges[col][1], color="k", linestyle="--", alpha=0.6)

            ax.grid()
            ax.set_title(col)

        return fig, axes

    def __len__(self):
        return self.pred.shape[0]


class Predictions:
    def __init__(self, predictions: List[PredictResult], state_cols):
        self.predictions = predictions
        self.state_cols = state_cols

        self.n_steps = [len(p) for p in predictions]

    @classmethod
    def new(cls, f_predict, trajectories, cols):
        predictions = []
        for traj in tqdm(trajectories, leave=False):
            pred = f_predict(traj)  # .ic, traj.dfx['t'])
            predictions.append(pred)

        return cls(predictions, cols)

    def error(self, trajectories):
        trajectories = list(trajectories)
        max_steps = max(self.n_steps)
        cols = self.state_cols

        counts = np.zeros(max_steps, dtype=int)
        err = np.zeros((max_steps, len(cols)))

        for pred, traj in zip(self.predictions, trajectories):
            dfx = traj.dfx
            diff = pred.pred[cols].values - dfx[cols].values

            l = len(pred)
            counts[:l] += 1
            err[:l, :] += diff**2

        mean_err = err / counts.reshape(-1, 1)
        return mean_err

    def metric(self, trajectories, thresholds=(1e-3, 1e-6)):
        trajectories = list(trajectories)

        N = len(trajectories)
        K = len(self.state_cols) + 1

        metrics = np.zeros((N, K, len(thresholds)), dtype=int)
        for i, (pred, traj) in enumerate(zip(self.predictions, trajectories)):
            metric = pred.metric(traj.dfx, thresholds=thresholds)
            metrics[i, :, :] = metric

        return metrics

    def plot(self, axes=None, cols=None, n=10, ranges=None):
        if cols is None:
            cols = self.state_cols

        if axes is None:
            m = len(cols)
            fig = plt.figure(figsize=(6 * m, 2 * n))
            axes = fig.subplots(n, m)
        else:
            fig = None

        for i in range(n):
            self.predictions[i].plot(axes=axes[i, :], cols=cols, ranges=ranges)

        return fig, axes

    def __iter__(self):
        for pred in self.predictions:
            yield pred
