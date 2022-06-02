import time
import numpy as np
from tqdm import tqdm

from .model import Predictions
from .data_splits import DataSplits


class ParamGrid:
    N: int
    keys: list
    index: np.ndarray
    lengths: np.ndarray
    stop: bool

    def __init__(self, params, keys=None):
        self.params = params

        if keys is None:
            keys = list(params)

        self.keys = keys

        self.N = None
        self.index = None
        self.lengths = None
        self.stop = False

        self._init()

    def _init(self):
        N = len(self.params)

        idx = np.zeros(N, dtype=int)
        lengths = np.zeros(N, dtype=int)
        keys = self.keys

        for i, k in enumerate(keys):
            lengths[i] = len(self.params[k])

        self.N = N
        self.index = idx
        self.lengths = lengths

    def get(self):
        ret = {}
        for i, k in enumerate(self.keys):
            ret[k] = self.params[k][self.index[i]]

        return ret

    def reset(self):
        self.index[:] = 0
        self.stop = False

    def step(self):
        self.index[0] += 1
        for i in range(self.N):
            if self.index[i] >= self.lengths[i]:
                if i+1 == self.N:
                    self.stop = True
                else:
                    self.index[i] = 0
                    self.index[i+1] += 1

    def next(self):
        if self.stop:
            raise StopIteration

        ret = self.get()
        self.step()
        return ret

    def make_grid(self, data):
        xshape = list(self.shape())
        shape = xshape + list(data.shape[1:])

        grid = np.zeros(shape, dtype=data.dtype)
        for i in range(data.shape[0]):
            idx = self.reverse_index(i, index=True)
            grid[idx] = data[i]

        return grid

    def grid_index(self, params):
        index = []
        for k in self.keys:
            index.append(self.params[k].index(params[k]))

        return index

    def get_index(self, params):
        idx = 0
        factor = 1
        for k in self.keys:
            i = self.params[k].index(params[k])
            N = len(self.params[k])

            idx += i*factor
            factor *= N

        return idx

    def reverse_index(self, idx, values=False, index=False):
        params = {}
        shape = self.shape()

        keys = list(reversed(self.keys))
        shape = [len(self.params[k]) for k in keys]
        factor = np.product(shape)

        for i, k in enumerate(keys):
            factor /= shape[i]
            params[k] = int(idx // factor)
            idx -= params[k] * factor

        if values:
            for k in keys:
                params[k] = self.params[k][params[k]]

        if index:
            return tuple([params[k] for k in self.keys])

        return params

    def shape(self):
        return tuple([len(self.params[i]) for i in self.keys])

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return np.product(self.lengths)


class Experiment:
    def __init__(self, param_grid: ParamGrid):
        self.param_grid = param_grid

        self.trials = None

    def new_trial(self, params):
        return Trial(params)

    def get_trial(self, params):
        idx = self.param_grid.get_index(params)
        return self.trials[idx]

    def run(self, data):
        trials = []
        for params in tqdm(self.param_grid):
            trial = self.new_trial(params)
            trial.run(data)
            trials.append(trial)

        self.trials = trials

        return trials

    def __iter__(self):
        return iter(self.trials)


class Trial:
    def __init__(self, params):
        self.params = params
        self.metric = None

    def model_train(self, train_data):
        raise NotImplementedError

    def model_predict(self, trajectory):
        raise NotImplementedError

    def prediction_eval(self, pred):
        raise NotImplementedError

    def model_predictions(self, data):
        predictions = []
        for trajectory in tqdm(data, leave=False):
            pred = self.model_predict(trajectory)
            predictions.append(pred)

        return predictions

    def model_eval(self, data):
        predictions = self.model_predictions(data)
        metrics = []

        for pred in predictions:
            metric = self.prediction_eval(pred)
            metrics.append(metric)

        metric = np.mean(np.array(metrics), axis=0)

        return predictions, metrics, metric

    def run(self, data):
        t0 = time.time()
        self.model_train(data)
        t1 = time.time()
        predictions, metrics, metric = self.model_eval(data)
        t2 = time.time()

        self.predictions = predictions
        self.metrics = metrics
        self.metric = metric
        self.runtimes = [t0, t1, t2]

    def __repr__(self):
        name = str(self.__class__.__name__)
        return f'<{name} {self.params} {self.metric}>'


