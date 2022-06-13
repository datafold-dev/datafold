from tqdm import tqdm

from datafold.pcfold import TSCDataFrame
from .model import Augmenter, ScalingAugmenter

from typing import List
import matplotlib.pyplot as plt


class Trajectory:
    """
    Container for an initial condition and a data frame describing a
    trajectory. Also caches augmented version of initial condition and
    trajectory.
    """
    def __init__(self, ic, dfx):
        self.ic = ic
        self.dfx = dfx

        self.ic_a = None
        self.dfx_a = None

    def init_augment(self, augmenter: Augmenter):
        self.ic_a = augmenter.augment(self.ic.copy())
        self.dfx_a = augmenter.augment(self.dfx.copy())

    def __len__(self):
        return self.dfx.shape[0]

    def is_augmented(self):
        return self.ic_a is not None

    def __repr__(self):
        aug = 'A' if self.is_augmented() else ''
        name = self.__class__.__name__

        cols = [c for c in self.ic.columns if c not in ['t', 'u']]
        ic = self.ic[cols].iloc[0].to_dict()
        ic = {k: f'{v:.2e}' for k, v in ic.items()}
        return f'<{name} {ic} {aug}>'

    def plot(self, cols, axes=None):
        if axes is None:
            m = len(cols)
            fig = plt.figure(figsize=(6*m, 3))
            axes = fig.subplots(1, m).flatten()
        else:
            fig = None

        for i, col in enumerate(cols):
            ax = axes[i]
            ax.plot(self.dfx[col].values)

            ax.grid()
            ax.set_title(col)

        return fig, axes


class DataSplits:
    """
    Container for train/test data splits, and to manage data augmentation
    """
    def __init__(self, train: List[Trajectory], test: List[Trajectory],
                 state_cols, input_cols):
        self.train = self._init_trajectories(train)
        self.test = self._init_trajectories(test)
        self.state_cols = state_cols
        self.input_cols = input_cols

        self._augmented = True
        self._augmenter = None

        self._init_tsc()

    @staticmethod
    def _init_trajectories(trajectories):
        ret = []
        for traj in trajectories:
            if isinstance(traj, Trajectory):
                pass
            elif isinstance(traj, dict):
                traj = Trajectory(ic=traj['ic'], dfx=traj['dfx'])
            elif isinstance(traj, tuple):
                ic, dfx = traj
                traj = Trajectory(ic=ic, dfx=dfx)
            elif isinstance(traj, list):
                ic, dfx = traj
                traj = Trajectory(ic=ic, dfx=dfx)

            ret.append(traj)

        return ret

    def _init_tsc(self):
        self.train_tsc = self._make_tsc(self.train)
        self.test_tsc = self._make_tsc(self.test)

    def init_augment(self):
        self._augmenter = self._make_augmenter()

        for trajectories in [self.train, self.test]:
            if trajectories is not None:
                for traj in tqdm(trajectories):
                    traj.init_augment(self._augmenter)

        self._augmented = True

    def _make_augmenter(self):
        return ScalingAugmenter.new(self.train_tsc)

    def _make_tsc(self, trajectories: List[Trajectory]) -> TSCDataFrame:
        if trajectories:
            return TSCDataFrame.from_frame_list([t.dfx for t in trajectories])
        raise ValueError

    @property
    def train_augment(self):
        if not self._augmented:
            self.init_augment()

        for traj in self.train:
            yield Trajectory(traj.ic_a, traj.dfx_a)

    @property
    def test_augment(self):
        if not self._augmented:
            self.init_augment()

        for traj in self.test:
            yield Trajectory(traj.ic_a, traj.dfx_a)

    @property
    def columns(self):
        return self.state_cols + self.input_cols

    @property
    def n_train(self):
        return len(self.train)

    @property
    def n_test(self):
        return len(self.test)

    def plot_tsc(self, axes=None, cols=None):
        if cols is None:
            cols = self.state_cols + self.input_cols

        if axes is None:
            n = len(cols)
            fig = plt.figure(figsize=(6, 3*n))
            axes = fig.subplots(n, 1).flatten()

        for i, col in enumerate(cols):
            ax = axes[i]
            ax.plot(self.train_tsc[col].values, label='train')
            ax.plot(self.test_tsc[col].values, label='test')
            ax.set_title(col)
            ax.grid()

        axes[-1].legend()

        return axes

    def plot(self, axes=None, cols=None, n=10, split='train', aug=False):
        data = self.get_split(split, aug)
        n = min(len(data), n)

        if cols is None:
            cols = list(data[0].dfx.columns)

        if axes is None:
            m = len(cols)
            fig = plt.figure(figsize=(6*m, 3*n))
            axes = fig.subplots(n, m)
        else:
            fig = None

        for i in range(n):
            data[i].plot(cols=cols, axes=axes[i,:])

        return fig, axes

    def get_split(self, split='train', aug=False) -> List[Trajectory]:
        if split == 'train':
            if aug:
                return list(self.train_augment)
            else:
                return self.train
        else:
            if aug:
                return list(self.test_augment)
            else:
                return self.test

    def get_tsc(self, split='train', aug=False) -> TSCDataFrame:
        if split == 'train':
            if aug:
                return self._augmenter.augment(self.train_tsc)
            else:
                return self.train_tsc
        else:
            if aug:
                return self._augmenter.augment(self.test_tsc)
            else:
                return self.test_tsc
