import numpy as np
import pandas
from pandas import DataFrame

import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.collections
import matplotlib.lines as mlines

from datafold.utils._systems import InvertedPendulum

from .model import Model, ScalingAugmenter


class CartPole(Model):
    cols = ['x', 'xdot', 'theta', 'thetadot']

    def __init__(self, ic=None):
        self._ic = ic
        self._model = self._init_model(ic=ic)

    @classmethod
    def _init_model(cls, ic=None):
        model = InvertedPendulum(initial_condition=cls._cast_ic(ic))
        return model

    @classmethod
    def _cast_ic(cls, ic):
        if ic is not None:
            ic = ic[cls.cols].values
        return ic

    def _predict(self, t_step, n_steps, control_func):
        model = self._model
        traj = model.predict(
            time_step=t_step,
            num_steps=n_steps,
            control_func=control_func,
        )

        assert model.sol.success

        return traj

    def reset(self, ic=None):
        self._ic = ic

        ic = self._cast_ic(ic)
        self._model.initial_condition = ic
        self._model.reset(ic)

    def predict(self, t_step, n_steps, control):
        time_values = np.arange(n_steps) * t_step

        if isinstance(control, np.ndarray):
            control_func = self._control_func(time_values, control)
        else:
            control_func = control

        traj = self._predict(t_step, n_steps, control_func)
        t = self._model.sol.t
        control = control_func(t, traj)

        state = {
            't': t,
            'x': traj[0,:],
            'xdot': traj[1,:],
            'theta': traj[2,:],
            'thetadot': traj[3,:],
            'u': control,
        }

        return pandas.DataFrame(state)

    def draw_state(self, ax: matplotlib.axes.Axes, state, xmin=-5, xmax=5):
        artist = CartpoleArtist()
        artist(ax)
        # patches = []
        # 
        # # draw the centerline
        # line = mlines.Line2D([0,1], [0.5, 0.5], color='black')
        # ax.add_artist(line)

        # x = 0

        # # draw the body
        # patch = mpatches.Rectangle((0.5,0.5), .10, .05, 0)
        # patches.append(patch)

        # # draw the pole
        # line = mlines.Line2D([0.5,])

        # collection = matplotlib.collections.PatchCollection(patches)
        # ax.add_collection(collection)


class CartpoleArtist:
    def __init__(self):
        self.offset = np.array([0.5,0.5])
        self.scale = np.array([1,1])

        self.body_size = np.array([0.1,0.05])

    def draw_centerline(self, ax):
        line = mlines.Line2D([0,1], [0.5, 0.5], color='black')
        ax.add_artist(line, zorder=-1)

    def draw_body(self, ax, xy, size):
        xy = (xy * self.scale) + self.offset
        size = size * self.scale

        xy_ = xy - size/2
        width, height = size

        patch = mpatches.Rectangle(xy_, width, height)
        ax.add_artist(patch)

    def draw(self, ax, xy):
        self.draw_centerline(ax)
        self.draw_body(ax, xy, self.body_size)





class CartpoleAugmenter(ScalingAugmenter):
    @classmethod
    def new(cls, dfx: DataFrame):
        aug = cls(0, 1)
        dfx_a = aug.augment(dfx)

        offset, scale = cls._init_params(dfx_a)
        return cls(offset, scale)

    @classmethod
    def _init_params(cls, dfx):
        offset, scale = super()._init_params(dfx)
        offset['sin_th1'] = 0
        offset['cos_th1'] = 0
        scale['sin_th1'] = 1
        scale['cos_th1'] = 1

        return offset, scale

    def augment(self, state):
        state = state.copy()
        if 'theta' in state:
            state['sin_th1'] = np.sin(state['theta'])
            state['cos_th1'] = np.cos(state['theta'])
        return super().augment(state)

    def deaugment(self, state):
        state = super().deaugment(state)

        if 'sin_th1' in state and 'cos_th1' in state:
            sin = state['sin_th1']
            cos = state['cos_th1']

            th1 = np.arcsin(sin).values
            th2 = np.arccos(cos).values

            th = np.mean([th1, th2], axis=0)
            state['theta'] = th1

        return state


class SineControl:
    def __init__(self, amplitude, frequency, phase):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def __call__(self, t, _):
        return self.amplitude * \
                np.sin(2*np.pi * (self.frequency*t + self.phase))

    @classmethod
    def new_rand(cls):
        amplitude = np.random.uniform(0.1, 5.0)
        frequency = np.random.uniform(1/4, 5/2)
        phase = np.random.uniform(0, 1)

        return cls(amplitude, frequency, phase)
