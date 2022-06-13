import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas
from matplotlib.animation import FuncAnimation

from datafold.utils._systems import InvertedPendulum

from .model import Model


class CartPole(Model):
    cols = ["x", "xdot", "theta", "thetadot"]

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
            "t": t,
            "x": traj[0, :],
            "xdot": traj[1, :],
            "theta": traj[2, :],
            "thetadot": traj[3, :],
            "u": control,
        }

        return pandas.DataFrame(state)


class CartpoleArtist:
    def __init__(self, xmin=-5, xmax=5):
        self.xmin = xmin
        self.xmax = xmax

        self.body_size = np.array([0.5, 0.25])
        self.arm_length = 0.365
        self.mass_radius = 0.1

        self.artists = {}

        self._init()

    def _init(self):
        x1 = self.xmax
        x0 = self.xmin

        a1 = 1 / (x1 - x0)
        b1 = -a1 * x0
        b2 = 0.5

        self.scale = a1
        self.offset = np.array([b1, b2])

    def _project(self, xy):
        if xy.ndim > 1:
            offset = self.offset.reshape(1, -1)
        else:
            offset = self.offset

        xy = np.array(xy)
        return self.scale * xy + offset

    def _unproject(self, xy):
        return (xy - self.offset) / self.scale

    def draw_centerline(self):
        name = "centerline"
        if name not in self.artists:
            x1 = np.array([self.xmin, 0])
            x2 = np.array([self.xmax, 0])
            X = self._project(np.array([x1, x2]))
            line = mlines.Line2D(X[:, 0], X[:, 1], color="black", zorder=-1)
            self.artists[name] = line

    def draw_body(self, state):
        xy = self._project(np.array([state["x"], 0]))
        size = self.body_size * self.scale

        xy_ = xy - size / 2
        width, height = size

        if "body" not in self.artists:
            patch = mpatches.Rectangle(xy_, width, height)
            self.artists["body"] = patch
        else:
            patch = self.artists["body"]
            patch.set_xy(xy_)
            patch.set_width(width)
            patch.set_height(height)

    def draw_arm(self, state):
        th = state["theta"]
        xy1 = np.array([state["x"], 0])
        xy2 = xy1 + np.array([-np.sin(th), np.cos(th)]) * self.arm_length
        X = np.array([xy1, xy2])
        X_ = self._project(X)

        x = X_[:, 0]
        y = X_[:, 1]
        r = self.mass_radius * self.scale

        if "arm" not in self.artists:
            line = mlines.Line2D(x, y, color="black")
            self.artists["arm"] = line
        else:
            line = self.artists["arm"]
            line.set_xdata(x)
            line.set_ydata(y)

        if "mass" not in self.artists:
            patch = mpatches.Ellipse(X_[1], r, r, color="black")
            self.artists["mass"] = patch
        else:
            patch = self.artists["mass"]
            patch.set_center(X_[1])
            patch.set_width(r)
            patch.set_height(r)

    def _draw(self, state):
        self.draw_centerline()
        self.draw_body(state)
        self.draw_arm(state)

    def draw(self, ax, state):
        print(self.__dict__)
        self._draw(state)
        for artist in self.artists.values():
            ax.add_artist(artist)

    def animate(self, ax, dfx):
        def update(i):
            self._draw(dfx.iloc[i].to_dict())
            return self.artists.values()

        anim = FuncAnimation(ax, update, frames=dfx.shape[0], interval=20, blit=True)
        return anim


class SineControl:
    def __init__(self, amplitude, frequency, phase):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def __call__(self, t, _):
        return self.amplitude * np.sin(2 * np.pi * (self.frequency * t + self.phase))

    @classmethod
    def new_rand(cls):
        amplitude = np.random.uniform(0.1, 5.0)
        frequency = np.random.uniform(1 / 4, 5 / 2)
        phase = np.random.uniform(0, 1)

        return cls(amplitude, frequency, phase)
