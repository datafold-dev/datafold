import numpy as np


class Metrics:
    @classmethod
    def compute_rmse(cls, y1, y2):
        dy = y2 - y1
        N = dy.shape[0]
        return np.sqrt(np.sum(dy*dy, axis=0))/N

    @classmethod
    def error_cumulative(cls, y1, y2):
        dy = y2 - y1
        N = dy.shape[0]

        err = dy*dy
        norm = np.arange(N).reshape(-1, 1) + 1
        c_err = np.sqrt(np.cumsum(err, axis=0)) / norm
        return c_err

    @classmethod
    def error_time(cls, y1, y2):
        dy = y2 - y1
        return np.abs(dy)
