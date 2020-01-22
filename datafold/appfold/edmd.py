#!/usr/bin/env python3


from sklearn.pipeline import Pipeline
from datafold.pcfold import TSCDataFrame
from datafold.pcfold.timeseries.transform import TSCTransformMixIn


class EDMDDict:
    def __init__(self, steps):
        self._pipeline = Pipeline(steps=steps, memory=None, verbose=True)

    def fit(self, X_ts: TSCDataFrame):
        self.fit(X_ts=X_ts)
        return self

    def transform(self, X_ts: TSCDataFrame):
        return self._pipeline.transform(X_ts)

    def fit_transform(self, X_ts: TSCDataFrame):
        return self._pipeline.fit_transform(X_ts)

    def inverse_transform(self, X_ts: TSCDataFrame):
        return self._pipeline.inverse_transform(X_ts)


class EDMD(object):
    def __init__(self):
        pass


if __name__ == "__main__":
    from datafold.pcfold.timeseries.transform import (
        TSCQoiScale,
        TSCTakensEmbedding,
        TSCPrincipalComponents,
    )
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    time = np.linspace(0, 2 * np.pi, 100)
    df = pd.DataFrame(np.sin(time) + 10, index=time, columns=["sin"])
    tsc = TSCDataFrame.from_single_timeseries(df)

    _edmd_dict = EDMDDict(
        steps=(
            ("scale", TSCQoiScale(name="min-max")),
            ("delays", TSCTakensEmbedding(delays=10)),
            ("pca", TSCPrincipalComponents(n_components=2)),
        )
    )

    forward_dict = _edmd_dict.fit_transform(X_ts=tsc)
    inverse_dict = _edmd_dict.inverse_transform(X_ts=forward_dict)
    print(forward_dict)
    print(inverse_dict)

    ax = tsc.plot()
    inverse_dict.plot(ax=ax)
    plt.show()
