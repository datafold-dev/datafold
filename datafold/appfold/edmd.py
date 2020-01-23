#!/usr/bin/env python3


from sklearn.pipeline import Pipeline
from datafold.pcfold import TSCDataFrame
from datafold.pcfold.timeseries.transform import TSCIdentity


class EDMDDict(object):
    def __init__(self, steps=None):
        if steps is None:
            steps = [("id", TSCIdentity())]

        self._pipeline = Pipeline(steps=steps, memory=None, verbose=True)

    def fit(self, X_ts: TSCDataFrame, y=None, **fit_params):
        self._pipeline.fit(X=X_ts, y=y, **fit_params)
        return self

    def transform(self, X_ts: TSCDataFrame):
        return self._pipeline.transform(X=X_ts)

    def fit_transform(self, X_ts: TSCDataFrame, y=None, **fit_params):
        return self._pipeline.fit_transform(X=X_ts, y=y, **fit_params)

    def inverse_transform(self, X_ts: TSCDataFrame):
        return self._pipeline.inverse_transform(X_ts)


class EDMD(object):
    def __init__(self, steps=None):
        pass

    def predict(self, X_ic):
        pass
