#!/usr/bin/env python3


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from datafold.dynfold.dmd import DMDBase, DMDFull
from datafold.pcfold import TSCDataFrame
from datafold.pcfold.timeseries.base import (
    PRE_FIT_TYPES,
    PRE_IC_TYPES,
    TRANF_TYPES,
    TSCPredictMixIn,
    TSCTransformMixIn,
)
from datafold.pcfold.timeseries.transform import TSCIdentity


class EDMDDict(TSCTransformMixIn):
    def __init__(self, steps=None):
        """NOTE: the typing is different to the TSCTransformMixIn, Because this (meta-)
        transformer is used for DMD models.

        * in  fit a TSCDataFrame is required
        * in transform also initial conditions (pd.DataFrame or np.ndarray) are
          transformed
        . """

        if steps is None:
            steps = [("id", TSCIdentity())]

        self._pipeline = Pipeline(steps=steps, memory=None, verbose=True)

    def fit(self, X: TSCDataFrame, y=None, **fit_params):
        self._pipeline.fit(X=X, y=y, **fit_params)
        return self

    def transform(self, X: TRANF_TYPES):
        if isinstance(X, pd.Series):
            raise TypeError(
                "Currently, all pd.Series have to be casted to pd.DataFrame"
            )
        return self._pipeline.transform(X=X)

    def fit_transform(self, X: TSCDataFrame, y=None, **fit_params):
        return self._pipeline.fit_transform(X=X, y=y, **fit_params)

    def inverse_transform(self, X: TSCDataFrame):
        return self._pipeline.inverse_transform(X)


class EDMD(TSCPredictMixIn):
    def __init__(self, steps=None, dmd_model: DMDBase = DMDFull()):
        self.dmd_dict = EDMDDict(steps=steps)
        self.dmd_model = dmd_model

    def fit(self, X: PRE_FIT_TYPES, **fit_params) -> "EDMD":
        X_latent = self.dmd_dict.fit_transform(X)
        self.dmd_model.fit(X=X_latent)
        return self

    def predict(self, X: PRE_IC_TYPES, t, **predict_params):
        # TODO. if X is an np.ndarray, it should be converted to a DataFrame that gives
        #  a description of the initial condition of the time series.

        if isinstance(X, pd.Series):
            X = pd.DataFrame(X).T
            X.index.names = [TSCDataFrame.IDX_ID_NAME, TSCDataFrame.IDX_TIME_NAME]

        if isinstance(X, np.ndarray) and X.ndim == 1:
            raise ValueError("1D arrays are ambiguous, input must be 2D")

        X_latent_ic = self.dmd_dict.transform(X)
        X_latent_ts = self.dmd_model.predict(X=X_latent_ic, t=t, **predict_params)
        X_ts = self.dmd_dict.inverse_transform(X_latent_ts)

        return X_ts
