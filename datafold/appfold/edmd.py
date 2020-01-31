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
)
from datafold.pcfold.timeseries.metric import TSCMetric


class EDMDDict(Pipeline):

    # TODO: need to check that all steps are TSCtransformers! --> overwrite and super()
    #  _validate

    def __init__(self, steps, memory=None, verbose=True):
        """NOTE: the typing is different to the TSCTransformMixIn, Because this (meta-)
        transformer is used for DMD models.

        * in  fit a TSCDataFrame is required
        * in transform also initial conditions (pd.DataFrame or np.ndarray) are
          transformed
        """
        super(EDMDDict, self).__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X: TSCDataFrame, y=None, **fit_params):
        return super(EDMDDict, self).fit(X=X, y=y, **fit_params)

    def _transform(self, X: TRANF_TYPES):
        if isinstance(X, pd.Series):
            raise TypeError(
                "Currently, all pd.Series have to be casted to pd.DataFrame"
            )
        return super(EDMDDict, self)._transform(X=X)

    def fit_transform(self, X: TSCDataFrame, y=None, **fit_params):
        return super(EDMDDict, self).fit_transform(X=X, y=y, **fit_params)

    def _inverse_transform(self, X: TSCDataFrame):
        return super(EDMDDict, self)._inverse_transform(X=X)


class EDMD(Pipeline):
    def __init__(
        self, dict_steps, dmd_model: DMDBase = DMDFull(), memory=None, verbose=True
    ):

        self.dict_steps = dict_steps
        self.dmd_model = dmd_model

        all_steps = self.dict_steps + [("dmd", self.dmd_model)]
        super(EDMD, self).__init__(steps=all_steps, memory=memory, verbose=verbose)

    @property
    def edmd_dict(self):
        # TODO: not sure if it is better to make a getter?
        # probably better to do a deepcopy of steps
        return EDMDDict(steps=self.steps[:-1])

    def _inverse_transform_latent_time_series(self, X):
        reverse_iter = reversed(list(self._iter(with_final=False)))
        for _, _, transform in reverse_iter:
            X = transform.inverse_transform(X)
        return X

    def fit(self, X: PRE_FIT_TYPES, y=None, **fit_params) -> "Pipeline":
        print(f"fitting a model with {self.get_params()}")
        return super(EDMD, self).fit(X=X, y=y, **fit_params)

    def predict(self, X: PRE_IC_TYPES, t=None, **predict_params):
        # TODO. if X is an np.ndarray, it should be converted to a DataFrame that gives
        #  a description of the initial condition of the time series.

        if isinstance(X, pd.Series):
            X = pd.DataFrame(X).T
            X.index.names = [TSCDataFrame.IDX_ID_NAME, TSCDataFrame.IDX_TIME_NAME]

        if isinstance(X, np.ndarray) and X.ndim == 1:
            raise ValueError("1D arrays are ambiguous, input must be 2D")

        X_latent_ts = super(EDMD, self).predict(X=X, t=t, **predict_params)
        X_ts = self._inverse_transform_latent_time_series(X_latent_ts)
        return X_ts

    def fit_predict(self, X: TSCDataFrame, y=None, **fit_params):
        X_latent_ts = super(EDMD, self).fit_predict(X=X, y=y, **fit_params)
        X_ts = self._inverse_transform_latent_time_series(X_latent_ts)
        return X_ts

    def score(self, X: TSCDataFrame, y=None, sample_weight=None):

        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight

        X_latent_ic = Xt.initial_states_df()

        X_latent_ts = self.steps[-1][-1].predict(
            X=X_latent_ic, t=Xt.time_indices(unique_values=True)
        )

        X_est_ts = self._inverse_transform_latent_time_series(X_latent_ts)

        score_per_qoi = TSCMetric(metric="rmse", mode="qoi", scaling="min-max").score(
            y_true=X,
            y_pred=X_est_ts,
            sample_weight=sample_weight,
            multi_qoi="uniform_average",
        )

        assert isinstance(score_per_qoi, pd.Series)

        return float(score_per_qoi.mean())
