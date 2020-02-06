#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# import not used in this file, but leave it such that
# 'from datafold.appfold.edmd import EDMDCV' works too
from datafold.appfold._edmd import EDMDCV
from datafold.dynfold.dmd import DMDBase, DMDFull
from datafold.pcfold import TSCDataFrame
from datafold.pcfold.timeseries.base import (
    PRE_FIT_TYPES,
    PRE_IC_TYPES,
    TRANF_TYPES,
    TSCTransformerMixIn,
)
from datafold.pcfold.timeseries.metric import TSCMetric, make_tsc_scorer


class EDMDDict(Pipeline):

    # TODO: need to check that all steps are TSCtransformers! --> overwrite and super()
    #  _validate

    def __init__(self, steps, memory=None, verbose=False):
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
        self, dict_steps, dmd_model: DMDBase = DMDFull(), memory=None, verbose=False
    ):

        self.dict_steps = dict_steps
        self.dmd_model = dmd_model

        self._setup_default_score_and_metric()

        all_steps = self.dict_steps + [("dmd", self.dmd_model)]
        super(EDMD, self).__init__(steps=all_steps, memory=memory, verbose=verbose)

    @property
    def edmd_dict(self):
        # TODO: not sure if it is better to make a getter?
        # probably better to do a deepcopy of steps
        return EDMDDict(steps=self.steps[:-1], memory=self.memory, verbose=self.verbose)

    def _setup_default_score_and_metric(self):
        self._metric_eval = TSCMetric.make_tsc_metric(
            metric="rmse", mode="qoi", scaling="min-max"
        )
        self._score_eval = make_tsc_scorer(self._metric_eval)

    def _inverse_transform_latent_time_series(self, X):
        reverse_iter = reversed(list(self._iter(with_final=False)))
        for _, _, transform in reverse_iter:
            X = transform.inverse_transform(X)
        return X

    def fit(self, X: PRE_FIT_TYPES, y=None, **fit_params) -> "Pipeline":

        assert X.is_const_dt()  # TODO: make proper error

        for (_, trans_str, transformer) in self._iter(with_final=False):
            if not isinstance(transformer, TSCTransformerMixIn):
                raise TypeError(
                    "Currently, in the pipeline only supports transformers "
                    "that can handle indexed data structures (pd.DataFrame "
                    "and TSCDataFrame)"
                )

        return super(EDMD, self).fit(X=X, y=y, **fit_params)

    def predict(self, X: PRE_IC_TYPES, t=None, **predict_params):
        # TODO: if X is an np.ndarray, it should be converted to a DataFrame that gives
        #  a description of the initial condition of the time series.

        if t is None:
            # Note it cannot simply take the time values from fit, as some time series
            # may have started at different times (esp. during cross fitting)
            raise NotImplementedError("")

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
        """Docu note: y is kept for consistency to sklearn, but should always be None."""
        assert y is None

        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight

        Xt = X

        # exclude last (the DMD model) because the
        for _, name, dict_step in self._iter(with_final=False):
            Xt = dict_step.transform(Xt)

        X_latent_ts_folds = []
        for X_latent_ic, time_values in Xt.tsc.initial_states_folds():
            current_ts = self.steps[-1][-1].predict(X=X_latent_ic, t=time_values)
            X_latent_ts_folds.append(current_ts)

        X_latent_ts = pd.concat(X_latent_ts_folds, axis=0)
        X_est_ts = self._inverse_transform_latent_time_series(X_latent_ts)

        # Important note for getting initial states in latent space:
        # during .transform() samples can be discarted (e.g. when applying Takens)
        # This means that in the latent space there can be less samples than in the
        # "pyhsical" space and this is corrected:
        if X.shape[0] > X_est_ts.shape[0]:
            X = X.select_times(time_points=X_est_ts.time_values(unique_values=True))

        return self._score_eval(X, X_est_ts, sample_weight)
