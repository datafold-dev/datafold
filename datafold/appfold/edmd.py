#!/usr/bin/env python3

import copy

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
    TSCPredictMixIn,
    TSCTransformerMixIn,
)
from datafold.pcfold.timeseries.metric import TSCMetric, make_tsc_scorer


class EDMDDict(Pipeline):
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
                "Currently, all pd.Series have to be casted to pd.DataFrame before."
            )
        return super(EDMDDict, self)._transform(X=X)

    def fit_transform(self, X: TSCDataFrame, y=None, **fit_params):
        return super(EDMDDict, self).fit_transform(X=X, y=y, **fit_params)

    def _inverse_transform(self, X: TSCDataFrame):
        return super(EDMDDict, self)._inverse_transform(X=X)


class EDMD(Pipeline, TSCPredictMixIn):
    def __init__(
        self, dict_steps, dmd_model: DMDBase = DMDFull(), memory=None, verbose=False
    ):

        self.dict_steps = dict_steps
        self.dmd_model = dmd_model

        self._setup_default_score_and_metric()

        all_steps = self.dict_steps + [("dmd", self.dmd_model)]
        super(EDMD, self).__init__(steps=all_steps, memory=memory, verbose=verbose)

    def get_edmd_dict(self):
        # probably better to do a deepcopy of steps
        return EDMDDict(
            steps=copy.deepcopy(self.steps[:-1]),
            memory=self.memory,
            verbose=self.verbose,
        )

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

    def fit(self, X: PRE_FIT_TYPES, y=None, **fit_params) -> "EDMD":

        for (_, trans_str, transformer) in self._iter(with_final=False):
            if not isinstance(transformer, TSCTransformerMixIn):
                raise TypeError(
                    "Currently, in the pipeline only supports transformers "
                    "that can handle indexed data structures (pd.DataFrame "
                    "and TSCDataFrame)"
                )

        self._setup_features_and_time_fit(X)

        return super(EDMD, self).fit(X=X, y=y, **fit_params)

    def predict(self, X: PRE_IC_TYPES, time_values=None, **predict_params):

        # TODO: when applying Takens, etc. an initial condition in X must be a time
        #  series. During fit there should be a way to compute how many samples are
        #  needed to make one IC -- this would allow a good error msg. here if X does
        #  not meet this requirement here.

        if isinstance(X, pd.Series):
            raise TypeError(f"X has to be of the following types {PRE_IC_TYPES}")

        if isinstance(X, np.ndarray):
            # TODO: this requires some assumptions, especially if
            raise NotImplementedError("make proper handling of np.ndarray input later")

        if time_values is None:
            time_values = self.time_values_in_[1]

        self._validate_features_and_time_values(X=X, time_values=time_values)

        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)

        # TODO: needs a better check...
        assert isinstance(Xt, pd.DataFrame), (
            "at the lowest level there must be only one "
            "sample per IC (at the highest level, "
            "many samples may be required. There is a "
            "proper check required. "
        )

        X_latent_ts = self.steps[-1][-1].predict(Xt, **predict_params)
        X_ts = self._inverse_transform_latent_time_series(X_latent_ts)

        return X_ts

    def reconstruct(self, X: TSCDataFrame):
        # TODO: here a valuable parameter can be inferred:
        #  how many time samples are required to make a I.C.?

        self._validate_data(X)
        self._validate_feature_names(X)

        Xt = X

        # exclude last (the DMD model) and transform input data
        for _, name, dict_step in self._iter(with_final=False):
            Xt = dict_step.transform(Xt)

        # extract time series with different initial times
        # NOTE: usually for the DMD model is irrelevant (as it treads every initial
        # condition as time=0, but to correctly shift the time back for the
        # reconstruction athe time series are called separately.
        # NOTE2: this feature is especially important for cross-validation, where the
        # folds are split in time
        X_latent_ts_folds = []
        for X_latent_ic, time_values in Xt.tsc.initial_states_folds():
            current_ts = self.steps[-1][-1].predict(
                X=X_latent_ic, time_values=time_values
            )
            X_latent_ts_folds.append(current_ts)

        X_latent_ts = pd.concat(X_latent_ts_folds, axis=0)
        X_est_ts = self._inverse_transform_latent_time_series(X_latent_ts)
        # NOTE: time series contained in X_est_ts can be shorter in length, as some
        # TSCTransform models drop samples (e.g. Takens)
        return X_est_ts

    def fit_reconstruct(self, X: TSCDataFrame, **fit_params):
        return self.fit(X, **fit_params).reconstruct(X)

    def score(self, X: TSCDataFrame, y=None, sample_weight=None):
        """Docu note: y is kept for consistency to sklearn, but should always be None."""
        assert y is None

        # does the checking:
        X_est_ts = self.reconstruct(X)

        # Important note for getting initial states in latent space:
        # during .transform() samples can be discarted (e.g. when applying Takens)
        # This means that in the latent space there can be less samples than in the
        # "physical" space and this is corrected:
        if X.shape[0] > X_est_ts.shape[0]:
            X = X.loc[X_est_ts.index, :]

        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight

        return self._score_eval(X, X_est_ts, sample_weight)
