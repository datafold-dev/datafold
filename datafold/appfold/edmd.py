#!/usr/bin/env python3

import copy
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from datafold.appfold._edmd import EDMDCV  # do not remove, even if not used here
from datafold.dynfold.base import (
    PRE_FIT_TYPES,
    PRE_IC_TYPES,
    TRANF_TYPES,
    TSCPredictMixIn,
    TSCTransformerMixIn,
)
from datafold.dynfold.dmd import DMDBase, DMDFull
from datafold.pcfold import TSCDataFrame


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
        self,
        dict_steps: List[Tuple],
        dmd_model: DMDBase = DMDFull(),
        include_id_state=True,
        memory=None,
        verbose=False,
    ):

        self.dict_steps = dict_steps
        self._dmd_model = dmd_model
        self.include_id_state = include_id_state

        # TODO: if necessary provide option to give user defined metric
        self._setup_default_tsc_metric_and_score()

        all_steps = self.dict_steps + [("dmd", self._dmd_model)]
        super(EDMD, self).__init__(steps=all_steps, memory=memory, verbose=verbose)

    @property
    def dmd_model(self):
        # Improves code readability when using  attribute
        # 'dmd_model' instead of general 'final_estimator'
        return self._final_estimator

    def get_edmd_dict(self):
        # probably better to do a deepcopy of steps
        return EDMDDict(
            steps=copy.deepcopy(self.steps[:-1]),
            memory=self.memory,
            verbose=self.verbose,
        )

    def _transform_original2dictionary(self, X):
        """Forward transformation of dictionary."""

        if self.include_id_state:
            # copy required to properly attach X later on
            X_dict = X.copy(deep=True)
        else:
            X_dict = X

        # carry out dictionary transformations:
        for _, name, transform in self._iter(with_final=False):
            X_dict = transform.transform(X_dict)

        if self.include_id_state:
            X_dict = self._attach_id_state(X=X, X_dict=X_dict)

        return X_dict

    def _transform_dictionary2original(self, X):
        """Inverse transformation. """

        if self.include_id_state:
            # simply take the carried original columns that were carried along:
            X_ts = X.loc[:, self.features_in_[1]]
        else:
            # it is required to inverse_transform, because the initial states are not
            # available:
            X_ts = X
            reverse_iter = reversed(list(self._iter(with_final=False)))
            for _, _, transform in reverse_iter:
                X_ts = transform.inverse_transform(X_ts)

        return X_ts

    def _compute_n_samples_ic(self, X, X_dict):
        diff = X_dict.n_timesteps - X.n_timesteps

        if isinstance(diff, pd.Series):
            # time series can have different number of time values (in which case it is
            # a Series), however, the the difference has to be the same for all time
            # series
            assert (diff.iloc[0] == diff).all()
            diff = diff.iloc[0]

        # +1 because the diff indicates how many samples were removed -- we want the
        # number that is required for the initial condition
        return int(diff) + 1

    def _validate_type_and_n_samples_ic(self, X_ic):

        if self.n_samples_ic_ == 1:
            if isinstance(X_ic, TSCDataFrame):
                raise TypeError(
                    "The n_samples to define an inital condition ("
                    "n_samples_ic_={}) is incorrect. Got a time series "
                    "collection with minimum 2 samples per time series."
                )
        else:  # self.n_samples_ic_ > 1
            if not isinstance(X_ic, TSCDataFrame):
                raise TypeError(
                    "For the initial condition a TSCDataFrame is required, "
                    f"with {self.n_samples_ic_} (n_samples_ic_) samples per initial "
                    f"condition. Got type={type(X_ic)}."
                )

            if not (X_ic.n_timesteps > self.n_samples_ic_).all():
                raise ValueError(
                    f"For each initial condition exactly {self.n_samples_ic_} samples "
                    f"(attribute n_samples_ic_) are required. Got: \n {X_ic.n_timesteps}"
                )

    def _attach_id_state(self, X, X_dict):
        # remove states from X (the id-states) that are also removed during dictionary
        # transformations
        X = X.loc[X_dict.index, :]
        return pd.concat([X, X_dict], axis=1)

    def fit(self, X: PRE_FIT_TYPES, y=None, **fit_params):

        for (_, trans_str, transformer) in self._iter(with_final=False):
            if not isinstance(transformer, TSCTransformerMixIn):
                raise TypeError(
                    "Currently, in the pipeline only supports transformers "
                    "that can handle indexed data structures (pd.DataFrame "
                    "and TSCDataFrame)"
                )

        self._validate_data(
            X,
            ensure_feature_name_type=True,
            validate_tsc_kwargs={"ensure_const_delta_time": True},
        )
        self._setup_features_and_time_fit(X)

        # NOTE: internal from sklearn!
        from sklearn.utils import _print_elapsed_time

        # calls internally fit_transform (!!), and stores results into cache if
        # "self.memory is not None" (see docu)
        X_dict, fit_params = self._fit(X, y, **fit_params)

        self.n_samples_ic_ = self._compute_n_samples_ic(X, X_dict)

        if self.include_id_state:
            X_dict = self._attach_id_state(X=X, X_dict=X_dict)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            self.dmd_model.fit(X=X_dict, y=y, **fit_params)

        return self

    def predict(self, X: PRE_IC_TYPES, time_values=None, **predict_params):

        check_is_fitted(self)

        self._validate_data(
            X, ensure_feature_name_type=True,
        )

        X, time_values = self._validate_features_and_time_values(
            X=X, time_values=time_values
        )

        self._validate_type_and_n_samples_ic(X_ic=X)

        X_dict = self._transform_original2dictionary(X)

        # this needs to always hold if the checks _validate_type_and_n_samples_ic are
        #  correct
        assert isinstance(X_dict, pd.DataFrame)

        # now we compute the time series in "dictionary space":
        X_latent_ts = self._dmd_model.predict(
            X_dict, time_values=time_values, **predict_params
        )

        # transform from "dictionary space" to "user space"
        X_ts = self._transform_dictionary2original(X_latent_ts)

        return X_ts

    def reconstruct(self, X: TSCDataFrame):
        # TODO: after solving the issue "how many samples required for initial
        #  condition, then this function can be improved computationally. Currently,
        #  the entire time series is transformed, but actually only the I.C. has to!

        check_is_fitted(self)
        X = self._validate_data(X)
        self._validate_feature_names(X)

        # TODO: could that not be read from fit? Check what happens in
        #  super()._fit (which is called during fit)
        #  --> possibly there is no need to transform a second time.
        Xt = self._transform_original2dictionary(X)

        # extract time series with different initial times
        # NOTE: usually for the DMD model is irrelevant (as it treads every initial
        # condition as time=0, but to correctly shift the time back for the
        # reconstruction the time series having different initial times are evaluated
        # separately.
        # NOTE2: this feature is especially impo withrtant for cross-validation, where the
        # folds are split in time
        X_latent_ts_folds = []
        # TODO: think of better name for initial_states_folds
        for X_latent_ic, X_time_values in Xt.tsc.initial_states_folds():
            current_ts = self._dmd_model.predict(
                X=X_latent_ic, time_values=X_time_values
            )
            X_latent_ts_folds.append(current_ts)

        X_latent_ts = pd.concat(X_latent_ts_folds, axis=0)
        X_est_ts = self._transform_dictionary2original(X_latent_ts)
        # NOTE: time series contained in X_est_ts can be shorter in length, as some
        # TSCTransform models drop samples (e.g. Takens)
        return X_est_ts

    def fit_reconstruct(self, X: TSCDataFrame, **fit_params):
        # TODO: this is currently very costly, as it carries out "transform" in fit and
        #  in reconstruct. In fit() the fit_transform() is called and transform() is
        #  also called. So somehow X_dict from fit() can be further used!
        return self.fit(X, **fit_params).reconstruct(X)

    def fit_predict(self, X, y=None, **fit_params):
        raise NotImplementedError(
            "Not implemented for EDMD. Look at 'fit_reconstruct' which is similar and "
            "better addresses for time series data."
        )

    def score(self, X: TSCDataFrame, y=None, sample_weight=None):
        """Docu note: y is kept for consistency to sklearn, but should always be None."""
        assert y is None
        self._check_attributes_set_up(check_attributes=["score_eval"])

        # does all the checkings:
        X_est_ts = self.reconstruct(X)

        # Important note for getting initial states in latent space:
        # during .transform() samples can be discarded (e.g. when applying Takens)
        # This means that in the latent space there can be less samples than in the
        # "physical" space and this is corrected:
        if X.shape[0] > X_est_ts.shape[0]:
            X = X.loc[X_est_ts.index, :]

        return self.score_eval(X, X_est_ts, sample_weight)
