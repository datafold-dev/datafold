#!/usr/bin/env python3


import numbers
import time
import warnings
from itertools import product
from traceback import format_exception_only

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import GridSearchCV, check_cv
from sklearn.model_selection._validation import is_classifier
from sklearn.pipeline import Pipeline
from sklearn.utils import _message_with_time
from sklearn.utils.validation import _check_fit_params, indexable

from datafold.dynfold.dmd import DMDBase, DMDFull
from datafold.pcfold import TSCDataFrame
from datafold.pcfold.timeseries.base import (
    PRE_FIT_TYPES,
    PRE_IC_TYPES,
    TRANF_TYPES,
    TSCTransformerMixIn,
)
from datafold.pcfold.timeseries.metric import TSCKfoldSeries, TSCKFoldTime, TSCMetric


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
        """Docu note: y is kept for consistency , but y is basically X_test"""

        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)

        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight

        # get initial states in latent space.
        # important note: during .transform() samples can be discarted (e.g. when
        # applying Takens)
        X_latent_ic = Xt.initial_states_df()

        X_latent_ts = self.steps[-1][-1].predict(
            X=X_latent_ic, t=Xt.time_indices(unique_values=True)
        )

        X_est_ts = self._inverse_transform_latent_time_series(X_latent_ts)

        if X.shape[0] > X_est_ts.shape[0]:
            # Adapt X if time series samples are discareded during transform, and not
            # recovered during inverse_transform (e.g. for Takens)
            X = X.select_times(time_points=X_est_ts.time_indices(unique_values=True))

        score_per_qoi = TSCMetric(metric="rmse", mode="qoi", scaling="min-max").score(
            y_true=X,
            y_pred=X_est_ts,
            sample_weight=sample_weight,
            multi_qoi="uniform_average",
        )

        assert isinstance(score_per_qoi, pd.Series)

        return float(score_per_qoi.mean())


def _split_X_edmd(X: TSCDataFrame, y, train_indices, test_indices):
    X_train, X_test = X.tsc.kfold_cv_reassign_ids(
        train_indices=train_indices, test_indices=test_indices
    )

    # TODO: make proper error, the folds are likely too small...
    assert isinstance(X_train, TSCDataFrame) and (X_test, TSCDataFrame)

    return X_train, X_test


def _fit_and_score_edmd(
    edmd,
    X: TSCDataFrame,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    error_score=np.nan,
):
    """TODO: code copied and modified from scikit-learn -- add copyright"""

    if verbose > 1:
        if parameters is None:
            msg = ""
        else:
            msg = "%s" % (", ".join("%s=%s" % (k, v) for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * "."))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    train_scores = {}
    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        edmd = edmd.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, X_test = _split_X_edmd(X, y, train_indices=train, test_indices=test)

    try:
        edmd = edmd.fit(X_train, y, **fit_params)
    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn(
                "Estimator fit failed. The score on this train-test"
                " partition for these parameters will be set to %f. "
                "Details: \n%s" % (error_score, format_exception_only(type(e), e)[0]),
                FitFailedWarning,
            )
        else:
            raise ValueError(
                "error_score must be the string 'raise' or a"
                " numeric value. (Hint: if using 'raise', please"
                " make sure that it has been spelled correctly.)"
            )

    else:
        fit_time = time.time() - start_time
        test_scores = {"score": edmd.score(X_test, y=None)}

        score_time = time.time() - start_time - fit_time

        if return_train_score:
            train_scores = {"score": edmd.score(X_train, y=None)}

    if verbose > 2:
        if isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                msg += ", %s=" % scorer_name
                if return_train_score:
                    msg += "(train=%.3f," % train_scores[scorer_name]
                    msg += " test=%.3f)" % test_scores[scorer_name]
                else:
                    msg += "%.3f" % test_scores[scorer_name]
        else:
            msg += ", score="
            msg += (
                "%.3f" % test_scores
                if not return_train_score
                else "(train=%.3f, test=%.3f)" % (train_scores, test_scores)
            )

    if verbose > 1:
        total_time = score_time + fit_time
        print(_message_with_time("CV", msg, total_time))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(X_test.shape[0])
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(edmd)
    return ret


class EDMDCV(GridSearchCV):
    def _validate_settings_edmd(self):
        if not isinstance(self.estimator, EDMD):
            raise TypeError("EDMDCV only supports EDMD estimators.")

        if not isinstance(self.cv, (TSCKfoldSeries, TSCKFoldTime)):
            raise TypeError(f"cv must be of type {(TSCKfoldSeries, TSCKFoldTime)}")

    def fit(self, X: TSCDataFrame, y=None, groups=None, **fit_params):

        self._validate_settings_edmd()

        """TODO: code copied and modified from scikit-learn -- add copyright"""
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring
        )

        if self.multimetric_:
            if (
                self.refit is not False
                and (
                    not isinstance(self.refit, str)
                    or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers
                )
                and not callable(self.refit)
            ):
                raise ValueError(
                    "For multi-metric scoring, the parameter "
                    "refit must be set to a scorer key or a "
                    "callable to refit an estimator with the "
                    "best parameter setting on the whole "
                    "data and make the best_* attributes "
                    "available for that metric. If this is "
                    "not needed, refit should be set to "
                    "False explicitly. %r was passed." % self.refit
                )
            else:
                refit_metric = self.refit
        else:
            refit_metric = "score"

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch
        )

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score_edmd)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        **fit_and_score_kwargs,
                    )
                    for parameters, (train, test) in product(
                        candidate_params, cv.split(X, y, groups)
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, scorers, n_splits, all_out
                )
                return results

            self._run_search(evaluate_candidates)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError("best_index_ returned is not an integer")
                if self.best_index_ < 0 or self.best_index_ >= len(results["params"]):
                    raise IndexError("best_index_ index out of range")
            else:
                self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers["score"]

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def predict(self, X, t=None, **predict_params):
        # TODO: test that EDMDCV is fitted!
        return self.best_estimator_.predict(X, t, **predict_params)
