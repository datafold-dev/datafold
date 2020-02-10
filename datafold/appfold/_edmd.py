"""This file contains code that is copied and modified from

scikit-learn
version 0.22.1.
repository: https://github.com/scikit-learn/scikit-learn/
project homepage: https://scikit-learn.org/stable/

Specifically, this applies to the following code parts:

Sources copied from original source (filepath, function)

*  from:  (sklearn/model_selection/_validation.py, _fit_and_score)
   to:    (datafold/appfold/_edmd.py, _fit_and_score_edmd)

*  from: sklearn/model_selection/_search.py BaseSearchCV.fit
   to:   (datafold/appfold/_edmd.py,  EDMDCV.fit())

For the datafold module "_edmd.py" (this file) the following license from the
scikit-learn project is added in addition to the datafold license (see LICENSE file)

-- scikit-learn license and copyright notice START

New BSD License

Copyright (c) 2007–2019 The scikit-learn developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

-- scikit-learn license and copyright notice END
"""


import numbers
import time
import warnings
from itertools import product
from traceback import format_exception_only

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import GridSearchCV, check_cv
from sklearn.model_selection._validation import is_classifier
from sklearn.utils import _message_with_time
from sklearn.utils.validation import _check_fit_params, check_is_fitted, indexable

from datafold.pcfold import TSCDataFrame
from datafold.pcfold.timeseries.base import TSCPredictMixIn
from datafold.pcfold.timeseries.metric import (
    TSCKfoldSeries,
    TSCKFoldTime,
)
from datafold.utils.datastructure import is_integer


def _split_X_edmd(X: TSCDataFrame, y, train_indices, test_indices):
    X_train, X_test = X.tsc.kfold_cv_reassign_ids(
        train_indices=train_indices, test_indices=test_indices
    )

    # TODO: make proper error, the folds are likely too small if assert fails
    assert isinstance(X_train, TSCDataFrame) and isinstance(X_test, TSCDataFrame)
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
                f" partition for these parameters will be set to {error_score}. "
                f"Details: \n{format_exception_only(type(e), e)[0]}",
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
                msg += f", {scorer_name}="
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


class EDMDCV(GridSearchCV, TSCPredictMixIn):
    def _validate_settings_edmd(self):
        # leave import here to avoid circular imports
        import datafold.appfold.edmd as edmd_typing

        if not isinstance(self.estimator, edmd_typing.EDMD):
            raise TypeError("EDMDCV only supports EDMD estimators.")

        if not isinstance(self.cv, (TSCKfoldSeries, TSCKFoldTime)):
            raise TypeError(f"cv must be of type {(TSCKfoldSeries, TSCKFoldTime)}")

    def _check_multiscore(self):
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

        return scorers, refit_metric

    def fit(self, X: TSCDataFrame, y=None, groups=None, **fit_params):

        self._validate_settings_edmd()
        X = self._validate_data(X)

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        scorers, refit_metric = self._check_multiscore()

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
                        f"Fitting {n_splits} folds for each of {n_candidates} candidates,"
                        f" totalling {n_candidates * n_splits} fits"
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
                        f"inconsistent results. Expected {n_splits} "
                        f"splits, got {len(out) // n_candidates}"
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

                if not is_integer(self.best_index_):
                    raise TypeError("best_index_ returned is not an integer")
                if self.best_index_ < 0 or self.best_index_ >= len(results["params"]):
                    raise IndexError("best_index_ index out of range")
            else:
                self.best_index_ = results[f"rank_test_{refit_metric}"].argmin()
                self.best_score_ = results[f"mean_test_{refit_metric}"][
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
        check_is_fitted(self)
        return self.best_estimator_.predict(X, t, **predict_params)
