import os
import pickle
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

import datafold.dynfold.geometric_harmonics as gh
import datafold.pcfold.timeseries as ts
from datafold.dynfold.koopman import EDMDExact


class KoopmanSumo(object):

    def __init__(self, gh_options=None, gh_exist=None):
        # TODO: proper errors

        if (gh_options is None) + (gh_exist is None) != 1:  # ^ --> XOR
            raise ValueError("TODO: write error")  # TODO

        if gh_options is not None:
            if gh_options.get("is_stochastic", False):  # defaults to "not True" if not present
                import warnings
                warnings.warn("Currently it is not recommended to use is_stochastic=True, because the "
                              "out-of-sample does not work!")

            self.gh_interpolator_ = gh.GeometricHarmonicsFunctionBasis(**gh_options)
        else:
            self.gh_interpolator_ = gh_exist
            # TODO: for now, the fit() function has to be called already for the exist_gh -- check this!
            assert gh_exist.eigenvalues_ is not None and gh_exist.eigenvectors_ is not None

        self.edmd_ = None
        self.gh_coeff_ = None  # matrix that maps from the GH space to the QoI

    def _extract_dynamics_with_edmd(self, X):
        # transpose eigenvectors, because the eigenvectors are row-wise in pydmap
        gh_values = self.gh_interpolator_.eigenvectors_.T

        columns = [f"phi{i}" for i in range(gh_values.shape[1])]

        self.dict_data = ts.TSCDataFrame.from_same_indices_as(indices_from=X, values=gh_values, except_columns=columns)

        self.edmd_ = EDMDExact()
        self.edmd_ = self.edmd_.fit(self.dict_data, diagonalize=True)

    def _gh_coeff_with_least_square(self, X):
        # TODO: check residual somehow, user info, check etc.
        # Phi * C = D
        # obs_basis * data_coeff = data
        self.gh_coeff_, res = np.linalg.lstsq(self.dict_data, X, rcond=1E-14)[:2]

    def _set_X_info(self, X):
        if not X.is_const_frequency():
            raise ValueError("Only data with const. frequency is supported.")

        self._qoi_columns = X.columns

        self._time_interval = X.time_interval()
        self._normalize_frequency = X.frequency
        self._normalize_shift = self._time_interval[0]
        assert (self._time_interval[1] - self._normalize_shift) / self._normalize_frequency % 1 == 0
        self._max_normtime = int((self._time_interval[1] - self._normalize_shift) / self._normalize_frequency)

    def fit(self, X_ts: ts.TSCDataFrame):

        self._set_X_info(X_ts)

        # 1. transform data via GH-function basis
        # TODO: not call this if gh_interpolator was already fitted! --> Check
        self.gh_interpolator_ = self.gh_interpolator_.fit(X_ts.to_numpy())

        # 2. Compute Koopman matrix via EDMD
        self._extract_dynamics_with_edmd(X_ts)

        # 3. Linear map from new observable space to qoi data space
        self._gh_coeff_with_least_square(X_ts)

        return self

    def _compute_sumo_timeseries(self, initial_condition_gh: np.ndarray, eval_normtime):
        nr_qoi = len(self._qoi_columns)

        if initial_condition_gh.ndim == 1:
            initial_condition_gh = initial_condition_gh[np.newaxis, :]

        # This indexing is for C-aligned arrays
        # index order for "tensor[depth, row, column]"
        #     1) depth = timeseries (i.e. for respective initial condition),
        #     2) row = time step [k],
        #     3) column = qoi
        time_series_tensor = np.zeros([initial_condition_gh.shape[0], eval_normtime.shape[0], nr_qoi])

        # This loop solves the linear dynamical system with:
        # QoI_state_{k} = initial_condition vector in GH space @ (Koopman_matrix)^k @ back transformation to QoI space

        aux = initial_condition_gh @ self.edmd_.eigenvectors_right_  # can pre-compute

        for k, t in enumerate(eval_normtime):
            koopman_t = (aux * np.power(self.edmd_.eigenvalues_, t)) @ self.edmd_.eigenvectors_left_

            # TODO: need to check for too large imaginary part (the part is dropped here!)
            time_series_tensor[:, k, :] = np.real(koopman_t @ self.gh_coeff_)

        eval_usertime = eval_normtime * self._normalize_frequency + self._normalize_shift
        result_tc = ts.TSCDataFrame.from_tensor(time_series_tensor, columns=self._qoi_columns, time_index=eval_usertime)

        return result_tc

    @DeprecationWarning
    def _compute_sumo_timeseries_old(self, initial_condition_gh: np.ndarray, time_series_length: int):
        # This is efficient, but harder to read and extensible for different time values.
        # Maybe use later on again if required  -- for now use '_compute_sumo_timeseries'

        nr_qoi = len(self._qoi_columns)

        if initial_condition_gh.ndim == 1:
            initial_condition_gh = initial_condition_gh[np.newaxis, :]

        # This indexing is for C-aligned arrays
        # index order for "tensor[depth, row, column]"
        #     1) depth = timeseries (i.e. for respective initial condition),
        #     2) row = time step [k],
        #     3) column = qoi
        time_series_tensor = np.zeros([initial_condition_gh.shape[0], time_series_length, nr_qoi])

        # evec_right_incl_powered_evals stores the current right_koopman_eigenvec * koopman_eigenvalues^k
        # This is used to speed up the computation:
        # Koopman_matrix^k = (evec_right @ eval^k) @ evec_left
        #   -- where the part in brackets are stored in evec_right_incl_powered_evals
        evec_right_incl_powered_evals = np.copy(self.edmd_.eigenvectors_right_)

        # This loop solves the linear dynamical system with:
        # QoI_state_{k} = initial_condition vector in GH space @ (Koopman_matrix)^k @ back transformation to QoI space
        for k in range(time_series_length):
            koopman_matrix_k = evec_right_incl_powered_evals @ self.edmd_.eigenvectors_left_

            # k -> k+1 on the right Koopman eigenvectors
            if k != time_series_length-1:  # multiplication is not required for the last iteration
                evec_right_incl_powered_evals = np.multiply(evec_right_incl_powered_evals, self.edmd_.eigenvalues_,
                                                            out=evec_right_incl_powered_evals)

            koopman_matrix_k = np.real(koopman_matrix_k)

            # self.gh_coeff -- back transformation to QoI space
            # koopman_matrix_k -- Koopman matrix to the power of k
            # initial_condition_gh -- initial_condition vector in GH space
            time_series_tensor[:, k, :] = initial_condition_gh @ koopman_matrix_k @ self.gh_coeff_

        result_tc = ts.TSCDataFrame.from_tensor(time_series_tensor, columns=self._qoi_columns)
        result_tc = result_tc.tsc.normalize_time()  # TODO: check if this is required here...

        return result_tc

    def __call__(self, X_ic: np.ndarray, t):
        return self.predict_timeseries(X_ic, t)

    def predict_timeseries(self, X_ic, t=None):
        # TODO: check if initial_condition values are inside the training data range (manifold)

        if t is None:
            normtime = np.arange(0, self._max_normtime+1)
        elif isinstance(t, (float, int)):
            t = (t - self._normalize_shift) / self._normalize_frequency
            normtime = np.arange(0, t + 1)
        elif isinstance(t, np.ndarray):
            normtime = (t - self._normalize_shift) / self._normalize_frequency
        else:
            raise TypeError("")

        if isinstance(X_ic, (pd.Series, pd.DataFrame)):
            # TODO: Here should be a correct sorting (like in the data) and a check that the columns are identical
            X_ic = X_ic.to_numpy()

        initial_condition_gh = self.gh_interpolator_(X_ic)
        return self._compute_sumo_timeseries(initial_condition_gh, normtime)

    @staticmethod
    def _compare_train_data(sumo, Y_ts, use_exact_initial_condition=True):

        if not Y_ts.is_same_ts_length():
            raise NotImplementedError("Time series have to have same length.")

        time_series_length = Y_ts.lengths_time_series

        initial_condition_qoi = Y_ts.initial_states_df().to_numpy()
        if use_exact_initial_condition:  # use the exact initial conditions from the training data
            # warp in TSCDataFrame and use the `initial_states_df` function.
            gh_values = sumo.gh_interpolator_.eigenvectors_.T
            initial_condition_gh = ts.TSCDataFrame.from_same_indices_as(
                indices_from=Y_ts, values=gh_values, except_columns=np.arange(gh_values.shape[1]))
            initial_condition_gh = initial_condition_gh.initial_states_df().to_numpy()

        else:  # use the mapped initial condition with the GH interpolator
            initial_condition_gh = sumo.gh_interpolator_(initial_condition_qoi)

        reconstructed_time_series = sumo._compute_sumo_timeseries(initial_condition_gh, Y_ts.time_index_fill())

        # use same ids than from the original time series
        reconstructed_time_series.index = Y_ts.index.copy()
        return reconstructed_time_series

    def _set_error_metric_from_input(self, error_metric: str):
        # TODO: what are good time series metrics?

        if error_metric == "L2":
            error_metric = lambda y_true, y_pred: np.linalg.norm(y_true - y_pred, axis=0)
        elif error_metric == "rmse":
            error_metric = lambda y_true, y_pred: np.sqrt(metrics.mean_squared_error(
                y_true, y_pred, multioutput="raw_values"))
        elif error_metric == "mse":
            error_metric = metrics.mean_squared_error
        elif error_metric == "mae":
            error_metric = metrics.mean_absolute_error
        elif error_metric == "max":
            error_metric = metrics.max_error
        else:
            ValueError(f"Metric {error_metric} not known")
        return error_metric

    def score_timeseries(self, X_ic, Y_ts, sample_weight=None):
        Y_pred = self.predict_timeseries(X_ic, t=Y_ts.initial_df().time)
        metric = self._set_error_metric_from_input(error_metric="L2")
        return metric(Y_pred.to_numpy(), Y_ts.to_numpy())

    @staticmethod
    def reconstruction_error(sumo, Y_ts, return_per_time_series=False, return_per_qoi=False, return_per_time=False,
                             error_metric="L2", scaling="id"):

        assert return_per_time_series + return_per_qoi + return_per_time > 0

        # TODO: scaling: id (do nothing), normalize, min_max
        #  https://en.wikipedia.org/wiki/Normalization_(statistics)

        error_metric = sumo._set_error_metric_from_input(error_metric)

        # TODO: treat scaling here, according to input
        true_ts_collection = Y_ts.copy()  # has to be copied, because it is scaled
        true_traj_mean = Y_ts.mean()
        true_traj_std = Y_ts.std()

        # use_exact_initial_condition=False -> also interpolate the (known) initial conditions
        sumo_ts_collection = sumo._compare_train_data(sumo, Y_ts=Y_ts, use_exact_initial_condition=False)

        # TODO: scaling
        sumo_ts_collection = (sumo_ts_collection - true_traj_mean) / true_traj_std
        true_ts_collection = (true_ts_collection - true_traj_mean) / true_traj_std

        return_list = []

        # error per trajectory
        if return_per_time_series:
            error_per_traj = pd.DataFrame(np.nan,
                                          index=true_ts_collection.ids,
                                          columns=true_ts_collection.columns.to_list())

            for i, true_traj in true_ts_collection.itertimeseries():
                sumo_traj = sumo_ts_collection.loc[i, :]
                error_per_traj.loc[i, :] = error_metric(true_traj, sumo_traj)

            assert not np.any(error_per_traj.isna())
            return_list.append(error_per_traj)

        # error per quantity of interest
        if return_per_qoi:
            error_per_qoi = pd.Series(error_metric(true_ts_collection, sumo_ts_collection),
                                      index=true_ts_collection.columns)

            assert not np.any(error_per_qoi.isna())
            return_list.append(error_per_qoi)

        # error per time step
        if return_per_time:
            time_indices = true_ts_collection.time_indices(unique_values=True)
            assert np.all(time_indices == sumo_ts_collection.time_indices(unique_values=True))

            error_per_time = pd.DataFrame(np.nan,
                                          index=time_indices,
                                          columns=true_ts_collection.columns)

            for t in time_indices:
                true_traj = pd.DataFrame(true_ts_collection.loc[pd.IndexSlice[:, t], :])
                sumo_traj = pd.DataFrame(sumo_ts_collection.loc[pd.IndexSlice[:, t], :])

                error_per_time.loc[t, :] = error_metric(true_traj, sumo_traj)

            assert not np.any(error_per_time.isna())
            return_list.append(error_per_time)

        if len(return_list) == 1:
            return return_list[0]
        else:
            return return_list
