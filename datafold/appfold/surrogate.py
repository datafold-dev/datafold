#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn import metrics

import datafold.dynfold.geometric_harmonics as gh
import datafold.pcfold.timeseries as ts
from datafold.dynfold.koopman import EDMDFull, evolve_linear_system


class KoopmanSumo(object):
    def __init__(self, gh_options=None, gh_exist=None):
        # TODO: proper errors

        if (gh_options is None) + (gh_exist is None) != 1:
            raise ValueError("Either provide argument 'gh_options' or 'gh_exist'")

        if gh_options is not None:
            if gh_options.get(
                "is_stochastic", False
            ):  # defaults to "not True" if not present
                import warnings

                warnings.warn(
                    "Currently it is not recommended to use is_stochastic=True, because the "
                    "out-of-sample (__call__) does not work!"
                )

            self.gh_interpolator_ = gh.GeometricHarmonicsFunctionBasis(**gh_options)
        else:
            self.gh_interpolator_ = gh_exist
            # TODO: for now, the fit() function has to be called already for the exist_gh -- check this!
            assert (
                gh_exist.eigenvalues_ is not None and gh_exist.eigenvectors_ is not None
            )

        self.edmd_ = None
        self.gh_coeff_ = None  # matrix that maps from the GH space to the QoI

    def _extract_dynamics_with_edmd(self, X):
        # transpose eigenvectors, because the eigenvectors are row-wise in pydmap
        gh_values = self.gh_interpolator_.eigenvectors_.T

        columns = [f"phi{i}" for i in range(gh_values.shape[1])]
        self.dict_data = ts.TSCDataFrame.from_same_indices_as(
            indices_from=X, values=gh_values, except_columns=columns
        )

        self.edmd_ = EDMDFull(is_diagonalize=True)
        self.edmd_ = self.edmd_.fit(self.dict_data)

    def _gh_coeff_with_least_square(self, X):
        # TODO: check residual somehow, user info etc.
        # Phi * C = D
        # obs_basis * data_coeff = data
        self.gh_coeff_, res = np.linalg.lstsq(self.dict_data, X, rcond=1e-14)[:2]

    def fit(self, X_ts: ts.TSCDataFrame):

        self._fit_time_index = X_ts.time_indices(
            unique_values=True
        )  # is required to evaluate time
        self._fit_qoi_columns = X_ts.columns

        # 1. transform data via GH-function basis
        # TODO: not call this if gh_interpolator was already fitted! --> Check
        self.gh_interpolator_ = self.gh_interpolator_.fit(X_ts.to_numpy())

        # 2. Compute Koopman matrix via EDMD
        self._extract_dynamics_with_edmd(X_ts)

        # 3. Linear map from new observable space to qoi data space
        self._gh_coeff_with_least_square(X_ts)

        return self

    def _compute_sumo_timeseries(self, initial_condition_gh: np.ndarray, time_samples):

        # Integrate the linear transformation back to the physical space (via gh coefficients) into the dynamical matrix
        # of the linear dynamical system.
        dynmatrix = self.gh_coeff_.T @ self.edmd_.eigenvectors_right_

        result_tc = evolve_linear_system(
            ic=initial_condition_gh.T,  # NOTE: i.c. orientation must be column-wise here
            time_samples=time_samples,
            edmd=self.edmd_,
            dynmatrix=dynmatrix,
            qoi_columns=self._fit_qoi_columns,
        )

        return result_tc

    def __call__(self, X_ic: np.ndarray, t):
        return self.predict_timeseries(X_ic, t)

    def predict_timeseries(self, X_ic, t=None):

        if t is None:
            time_samples = self._fit_time_index
        elif isinstance(t, (float, int)):
            time_samples = np.arange(0, t + 1)
        elif isinstance(t, np.ndarray):
            time_samples = t
        else:
            raise TypeError("")

        if isinstance(X_ic, (pd.Series, pd.DataFrame)):
            assert len(X_ic.columns) == len(self._fit_qoi_columns)
            X_ic = X_ic[
                self._fit_qoi_columns
            ].to_numpy()  # fails if the columns do not match

        initial_condition_gh = self.gh_interpolator_(X_ic)

        return self._compute_sumo_timeseries(initial_condition_gh, time_samples)

    @staticmethod
    def _compare_train_data(sumo, Y_ts, use_exact_initial_condition=True):

        if not Y_ts.is_same_ts_length():
            raise NotImplementedError("Time series have to have same length.")

        time_series_length = Y_ts.lengths_time_series

        initial_condition_qoi = Y_ts.initial_states_df().to_numpy()
        if (
            use_exact_initial_condition
        ):  # use the exact initial conditions from the training data
            # warp in TSCDataFrame and use the `initial_states_df` function.
            gh_values = sumo.gh_interpolator_.eigenvectors_.T
            initial_condition_gh = ts.TSCDataFrame.from_same_indices_as(
                indices_from=Y_ts,
                values=gh_values,
                except_columns=np.arange(gh_values.shape[1]),
            )
            initial_condition_gh = initial_condition_gh.initial_states_df().to_numpy()

        else:  # use the mapped initial condition with the GH interpolator
            initial_condition_gh = sumo.gh_interpolator_(initial_condition_qoi)

        reconstructed_time_series = sumo._compute_sumo_timeseries(
            initial_condition_gh, Y_ts.time_index_fill()
        )

        # use same ids than from the original time series
        reconstructed_time_series.index = Y_ts.index.copy()
        return reconstructed_time_series

    def _set_error_metric_from_input(self, error_metric: str):
        # TODO: what are good time series metrics?

        if error_metric == "L2":
            error_metric = lambda y_true, y_pred: np.linalg.norm(
                y_true - y_pred, axis=0
            )
        elif error_metric == "rmse":
            error_metric = lambda y_true, y_pred: np.sqrt(
                metrics.mean_squared_error(y_true, y_pred, multioutput="raw_values")
            )
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
    def reconstruction_error(
        sumo,
        Y_ts,
        return_per_time_series=False,
        return_per_qoi=False,
        return_per_time=False,
        error_metric="L2",
        scaling="id",
    ):

        assert return_per_time_series + return_per_qoi + return_per_time > 0

        # TODO: scaling: id (do nothing), normalize, min_max
        #  https://en.wikipedia.org/wiki/Normalization_(statistics)

        error_metric = sumo._set_error_metric_from_input(error_metric)

        # TODO: treat scaling here, according to input
        true_ts_collection = Y_ts.copy()  # has to be copied, because it is scaled
        true_traj_mean = Y_ts.mean()
        true_traj_std = Y_ts.std()

        # use_exact_initial_condition=False -> also interpolate the (known) initial conditions
        sumo_ts_collection = sumo._compare_train_data(
            sumo, Y_ts=Y_ts, use_exact_initial_condition=False
        )

        # TODO: scaling
        sumo_ts_collection = (sumo_ts_collection - true_traj_mean) / true_traj_std
        true_ts_collection = (true_ts_collection - true_traj_mean) / true_traj_std

        return_list = []

        # error per trajectory
        if return_per_time_series:
            error_per_traj = pd.DataFrame(
                np.nan,
                index=true_ts_collection.ids,
                columns=true_ts_collection.columns.to_list(),
            )

            for i, true_traj in true_ts_collection.itertimeseries():
                sumo_traj = sumo_ts_collection.loc[i, :]
                error_per_traj.loc[i, :] = error_metric(true_traj, sumo_traj)

            assert not np.any(error_per_traj.isna())
            return_list.append(error_per_traj)

        # error per quantity of interest
        if return_per_qoi:
            error_per_qoi = pd.Series(
                error_metric(true_ts_collection, sumo_ts_collection),
                index=true_ts_collection.columns,
            )

            assert not np.any(error_per_qoi.isna())
            return_list.append(error_per_qoi)

        # error per time step
        if return_per_time:
            time_indices = true_ts_collection.time_indices(unique_values=True)
            assert np.all(
                time_indices == sumo_ts_collection.time_indices(unique_values=True)
            )

            error_per_time = pd.DataFrame(
                np.nan, index=time_indices, columns=true_ts_collection.columns
            )

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
