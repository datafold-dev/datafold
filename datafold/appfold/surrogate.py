#!/usr/bin/env python3

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import datafold.dynfold.diffusion_maps as dmap
import datafold.pcfold.timeseries as ts
from datafold.dynfold.dmd import DMDEco, DMDFull
from datafold.pcfold.timeseries import TSCDataFrame
from datafold.pcfold.timeseries.metric import TSCMetric
from datafold.pcfold.timeseries.transform import TSCQoiPreprocess
from datafold.utils.datastructure import if1dim_rowvec


class SumoKernelEigFuncDMD(object):
    def __init__(
        self,
        normalize_strategy="id",
        eigfunc_name=None,
        eigfunc_kwargs=None,
        eigfunc_exist=None,
    ):

        if (eigfunc_kwargs is None) + (eigfunc_exist is None) != 1:
            raise ValueError(
                "Either provide argument or 'eig_func_kwargs' or 'eigfunc_exist'."
            )

        if eigfunc_name is not None:
            if eigfunc_kwargs is None:
                eigfunc_kwargs = {}

            # call from with name
            self.eigfunc_interpolator = dmap.DiffusionMaps.from_operator_name(
                name=eigfunc_name, **eigfunc_kwargs
            )

        elif eigfunc_kwargs is not None:
            # call __init__
            self.eigfunc_interpolator = dmap.DiffusionMaps(**eigfunc_kwargs)
        else:
            self.eigfunc_interpolator = eigfunc_exist
            assert (
                eigfunc_exist.eigenvalues_ is not None
                and eigfunc_exist.eigenvectors_ is not None
            )

        if normalize_strategy == "id":
            self.qoi_scale_ = None
        else:
            self.qoi_scale_ = TSCQoiPreprocess.scale(normalize_strategy)

    def _extract_dynamics_with_edmd(self, X_ts: TSCDataFrame):
        # transpose eigenvectors, because the eigenvectors are row-wise in pydmap
        # eig_func_values = self.eigfunc_interpolator.eigenvectors_.T

        # columns = [f"phi{i}" for i in range(eig_func_values.shape[1])]
        # self.dict_data = ts.TSCDataFrame.from_same_indices_as(
        #     indices_from=X_ts, values=eig_func_values, except_columns=columns
        # )

        self.dict_data = self.eigfunc_interpolator.eigenvectors_

        # TODO: provide as option --> include the identity observable state?
        # # # TODO: experimental: add const vector to eig_func_values
        # self.dict_data["const"] = 1
        # self.dict_data = pd.concat([self.dict_data, X_ts], axis=1)
        # # # TODO: end experimental

        self.dmd_ = DMDFull(is_diagonalize=True)
        self.dmd_ = self.dmd_.fit(self.dict_data)

    def _coeff_matrix_least_square(self, X):
        # Phi * C = D
        # obs_basis * data_coeff = data
        self.coeff_matrix_, res = np.linalg.lstsq(self.dict_data, X, rcond=1e-14)[:2]

    def _compute_sumo_timeseries(self, X_ic_edmd, time_values) -> TSCDataFrame:

        tsc_result = self.dmd_.predict(
            X_ic_edmd,
            time_values=time_values,
            # post_map=self.coeff_matrix_.T,
            # qoi_columns=self._fit_qoi_columns,
        )

        tsc_result = self.eigfunc_interpolator.inverse_transform(tsc_result)

        if self.qoi_scale_ is not None:
            tsc_result = self.qoi_scale_.inverse_transform(tsc_result)

        return tsc_result

    def fit(self, X_ts: ts.TSCDataFrame) -> "SumoKernelEigFuncDMD":

        if self.qoi_scale_ is not None:
            X_ts = self.qoi_scale_.fit_transform(X_ts)

        # is required to evaluate time
        self._fit_time_index = X_ts.time_values(unique_values=True)
        self._fit_qoi_columns = X_ts.columns

        # 1. transform data via operator-function basis
        if not hasattr(self.eigfunc_interpolator, "eigenvectors_") and not hasattr(
            self.eigfunc_interpolator, "eigenvalues_"
        ):  # if not already fit...
            self.eigfunc_interpolator = self.eigfunc_interpolator.fit(X=X_ts)

        # 2. Compute Koopman matrix via DMD
        self._extract_dynamics_with_edmd(X_ts)

        # # 3. Linear map from new observable space to qoi data space
        # self._coeff_matrix_least_square(X_ts)

        return self

    def __call__(self, X_ic: np.ndarray, t) -> TSCDataFrame:
        return self.predict(X_ic, t)

    def _transform_dmd_ic(self, time_values, X_ic):

        # Time samples to evaluate the dmd model:
        if time_values is None:
            time_samples = self._fit_time_index
        elif isinstance(time_values, (float, int)):
            shift = self.dmd_.time_interval_[0]
            time_samples = np.arange(shift, shift + time_values + 1)
        elif isinstance(time_values, np.ndarray):
            time_samples = time_values
        else:
            raise TypeError(f"type(t)={type(time_values)} currently not supported.")

        # Initial condition for the DMD (needs to be transformed)
        if isinstance(X_ic, np.ndarray):
            X_ic = if1dim_rowvec(X_ic)
            nr_samples = X_ic.shape[0]

            idx = pd.MultiIndex.from_arrays(
                [np.arange(nr_samples), np.ones(nr_samples) * time_samples[0]],
                names=["ID", "initial_time"],
            )

            X_ic = pd.DataFrame(X_ic, index=idx, columns=self._fit_qoi_columns)

        elif isinstance(X_ic, pd.DataFrame):
            assert len(X_ic.columns) == len(self._fit_qoi_columns)

            # Re-organize columns, if required. Fails if the column names do not match.
            X_ic = X_ic[self._fit_qoi_columns]

        else:
            raise TypeError(
                f"type={type(X_ic)} is currently not supported to describe initial "
                f"conditions."
            )

        if self.qoi_scale_ is not None:
            # Apply the same normalization as to the data that was fit
            # Cannot use the .tsc extension, because it is no time series collection.
            X_ic = self.qoi_scale_.transform(X_ic)

        # Transform the initial conditions to obervable functions evaluations
        ic_obs_space = self.eigfunc_interpolator.transform(X_ic.to_numpy())

        X_ic_dmd = pd.DataFrame(
            if1dim_rowvec(ic_obs_space),
            index=X_ic.index,
            columns=self.dict_data.columns,
        )

        return time_samples, X_ic_dmd

    def predict(self, X_ic, time_values=None) -> TSCDataFrame:

        time_samples, X_ic_dmd = self._transform_dmd_ic(
            time_values=time_values, X_ic=X_ic
        )

        tsc_predicted = self._compute_sumo_timeseries(
            X_ic_edmd=X_ic_dmd, time_values=time_samples,
        )

        return tsc_predicted

    def score(
        self,
        X_ic,
        Y_ts: TSCDataFrame,
        metric="rmse",
        normalize_strategy="id",
        mode="qoi",
        sample_weight: Optional[np.ndarray] = None,
        multi_qoi: Union[str, np.ndarray] = "uniform_average",
    ):

        time_values = Y_ts.time_values(unique_values=True)
        Y_pred = self.predict(X_ic, time_values=time_values)

        tsc_metric = TSCMetric(metric=metric, mode=mode, scaling=normalize_strategy)
        return tsc_metric.eval_metric(
            y_true=Y_ts, y_pred=Y_pred, sample_weight=sample_weight, multi_qoi=multi_qoi
        )

    def reconstruction_score(
        self, Y_ts: TSCDataFrame, **kwargs,
    ):
        """
        Computes the reconstruction error (also: training error). It is discourages to
        use this error as a score to optimize the model, e.g. using cross-validation.
        There is a high risk of over-fitting.

        Parameters
        ----------
        Y_ts
            Data used for fitting the model.

        kwargs
            Arguments handled to
            :py:meth:`datafold.appfold.surrogate.SumoKernelEigFuncDMD.score` method.

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            Reconstruction error.

        """

        X_ic = Y_ts.initial_states_df()
        return self.score(X_ic, Y_ts, **kwargs)
