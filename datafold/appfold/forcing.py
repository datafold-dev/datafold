from typing import Optional, Union


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import datafold.dynfold.diffusion_maps as operator
import datafold.pcfold.timeseries as ts
from datafold.dynfold.dmd import DMDEco, DMDFull
from datafold.dynfold.system_evolution import LinearDynamicalSystem
from datafold.pcfold.timeseries import TSCDataFrame


class ForcingKernelEigFuncDMD(object):
    # TODO: integrate DynamicalSystemEstimatorMixIn and finish work

    def __init__(
        self,
        normalize_strategy="id",
        eigfunc_name=None,
        eigfunc_kwargs=None,
        eigfunc_exist=None,
    ):

        if (eigfunc_kwargs is None) + (eigfunc_exist is None) != 1:
            raise ValueError(
                "Either provide argument or 'eig_func_kwargs' or 'eig_func_exist'"
            )

        if eigfunc_name is not None:
            if eigfunc_kwargs is None:
                eigfunc_kwargs = {}

            # call from with name
            self.eigfunc_interpolator = operator.TSCEigfuncInterpolator.from_operator_name(
                name=eigfunc_name, **eigfunc_kwargs
            )

        elif eigfunc_kwargs is not None:
            # call __init__
            self.eigfunc_interpolator = operator.TSCEigfuncInterpolator(
                **eigfunc_kwargs
            )
        else:
            self.eigfunc_interpolator = eigfunc_exist
            # TODO: for now, the fit() function has to be called already for the
            #  eigfunc_exist -- check this!
            assert (
                eigfunc_exist.eigenvalues_ is not None
                and eigfunc_exist.eigenvectors_ is not None
            )

        self.normalize_data = NormalizeQoi.check_normalize_qoi_strategy(
            strategy=normalize_strategy
        )

    def _extract_dynamics_with_edmd(self, X_ts: TSCDataFrame):
        # transpose eigenvectors, because the eigenvectors are row-wise in pydmap
        eig_func_values = self.eigfunc_interpolator.eigenvectors_.T

        columns = [f"phi{i}" for i in range(eig_func_values.shape[1])]
        self.dict_data = ts.TSCDataFrame.from_same_indices_as(
            indices_from=X_ts, values=eig_func_values, except_columns=columns
        )

        # TODO: provide as option --> include the identity observable state?
        # # # TODO: experimental: add const vector to eig_func_values
        # self.dict_data["const"] = 1
        # self.dict_data = pd.concat([self.dict_data, X_ts], axis=1)
        # # # TODO: end experimental

        self.edmd_ = DMDFull(is_diagonalize=False)
        self.edmd_ = self.edmd_.fit(self.dict_data)

    def _coeff_matrix_least_square(self, X):
        # TODO: check residual somehow, user info etc.
        # Phi * C = D
        # obs_basis * data_coeff = data
        self.coeff_matrix_, res = np.linalg.lstsq(self.dict_data, X, rcond=1e-14)[:2]

    def _compute_forcing_timeseries(self, X_forcing) -> TSCDataFrame:

        # Integrate the linear transformation back to the physical space
        # (via coefficients matrix) into the dynamical matrix of the linear dynamical
        # system.
        dynmatrix = self.coeff_matrix_.T @ self.edmd_.eigenvectors_right_

        tsc_result = LinearDynamicalSystem(
            mode="continuous"
        ).evolve_edmd_forcing_system(
            edmd=self.edmd_,
            tsc_forcing=X_forcing,
            dynmatrix=dynmatrix,
            eigfunc_interp=self.eigfunc_interpolator,
        )

        tsc_result, _ = tsc_result.tsc.undo_normalize_qoi(self.normalize_data)

        return tsc_result

    def fit(self, X_ts: ts.TSCDataFrame) -> "ForcingKernelEigFuncDMD":

        X_ts, self.normalize_data = X_ts.tsc.normalize_qoi(
            normalize_strategy=self.normalize_data
        )

        # is required to evaluate time
        self._fit_time_index = X_ts.time_values(unique_values=True)
        self._fit_qoi_columns = X_ts.columns

        # 1. transform data via GH-function basis
        # TODO: there should be a method provided by GH "is_fit" (look at sklearn)
        if not hasattr(self.eigfunc_interpolator, "eigenvectors_") and not hasattr(
            self.eigfunc_interpolator, "eigenvalues_"
        ):  # if not already fit...
            self.eigfunc_interpolator = self.eigfunc_interpolator.fit(X_ts.to_numpy())

        # 2. Compute Koopman matrix via EDMD
        self._extract_dynamics_with_edmd(X_ts)

        # 3. Linear map from new observable space to qoi data space
        self._coeff_matrix_least_square(X_ts)

        return self

    def __call__(self, X_ic: np.ndarray) -> TSCDataFrame:
        return self.predict(X_ic)

    def predict(self, X_force) -> TSCDataFrame:

        # Apply the same normalization as to the data that was fit
        # Cannot use the .tsc extension, because it is no time series collection.
        X_force, _ = NormalizeQoi(
            normalize_strategy=self.normalize_data, undo=False
        ).transform(X_force)

        tsc_predicted = self._compute_forcing_timeseries(X_force)

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
        Y_pred = self.predict(X_ic)

        tsc_error = TimeSeriesError(  # setup to compute the TimeSeriesError
            metric=metric, mode=mode, normalize_strategy=normalize_strategy
        )

        return tsc_error.eval_metric(
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
