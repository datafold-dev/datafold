#!/usr/bin/env python3
from typing import Dict, Optional, Union

import numpy as np

from datafold.pcfold.timeseries import TSCDataFrame, allocate_time_series_tensor
from datafold.utils.datastructure import if1dim_colvec
from datafold.utils.maths import diagmat_dot_mat


class LinearDynamicalSystem:
    VALID_MODES = ("continuous", "discrete")

    def __init__(self, mode="continuous", time_invariant=True):
        self.mode = mode
        self.time_invariant = time_invariant

    def _check_time_samples(self, time_samples):

        if (time_samples < 0).any():
            raise ValueError("time samples contain negative values")

        if np.isnan(time_samples).any() or np.isinf(time_samples).any():
            raise ValueError("time samples contain invalid vales (nan/inf)")

        if self.mode == "discrete":

            if time_samples.dtype == np.integer:
                pass  # restrict time_sample
            elif (
                time_samples.dtype == np.floating
                and (np.mod(time_samples, 1) == 0).all()
            ):
                time_samples = time_samples.astype(np.int)
            else:
                raise TypeError(
                    "For mode=discrete the time_samples have to be integers"
                )

        return time_samples

    def _check_ic(self, ic, state_length):

        if ic.ndim == 1:
            ic = if1dim_colvec(ic)

        if ic.ndim != 2:
            raise ValueError(
                f"Parameter 'ic' must have 2 dimensions. Got ic.ndim={ic.ndim}"
            )

        if ic.shape[0] != state_length:
            raise ValueError(
                f"Mismatch in ic.shape[0]={ic.shape[0]} is not "
                f"dynmatrix.shape[1]={state_length}."
            )

        return ic

    def evolve_system_spectrum(
        self,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        dt: float,
        ic: np.ndarray,
        time_samples: np.ndarray,
        time_series_ids: Optional[Dict] = None,
        qoi_columns=None,
        post_map: Optional[np.ndarray] = None,
    ):
        """
           Parameters
           ----------
           eigenvectors
                Right eigenvectors of the dmatrix
           eigenvalues
                Eigenvalues of the matrix :math:`A` (see further comments below).
           dt
                Time difference between samples that resulted in the spectrum computation.
           ic
               Initial conditions corresponding to :math:`x_0`
           time_samples
               Array of times where the dynamical system should be evaluated.
           time_invariant
               If `True` the time at the initial condition is zero; ff `False` the
               time starts corresponds to the time_samples.
           post_map
               Maps the states of the system to another space. The dict has to contain
               a key "map" with a matrix, and qoi_columns for the mapped space.
           time_series_ids
               Time series ids in same order to initial conditions.

           Returns
           -------
           TSCDataFrame
               The resulting time series collection for each initial condition collected.

            #TODO properly explain:
            .. math::
                D A \exp(\lambda t) b

            where :math:`D * A` is the dynmatrix, where :math:`A` are the eigenvectors
            of the system. The matrix :math:`D` allows to linearly map the state.
           """

        if post_map is not None:
            if isinstance(post_map, np.ndarray) and post_map.ndim != 2:
                raise TypeError("TODO")
            dynmatrix = post_map @ eigenvectors
        else:
            dynmatrix = eigenvectors

        nr_qoi, state_length = dynmatrix.shape

        self._check_time_samples(time_samples)
        self._check_ic(ic, state_length=state_length)

        if time_series_ids is None:
            time_series_ids = np.arange(ic.shape[1])

        if qoi_columns is None:
            qoi_columns = np.arange(state_length)

        if len(qoi_columns) != nr_qoi:
            raise ValueError(
                f"len(qoi_columns)={qoi_columns} != state_length={state_length}"
            )

        # TODO: maybe revert to power() method!?
        #  see https://arxiv.org/pdf/1907.10807v2.pdf
        #  pdfp. 8 (eq. 3.3 and text below)
        omegas = np.log(eigenvalues.astype(np.complex)) / dt

        time_series_tensor = allocate_time_series_tensor(
            nr_time_series=ic.shape[1],
            nr_timesteps=time_samples.shape[0],
            nr_qoi=nr_qoi,
        )

        for idx, time in enumerate(time_samples):
            time_series_tensor[:, idx, :] = np.real(
                dynmatrix @ diagmat_dot_mat(np.exp(omegas * time), ic)
            ).T

        return TSCDataFrame.from_tensor(
            time_series_tensor,
            time_series_ids=time_series_ids,
            columns=qoi_columns,
            time_index=time_samples,
        )

    def evolve_edmd_forcing_system(
        self, edmd, tsc_forcing: TSCDataFrame, dynmatrix, eigfunc_interp
    ):
        if dynmatrix is None:
            dynmatrix = edmd.eigenvectors_right_

            # TODO: make a is fit request here to edmd to guarantee that EDMD was fit!
            assert dynmatrix is not None

        current_eigfunc_state = None

        for id_, ts in tsc_forcing.itertimeseries():
            initial_cond = True

            for t, row in ts.iterrows():
                if np.mod(t, 100) == 0:
                    print(t)

                row = row.to_numpy()[np.newaxis, :]
                if initial_cond:
                    current_eigfunc_state = np.linalg.lstsq(
                        edmd.eigenvectors_right_, eigfunc_interp(row), rcond=1e-15,
                    )[0]
                    initial_cond = False

                else:
                    bool_unforced_cols = np.isnan(row).ravel()
                    next_physical_state = np.real(dynmatrix @ current_eigfunc_state)

                    tsc_forcing.loc[(id_, t), bool_unforced_cols] = next_physical_state[
                        bool_unforced_cols
                    ]

                    corrected_physical_space = tsc_forcing.loc[(id_, t), :].to_numpy()[
                        np.newaxis, :
                    ]

                    current_eigfunc_state = eigfunc_interp(corrected_physical_space)

                    if (
                        np.isnan(current_eigfunc_state).any()
                        or np.isinf(current_eigfunc_state).any()
                    ):
                        import warnings

                        warnings.warn("Invalid interpolation values")

        return tsc_forcing
