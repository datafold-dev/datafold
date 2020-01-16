import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, ridge_regression

from datafold.dynfold.system_evolution import LinearDynamicalSystem
from datafold.pcfold.timeseries import TSCDataFrame, allocate_time_series_tensor
from datafold.utils.maths import diagmat_dot_mat, mat_dot_diagmat, sort_eigenpairs

import pydmd


class DMDBase(BaseEstimator):
    r"""Dynamic Mode Decomposition (DMD) approximates the Koopman operator with
    a matrix :math:`K`.

    The Koopman matrix :math:`K` defines a linear dynamical system of the form

    .. math::
        K x_k = x_{k+1} \\
        K^k x_0 = x_k

    where :math:`x_k` is the (column) state vector of the system at time :math:`k`. All
    subclasses should provide the right eigenvectors :math:`\Psi_r` and corresponding
    eigenvalues :math:`\omega` of the Koopman matrix to efficiently evolve the linear
    system TODO: link to method

    Evolving the linear system over many time steps is expensive due to the matrix power:

    .. math::
        K^k x_0 &= x_k

    Therefore, the (right) eigenpairs (:math:`\omega, \Psi_r`) of the Koopman matrix is
    computed and the initial condition is written in terms of the eigenvectors in a
    least-squares sense.

    .. math::
        K^k \Psi_r &= \Psi_r \Omega \\
        \Psi_r b &= x_0 \\
        K^k \Psi_r b &= x_k \\
        \Psi_r \Omega^k b &= x_k

    where the eigenproblem is stated in matrix form for all computed eigenpairs. Because
    :math:`\Omega` is a a diagonal matrix the power is very cheap compared to the
    generally full matrix :math:`K^k`.
    """

    def __init__(self):
        self.koopman_matrix_ = None
        self.eigenvalues_ = None
        self.eigenvectors_right_ = None

    def _set_X_info(self, X):

        if not X.is_const_dt():
            raise ValueError("Only data with constant frequency is supported.")

        self.dt_ = X.dt
        self._qoi_columns = X.columns

        self._time_interval = X.time_interval()
        self._normalize_shift = self._time_interval[0]

        assert (
            np.around(
                (self._time_interval[1] - self._normalize_shift) / self.dt_, decimals=5
            )
            % 1
            == 0
        )
        self._max_normtime = int(
            (self._time_interval[1] - self._normalize_shift) / self.dt_
        )

    def fit(self, X_ts, **fit_params):
        self._set_X_info(X_ts)

    def predict(self, X_ic, t, **predict_params):

        if t.ndim != 1:
            raise ValueError("TODO")

        if (t < 0).any():
            raise ValueError("TODO")

        t = np.sort(t)

        post_map = predict_params.pop("post_map", None)
        qoi_columns = predict_params.pop("qoi_columns", None)

        if len(predict_params.keys()) > 0:
            raise ValueError("TODO")

        if len(np.unique(X_ic.index.get_level_values("initial_time"))) != 1:
            raise NotImplementedError(
                "Currently alls initial conditions have to have "
                "the same initial time."
            )

        return self._evolve_edmd_system(
            X_ic=X_ic, time_samples=t, post_map=post_map, qoi_columns=qoi_columns
        )

    def _evolve_edmd_system(
        self,
        X_ic: pd.DataFrame,
        time_samples: np.ndarray,
        time_invariant=True,
        post_map: Optional[np.ndarray] = None,
        qoi_columns=None,
    ):
        """
        Evolce the linear system.

        Parameters
        ----------
        ic
            Initial condition in same space where EDMD was fit. The initial condition may
            be transformed internally.
        time_samples
            Array of times where the dynamical system should be evaluated.
        dynmatrix
            If not provided, the dynmatrix corresponds to the eigenvectors of EDMD.
        time_series_ids
            Time series ids in same order to initial conditions.
        qoi_columns
            List of quantity of interest names that are set in the TSCDataFrame
            returned.

        Returns
        -------
        TSCDataFrame
            The resulting time series collection for each initial condition collected.

        """

        if not np.isin(X_ic.columns, self._qoi_columns).all():
            raise ValueError("TODO")
        else:
            # sort, just in case they are given in a different order
            X_ic = X_ic.loc[:, self._qoi_columns]

        if qoi_columns is None:
            qoi_columns = self._qoi_columns

        # initial condition is numerical only, from now on
        ic = X_ic.to_numpy().T
        time_series_ids = X_ic.index.get_level_values("ID").to_numpy()

        # Choose alternative of how to evolve the linear system:
        if hasattr(self, "eigenvectors_left_") and (
            self.eigenvectors_left_ is not None and self.eigenvectors_right_ is not None
        ):
            # uses both eigenvectors (left and right). Used if is_diagonalize=True in
            # DMDFull
            ic = self.eigenvectors_left_ @ ic
        elif (
            hasattr(self, "eigenvectors_right_")
            and self.eigenvectors_right_ is not None
        ):
            # represent the initial condition in terms of right eigenvectors (by solving a
            # least-squares problem) -- only the right eigenvectors are required
            ic = np.linalg.lstsq(self.eigenvectors_right_, ic, rcond=1e-15)[0]

        else:
            # TODO: make this closer to scikit-learn (a "is_fit" function), probably best to
            #  do this at top of function
            raise RuntimeError("EDMD is not properly fit.")

        if time_invariant:
            shift = np.min(time_samples)
        else:
            # If the edmd time is shifted during data (e.g. the minimum processed data
            # starts with time=5, some positive value) then normalize the time_samples
            # with this shift. The linear system handles the shifted time start as time
            # zero.
            shift = self._normalize_shift

        norm_time_samples = time_samples - shift

        tsc_df = LinearDynamicalSystem(
            mode="continuous", time_invariant=True
        ).evolve_system_spectrum(
            eigenvectors=self.eigenvectors_right_,
            eigenvalues=self.eigenvalues_,
            dt=self.dt_,
            ic=ic,
            post_map=post_map,
            time_samples=norm_time_samples,
            time_series_ids=time_series_ids,
            qoi_columns=qoi_columns,
        )

        # correct the time shift again to return the correct time according to the
        # training data
        tsc_df = tsc_df.tsc.shift_time(shift_t=shift)

        return tsc_df


class DMDFull(DMDBase):
    r"""Full (i.e. using entire data matrix) EDMD method.

    The Koopman matrix is approximated

    .. math::
        K X &= X' \\
        K &= X' X^{\dagger},

    where :math:`\dagger` defines the Moore–Penrose inverse.

    """

    def __init__(self, is_diagonalize: bool = False):
        super(DMDFull, self).__init__()
        self.is_diagonalize = is_diagonalize

    def fit(self, X_ts: TSCDataFrame, **fit_params):
        super(DMDFull, self).fit(X_ts, **fit_params)

        self.koopman_matrix_ = self._compute_koopman_matrix(X_ts)
        self._compute_right_eigenpairs()

        if self.is_diagonalize:
            self._diagonalize_left_eigenvectors()

        return self

    def _compute_right_eigenpairs(self):
        self.eigenvalues_, self.eigenvectors_right_ = sort_eigenpairs(
            *np.linalg.eig(self.koopman_matrix_)
        )

    def _diagonalize_left_eigenvectors(self):
        """Compute right eigenvectors (not normed) such that
        Koopman matrix = right_eigenvectors @ diag(eigenvalues) @ left_eigenvectors .
        """

        # lhs_matrix = (np.diag(self.eigenvalues_) @ self.eigenvectors_right_)
        lhs_matrix = self.eigenvectors_right_ * self.eigenvalues_

        # NOTE: the left eigenvectors are not normed (i.e. ||ev|| != 1
        self.eigenvectors_left_ = np.linalg.solve(lhs_matrix, self.koopman_matrix_)

    def _compute_koopman_matrix(self, X: TSCDataFrame):

        # It is more suitable to get the shift_start and end in row orientation as this
        # is closer to the normal least squares parameter definition
        shift_start_transposed, shift_end_transposed = X.tsc.shift_matrices(
            snapshot_orientation="row"
        )

        # The easier to read version is:
        # koopman_matrix shift_start_transposed = shift_end_transposed
        # koopman_matrix.T = np.linalg.lstsq(shift_start_transposed,
        # shift_end_transposed, rcond=1E-14)[0]
        #
        # However, it is much more efficient to multiply shift_start from right
        # K^T (shift_start^T * shift_start) = (shift_end^T * shift_start)
        # K^T G = G'
        # This is because (shift_start^T * shift_start) is a smaller matrix and faster
        # to solve. For further info, see Williams et al. Extended DMD and DMD book,
        # Kutz et al. (book page 168).

        if shift_start_transposed.shape[1] > shift_start_transposed.shape[0]:

            warnings.warn(
                "There are more observables than snapshots. The current implementation "
                "favors more snapshots than obserables. This may result in a bad "
                "computational performance."
            )

        G = shift_start_transposed.T @ shift_start_transposed
        G_dash = shift_start_transposed.T @ shift_end_transposed

        # If a is square and of full rank, then x (but for round-off error) is the
        # “exact” solution of the equation.
        koopman_matrix, residual, rank, _ = np.linalg.lstsq(G, G_dash, rcond=1e-14)

        if rank != G.shape[1]:
            warnings.warn(
                f"Shift matrix (shape={G.shape}) has not full rank (={rank}), falling "
                f"back to least squares solution. The sum of residuals is: "
                f"{np.sum(residual)}"
            )

        # # TODO: Experimental (test other solvers, with more functionality)
        # #  ridge_regression, and sparisty promoting least squares solutions could be
        #    included here
        # # TODO: clarify if the ridge regression should be done better on lstsq with
        #     shift matrices (instead of the G, G_dash)

        # #  also possible to integrate "RidgeCV" which allows to select the best
        # #  alpha from a list
        # #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV

        # TODO: fit_intercept option useful to integrate?
        # koopman_matrix = ridge_regression(
        #     G, G_dash, alpha=self.alpha, verbose=0, return_intercept=False
        # )
        # koopman_matrix = koopman_matrix.T
        # # TODO: END Experimental

        # koopman_matrix = (
        #     LinearRegression(fit_intercept=False, normalize=False).fit(G, G_dash).coef_
        # )

        # The reason why it is transposed:
        # K * G_k = G_{k+1}
        # (G_k)^T * K = G_{k+1}^T  (therefore the row snapshot orientation at the
        #                           beginning)

        koopman_matrix = koopman_matrix.T
        return koopman_matrix


class DMDEco(DMDBase):
    r"""Approximates the Koopman matrix economically (compared to EDMDFull). It computes
    the principal components of the input data and computes the Koopman matrix there.
    Finally, it reconstructs the eigenpairs of the original system.

    1. Compute the singular value decomposition of the data and use the leading `k`
    singular values and corresponding vectors in :math:`U` and :math:`V`.

    .. math::
        X \approx U \Sigma V^*

    2. Compute the Koopman matrix on the PCA coordinates:

    .. math::
        K = U^T X' V \Sigma^{-1}

    3. Compute the eigenpairs (stated in matrix form):

    .. math::
        K W_r = W_r \Omega

    4. Reconstruct the (exact) eigendecomposition of :math:`K`

    .. math::
        \Psi_r = X' V \Sigma^{-1} W

    .. note::
        The eigenvectors in step 4 can also be computed with :math:`\Psi_r = U W`, which
        is then referred to the projected reconstruction.

    .. todo::
        cite Tu, 2014
    """

    def __init__(self, svd_rank=10):
        self.k = svd_rank
        super(DMDEco, self).__init__()

    def fit(self, X_ts, y=None, **fit_params):
        super(DMDEco, self).fit(X_ts, **fit_params)

        self._compute_internals(X_ts)
        return self

    def _compute_internals(self, X: TSCDataFrame):
        # TODO: different orientations are good for different cases:
        #  1 more snapshots than quantities
        #  2 more quantities than snapshots
        #  Currently it is optimized for the case 2.

        shift_start, shift_end = X.tsc.shift_matrices(snapshot_orientation="column")
        U, S, Vh = np.linalg.svd(shift_start, full_matrices=False)  # (1.18)

        U = U[:, : self.k]
        S = S[: self.k]
        S_inverse = np.reciprocal(S, out=S)

        V = Vh.T
        V = V[:, : self.k]

        koopman_matrix_low_rank = (
            U.T @ shift_end @ mat_dot_diagmat(V, S_inverse)
        )  # (1.20)

        self.eigenvalues_, eigenvector = np.linalg.eig(
            koopman_matrix_low_rank
        )  # (1.22)

        # As noted in the resource, there is also an alternative way
        # self.eigenvectors = U @ W

        self.eigenvectors_right_ = (
            shift_end @ V @ diagmat_dot_mat(S_inverse, eigenvector)
        )  # (1.23)

        return koopman_matrix_low_rank


class PyDMDWrapper(DMDBase):
    def __init__(
        self, method: str, svd_rank, tlsq_rank, exact, opt, **init_params,
    ):
        super(PyDMDWrapper, self).__init__()

        self.method_ = method.lower()

        standard_params = {
            "svd_rank": svd_rank,
            "tlsq_rank": tlsq_rank,
            "exact": exact,
            "opt": opt,
        }

        if method == "dmd":
            self.dmd_ = pydmd.DMD(**standard_params)
        elif method == "hodmd":
            standard_params["d"] = init_params.pop("d", 1)
            self.dmd_ = pydmd.HODMD(**standard_params)

        elif method == "fbdmd":
            self.dmd_ = pydmd.FbDMD(**standard_params)
        elif method == "mrdmd":
            standard_params["max_cycles"] = init_params.pop("max_cycles", 1)
            standard_params["max_level"] = init_params.pop("max_level", 6)
            self.dmd_ = pydmd.MrDMD(**standard_params)
        elif method == "cdmd":
            standard_params["compression_matrix"] = init_params.pop(
                "max_level", "uniform"
            )
            self.dmd_ = pydmd.CDMD(**standard_params)
        elif method == "optdmd":
            standard_params["factorization"] = init_params.pop("factorization", "evd")
            self.dmd_ = pydmd.OptDMD(**standard_params)
        elif method == "dmdc":
            # self.dmd_ = pydmd.DMDc(**init_params)
            raise NotImplementedError(
                "Currently not implemented as it requires " "additional input"
            )
        else:
            raise ValueError(f"method={method} not known")

    def fit(self, X_ts: TSCDataFrame, **fit_params) -> "PyDMDWrapper":

        super(PyDMDWrapper, self).fit(X_ts=X_ts)

        if len(X_ts.ids) > 1:
            raise NotImplementedError(
                "Provided DMD methods only allow single time " "series analysis."
            )

        # data is column major
        self.dmd_.fit(X=X_ts.to_numpy().T)
        self.eigenvectors_right_ = self.dmd_.modes
        self.eigenvalues_ = self.dmd_.eigs

        return self


class PCMKoopman(object):
    """
    Koopman operator on the point cloud manifold.
    """

    def __init__(self, pcm, rcond=1e-10, verbosity_level=0):
        """
        pcm:    PCManifold object
        rcond:  condition number used as minimum tolerance. default: 1e-10
        verbosity_level: 0 (silent), 1 (some messages)
        """

        # experimental code -- de-deprecate if required (refactor to above base class!)
        import warnings

        warnings.warn("Experimental code. Use with caution.")

        self._pcm = pcm
        self._rcond = rcond
        self._verbosity_level = verbosity_level

    def build(self, x_data, y_data, regularizer_strength=1e-6):
        """
        Constructs the Koopman operator matrix from snapshot data (x_data, y_data).
        """

        kernel_base = self._pcm.compute_kernel_matrix() + regularizer_strength ** 2 * scipy.sparse.identity(
            self._pcm.shape[0]
        )
        kernel_base.eliminate_zeros()  # has to be sparse matrix

        invdiag = scipy.sparse.diags(
            1.0 / (self._rcond + kernel_base.sum(axis=1).A.ravel())
        )
        kernel_base = invdiag @ kernel_base

        phi0 = self._pcm.compute_kernel_matrix(x_data).T
        phi1 = self._pcm.compute_kernel_matrix(y_data).T

        # more efficient format for spsolve
        phi0 = scipy.sparse.csc_matrix(phi0)
        phi1 = scipy.sparse.csc_matrix(phi1)

        atol = regularizer_strength
        btol = regularizer_strength

        # _K = scipy.sparse.linalg.spsolve(phi0, phi1)

        _K = np.zeros((phi0.shape[1], phi0.shape[1]))
        if self._verbosity_level > 1:
            print("EDMD: computing K, shape is", _K.shape)

        for k in range(_K.shape[1]):
            b = np.array(phi1[:, k].todense()).reshape(-1,)

            _K[:, k] = scipy.sparse.linalg.lsmr(phi0, b, atol=1e-15, btol=1e-15)[0]

        if self._verbosity_level > 1:
            print("EDMD: done.")

        # compute the "modes" given the dictionary
        # NOTE: this is different from standard EDMD!! There, one would compute
        # the modes given the eigenfunctions, not the dictionary!

        _C = np.zeros((kernel_base.shape[1], self._pcm.shape[1]))

        for k in range(_C.shape[1]):
            b = self._pcm[:, k].reshape(-1,)
            _C[:, k] = scipy.sparse.linalg.lsmr(kernel_base, b, atol=atol, btol=btol)[0]

        self._koopman = _K
        self._C = _C

    def _get_observables(self, x):
        # project into observable space

        phinew = self._pcm.compute_kernel_matrix(Y=x)
        invdiag = scipy.sparse.diags(1.0 / (self._rcond + phinew.sum(axis=0).A.ravel()))
        phinew = phinew @ invdiag
        return scipy.sparse.csr_matrix(phinew.T)

    @property
    def K(self):
        """ The Koopman operator matrix. """
        return self._koopman

    def predict(self, xold, NT=1):
        """ Predicts new points given old points. """
        phinew = self._get_observables(xold)

        phinew_dt = np.array(phinew.todense())
        result_dt = []

        # predict one step
        for it in range(NT):
            result_dt.append(phinew_dt @ self._C)
            phinew_dt = phinew_dt @ self._koopman

        # project back
        return result_dt


class DMDPowerAnalysis:
    @staticmethod
    def plot_dmd_power_spectrum(eigvals, dt, initial_state):

        import matplotlib.pyplot as plt

        freq = (np.log(eigvals.astype(np.complex)) / dt) / (2 * np.pi)

        power = np.abs(initial_state.ravel()) * 2 / np.sqrt(len(eigvals))
        plt.scatter(np.abs(np.imag(freq)), power)
        plt.show()
