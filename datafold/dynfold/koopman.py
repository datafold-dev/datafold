import warnings
from typing import Optional, Union

import numpy as np
import scipy.linalg
import scipy.sparse
from sklearn.base import TransformerMixin

from datafold.pcfold.timeseries import TSCDataFrame, allocate_time_series_tensor
from datafold.utils.math import diagmat_dot_mat, mat_dot_diagmat, sort_eigenpairs


class EDMDBase(TransformerMixin):
    r"""Extended Dynamic Mode Decomposition (EDMD) approximates the Koopman operator with a matrix :math:`K`.

    The Koopman matrix :math:`K` defines a linear dynamical system of the form

    .. math::
        K x_k = x_{k+1} \\
        K^k x_0 = x_k

    where :math:`x_k` is the (column) state vector of the system at time :math:`k`. All subclasses should provide the
    right eigenvectors :math:`\Psi_r` and corresponding eigenvalues :math:`\omega` of the Koopman matrix to
    efficiently evolve the linear system TODO: link to method

    Evolving the linear system over many time steps is expensive due to the matrix power:

    .. math::
        K^k x_0 &= x_k

    Therefore, the (right) eigenpairs (:math:`\omega, \Psi_r`) of the Koopman matrix is computed and the initial
    condition is written in terms of the eigenvectors in a least-squares sense.

    .. math::
        K^k \Psi_r &= \Psi_r \Omega \\
        \Psi_r b &= x_0 \\
        K^k \Psi_r b &= x_k \\
        \Psi_r \Omega^k b &= x_k

    where the eigenproblem is stated in matrix form for all computed eigenpairs. Because
    :math:`\Omega` is a a diagonal matrix the power is very cheap compared to the generally full matrix :math:`K^k`.
    """

    def __init__(self):
        self.koopman_matrix_ = None
        self.eigenvalues_ = None
        self.eigenvectors_right_ = None

    def _set_X_info(self, X):

        if not X.is_const_dt():
            raise ValueError("Only data with const. frequency is supported.")

        self.dt_ = X.dt
        self._qoi_columns = X.columns

        self._time_interval = X.time_interval()
        self._normalize_shift = self._time_interval[0]

        assert (
            np.around(
                (self._time_interval[1] - self._normalize_shift) / self.dt_, decimals=14
            )
            % 1
            == 0
        )
        self._max_normtime = int(
            (self._time_interval[1] - self._normalize_shift) / self.dt_
        )

    def fit(self, X, y, **fit_params):
        self._set_X_info(X)


class EDMDFull(EDMDBase):
    r"""Full (i.e. using entire data matrix) EDMD method.

    The Koopman matrix is approximated

    .. math::
        K X &= X' \\
        K &= X' X^{\dagger},

    where :math:`\dagger` defines the Moore–Penrose inverse.

    """

    def __init__(self, is_diagonalize=False):
        super(EDMDFull, self).__init__()
        self.is_diagonalize = is_diagonalize

    def fit(self, X, y=None, **fit_params):
        super(EDMDFull, self).fit(X, y, **fit_params)

        self.koopman_matrix_ = self._compute_koopman_matrix(X)
        self._compute_right_eigenpairs()

        if self.is_diagonalize:
            self._diagonalize_left_eigenvectors()

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.koopman_matrix_

    def _compute_right_eigenpairs(self):
        self.eigenvalues_, self.eigenvectors_right_ = sort_eigenpairs(
            *np.linalg.eig(self.koopman_matrix_)
        )

    def _diagonalize_left_eigenvectors(self):
        """Compute right eigenvectors (not normed) such that
             Koopman matrix = right_eigenvectors @ diag(eigenvalues) @ left_eigenvectors ."""

        # lhs_matrix = (np.diag(self.eigenvalues_) @ self.eigenvectors_right_)
        lhs_matrix = self.eigenvectors_right_ * self.eigenvalues_
        self.eigenvectors_left_ = np.linalg.solve(lhs_matrix, self.koopman_matrix_)

    def _compute_koopman_matrix(self, X):

        # It is more suitable to get the shift_start and end in row orientation as this is closer to the
        # normal least squares parameter definition
        shift_start_transposed, shift_end_transposed = X.tsc.shift_matrices(
            snapshot_orientation="row"
        )

        # The easier to read version is:
        # koopman_matrix shift_start_transposed = shift_end_transposed
        # koopman_matrix.T = np.linalg.lstsq(shift_start_transposed, shift_end_transposed, rcond=1E-14)[0]
        #
        # However, it is much more efficient to multiply shift_start from right
        # K^T (shift_start^T * shift_start) = (shift_end^T * shift_start)
        # K^T G = G'
        # This is because (shift_start^T * shift_start) is a smaller matrix and faster to solve.
        # For further info, see Williams et al. Extended DMD and DMD book, Kutz et al. (book page 168).

        if shift_start_transposed.shape[1] > shift_start_transposed.shape[0]:

            warnings.warn(
                "There are more observables than snapshots. The current implementation favors more snapshots"
                "than obserables. This may result in a bad computational performance."
            )

        G = shift_start_transposed.T @ shift_start_transposed
        G_dash = shift_start_transposed.T @ shift_end_transposed

        # TODO: check the residual
        koopman_matrix = np.linalg.lstsq(G, G_dash, rcond=1e-14)[0]

        # The reason why it is tranposed:
        # K * G_k = G_{k+1}
        # (G_k)^T * K = G_{k+1}^T  (therefore the row snapshot orientation at the beginning)
        koopman_matrix = koopman_matrix.T
        return koopman_matrix


class EDMDEco(EDMDBase):
    r"""Approximates the Koopman matrix economically (compared to EDMDFull). It computes the principal components of
    the input data and computes the Koopman matrix there. Finally, it reconstructs the eigenpairs of the original
    system.

    1. Compute the singular value decomposition of the data and use the leading `k` singular values and corresponding
       vectors in :math:`U` and :math:`V`.

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
        The eigenvectors in step 4 can also be computed with :math:`\Psi_r = U W`, which is then referred to the
        projected reconstruction.

    .. todo::
        cite Tu, 2014
    """

    def __init__(self, k=10):
        self.k = k
        super(EDMDEco, self).__init__()

    def fit(self, X, y=None, **fit_params):
        super(EDMDEco, self).fit(X, y, **fit_params)

        self._compute_internals(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.koopman_matrix_

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


def evolve_linear_system(
    ic: np.ndarray,
    time_samples: np.ndarray,
    edmd: EDMDBase,
    dynmatrix: Optional[None] = None,
    time_invariant: bool = True,
    qoi_columns: Optional[Union[list, np.ndarray]] = None,
) -> TSCDataFrame:
    """
    Parameters
    ----------
    ic
        Initial conditions corresponding to :math:`x_0`
    time_samples
        Array of times where the dynamical system should be evaluated.
    edmd
        A EDMD object that was fit with data.
    dynmatrix
        Matrix for the linear system. If `None`, attributes from EDMD are used.
        .. note:: In the `dynmatrix` the (linear) backtransformation can be contained.
    time_invariant
        If `True` the time at the initial condition is zero; ff `False` the time starts corresponds to the time_samples.
    qoi_columns
        List of quantity of interest names that are set in the TSCDataFrame returned.

    Returns
    -------
    time_series : TSCDataFrame
        The resulting time series for each initial condition collected in a DataFrame

    """

    if dynmatrix is None:
        dynmatrix = edmd.eigenvectors_right_
        assert (
            dynmatrix is not None
        )  # TODO: make a is fit request here to edmd to guarantee that EDMD was fit!

        if qoi_columns is None:
            qoi_columns = edmd._qoi_columns

    nr_qoi = dynmatrix.shape[0]

    if len(qoi_columns) != nr_qoi:
        raise ValueError(f"len(qoi_columns)={qoi_columns} != nr_qoi={nr_qoi}")

    if qoi_columns is None:
        qoi_columns = np.arange(nr_qoi)

    if ic.shape[0] != nr_qoi:
        raise ValueError(
            f"Mismatch in ic.shape[0]={ic.shape[0]} is not dynmatrix.shape[0]={dynmatrix.shape[0]}."
        )

    # Choose alternative of how to evolve the linear system:
    if hasattr(edmd, "eigenvectors_left_") and (
        edmd.eigenvectors_left_ is not None and edmd.eigenvectors_right_ is not None
    ):
        # uses both eigenvectors (left and right). Used if is_diagonalize=True in EDMDFull
        ic = edmd.eigenvectors_left_ @ ic
    elif hasattr(edmd, "eigenvectors_right_") and edmd.eigenvectors_right_ is not None:
        # represent the initial condition in terms of right eigenvectors (by solving a least-squares problem)
        # -- only the right eigenvectors are required
        ic = np.linalg.lstsq(edmd.eigenvectors_right_, ic, rcond=1e-15)[0]
    else:
        # TODO: make this closer to scikit-learn (a "is_fit" function), probably best to do this at top of function
        raise RuntimeError("EDMD is not properly fit.")

    norm_time_samples = (
        time_samples - edmd._normalize_shift
    )  # time samples are normalized relative to EDMD shift

    if (norm_time_samples < 0).any():
        raise ValueError("Normalized time cannot be negative!")

    if time_invariant:
        # if the initial conditions start it is effectivly t=0
        norm_time_samples = norm_time_samples - norm_time_samples.min()

    if ic.ndim == 1:
        ic = ic[:, np.newaxis]

    omegas = np.log(edmd.eigenvalues_.astype(np.complex)) / edmd.dt_

    time_series_tensor = allocate_time_series_tensor(
        nr_time_series=ic.shape[1], nr_timesteps=time_samples.shape[0], nr_qoi=nr_qoi
    )

    for j, time in enumerate(norm_time_samples):
        time_series_tensor[:, j, :] = np.real(
            dynmatrix @ diagmat_dot_mat(np.exp(omegas * time), ic)
        ).T

    return TSCDataFrame.from_tensor(
        time_series_tensor, columns=qoi_columns, time_index=time_samples
    )
