import numpy as np
import scipy.linalg
import scipy.sparse
from sklearn.base import TransformerMixin

import datafold.pcfold.timeseries as ts


class EDMDBase(TransformerMixin):
    """For all subclasses the Koopman matrix K is defined:

        K * Phi_k = Phi_{k+1}

    where Phi has snapshots row-wise. Note in the DMD book, the snapshots are column-wise!

    The column orientation for the Koopman operator is:

        Phi_k^T K^T = Phi_{k+1}^T

    i.e. K^T can be used, if the snapshots are column-wise.
    """

    def __init__(self, is_diagonalize):

        self.is_diagonalize = is_diagonalize

        self.koopman_matrix_ = None  # TODO: maybe no need to save the koopman matrix?
        self.eigenvalues_ = None
        self.eigenvectors_left_ = None
        self.eigenvectors_right_ = None

    def _set_X_info(self, X):

        if not X.is_const_dt():
            raise ValueError("Only data with const. frequency is supported.")

        self.dt_ = X.dt
        self._qoi_columns = X.columns

        self._time_interval = X.time_interval()
        self._normalize_shift = self._time_interval[0]
        assert (self._time_interval[1] - self._normalize_shift) / self.dt_ % 1 == 0
        self._max_normtime = int((self._time_interval[1] - self._normalize_shift) / self.dt_)

    def fit(self, X, y, **fit_params):
        self._set_X_info(X)

    def _compute_left_eigenpairs(self):
        self.eigenvalues_, self.eigenvectors_left_ = np.linalg.eig(self.koopman_matrix_.T)
        # transpose here, such that the eigenvectors are in rows (according to definition of left eigenvectors):
        self.eigenvectors_left_ = self.eigenvectors_left_.T

        # TODO: sorting of eigenpairs could be included in a helpers function (in utils)
        # Sort eigenvectors accordingly:
        idx = self.eigenvalues_.argsort()[::-1]
        self.eigenvalues_ = self.eigenvalues_[idx]
        self.eigenvectors_left_ = self.eigenvectors_left_[idx, :]

    def _diagonalize_right_eigenvectors(self):
        """Compute right eigenvectors (not normed) such that
             Koopman matrix = right_eigenvectors @ diag(eigenvalues) @ left_eigenvectors ."""
        #lhs_matrix = (np.diag(self.eigenvalues_) @ self.eigenvectors_left_).T
        lhs_matrix = (self.eigenvectors_left_ * self.eigenvalues_).T
        self.eigenvectors_right_ = np.linalg.solve(lhs_matrix, self.koopman_matrix_.T).T


class EDMDFull(EDMDBase):

    def __init__(self, is_diagonalize=False):
        super(EDMDFull, self).__init__(is_diagonalize)

    def fit(self, X, y=None, **fit_params):
        super(EDMDFull, self).fit(X, y, **fit_params)

        self.koopman_matrix_ = self._compute_koopman_matrix(X)
        self._compute_left_eigenpairs()

        if self.is_diagonalize:
            self._diagonalize_right_eigenvectors()

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.koopman_matrix_

    def _compute_koopman_matrix(self, X):
        shift_start, shift_end = X.tsc.shift_matrices(snapshot_orientation="row")

        # The easier to read version is:
        # np.linalg.lstsq(shift_start, shift_end, rcond=1E-14)[0]
        # K^T shift_start^T = shift_end^T (from the classical Koopman operator view on observables).
        # i.e. solve the Koopman with a least square problem and the two matrices shift_start and shift_end
        #
        # However, it is much more efficient to multiply shift_start from left
        # K^T (shift_start^T * shift_start) = (shift_end^T * shift_start)
        # K^T G = G'
        # This is because (shift_start^T * shift_start) is a smaller matrix and faster to solve.
        # For further info, see Williams et al. Extended DMD and DMD book, Kutz et al. (book page 168).

        if shift_start.shape[1] > shift_start.shape[0]:
            import warnings
            warnings.warn("There are more observables than snapshots. The current implementation favors more snapshots"
                          "than obserables. This may result in a bad computational performance.")

        G = shift_start.T @ shift_start
        G_d = (shift_start.T @ shift_end)
        return np.linalg.lstsq(G, G_d, rcond=1E-14)[0]  # TODO: check the residual


class EDMDEco(EDMDBase):

    def __init__(self, k=10, is_diagonalize=False):
        self.k = k
        super(EDMDEco, self).__init__(is_diagonalize=is_diagonalize)

    def fit(self, X, y=None, **fit_params):
        super(EDMDEco, self).fit(X, y, **fit_params)

        self._compute_internals(X)

        if self.is_diagonalize:
            self._diagonalize_right_eigenvectors()

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.koopman_matrix_

    def _compute_internals(self, X: ts.TSCDataFrame):
        # TODO: different orientations are good for different cases:
        #  1 more snapshots than quantities
        #  2 more quantities than snapshots
        #  Currently it is optimized for the case 2.

        shift_start, shift_end = X.tsc.shift_matrices(snapshot_orientation="column")
        U, S, Vh = np.linalg.svd(shift_start, full_matrices=False)  # (1.18)
        V = Vh.T

        U = U[:, :self.k]
        S = np.diag(S[:self.k])  # TODO: can be improved
        V = V[:, :self.k]

        koopman_matrix_low_rank = U.T @ shift_end @ V @ np.linalg.inv(S)  # (1.20)

        self.eigenvalues_, eigenvector = np.linalg.eig(koopman_matrix_low_rank)  # (1.22)

        # As noted in the resource, there is also an alternative way
        # self.eigenvectors = U @ W
        self.eigenvectors_left_ = shift_end @ V @ np.linalg.inv(S) @ eigenvector  # (1.23)
        self.eigenvectors_left_ = self.eigenvectors_left_.T

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
            self._pcm.shape[0])
        kernel_base.eliminate_zeros()  # has to be sparse matrix

        invdiag = scipy.sparse.diags(1.0 / (self._rcond + kernel_base.sum(axis=1).A.ravel()))
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
            print('EDMD: computing K, shape is', _K.shape)

        for k in range(_K.shape[1]):
            b = np.array(phi1[:, k].todense()).reshape(-1, )

            _K[:, k] = scipy.sparse.linalg.lsmr(phi0, b, atol=1E-15, btol=1E-15)[0]

        if self._verbosity_level > 1:
            print('EDMD: done.')

        # compute the "modes" given the dictionary
        # NOTE: this is different from standard EDMD!! There, one would compute
        # the modes given the eigenfunctions, not the dictionary!

        _C = np.zeros((kernel_base.shape[1], self._pcm.shape[1]))

        for k in range(_C.shape[1]):
            b = self._pcm[:, k].reshape(-1, )
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


def _create_time_series_tensor(nr_initial_condition, nr_timesteps, nr_qoi):
    # This indexing is for C-aligned arrays
    # index order for "tensor[depth, row, column]"
    #     1) depth = timeseries (i.e. for respective initial condition),
    #     2) row = time step [k],
    #     3) column = qoi
    return np.zeros([nr_initial_condition, nr_timesteps, nr_qoi])


def evolve_linear_system(ic, time_samples, edmd, dynmatrix=None, qoi_columns=None):

    if dynmatrix is None:
        # To set the dynmatrix allows for generalization
        dynmatrix = edmd.eigenvectors_left_

    nr_qoi = dynmatrix.shape[1]

    if qoi_columns is None and dynmatrix is None:
        qoi_columns = edmd._qoi_columns
    elif qoi_columns is None and dynmatrix is not None:
        qoi_columns = np.arange(nr_qoi)

    norm_time_samples = time_samples - edmd._normalize_shift  # process starts always at time=0

    assert len(qoi_columns) == nr_qoi

    if ic.ndim == 1:
        ic = ic[np.newaxis, :]

    omegas = np.log(edmd.eigenvalues_) / edmd.dt_  # divide by edmd.dt_
    time_series_tensor = _create_time_series_tensor(nr_initial_condition=ic.shape[0],
                                                    nr_timesteps=time_samples.shape[0],
                                                    nr_qoi=nr_qoi)

    # See: https://imgur.com/a/n4M7G2k  for equations -->TODO: explain properly in documentation!
    # # better readable code form, optimized below
    # for k, ic in enumerate(range(ic.shape[0])):
    #     for j, t in enumerate(eval_normtime):
    #         koopman_t = ic[k, :] @ np.diag(np.exp(omegas * t)) @ eigenvectors
    #         time_series_tensor[k, j, :] = np.real(koopman_t @ self.gh_coeff_)

    for j, time in enumerate(norm_time_samples):
        # rowwise elementwise multiplication instead of (with full diagonal matrix) n^3 -> n complexity
        #   ic @ np.diag(np.exp(omegas * t)) @ eigenvectors
        time_series_tensor[:, j, :] = np.real(ic * np.exp(omegas * time) @ dynmatrix)

    return ts.TSCDataFrame.from_tensor(time_series_tensor, columns=qoi_columns, time_index=time_samples)
