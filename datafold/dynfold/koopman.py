import numpy as np
import scipy.sparse
from sklearn.base import TransformerMixin

import pcmanifold.timeseries as ts

#  TODO: make EDMDEco (computationally economic version using the the SVD, see DMD book)
#      a reminder to substract (optionally) the mean of the data to center it
#      go to pdfp. 21 and take the equation 1.19 - 1.23


class EDMDBase(TransformerMixin):

    def __init__(self):

        self.koopman_matrix_ = None
        self.eigenvalues_ = None
        self.eigenvectors_left_ = None
        self.eigenvectors_right_ = None

    def fit(self, X, y=None, **fit_params):
        is_diagonalize = fit_params.pop("diagonalize", False)
        self.koopman_matrix_ = self._compute_koopman_matrix(X)

        if is_diagonalize:
            self.eigenvalues_, self.eigenvectors_left_, self.eigenvectors_right_ = self._diagonalize_koopman_matrix()
        return self

    def fit_transform(self, X, y=None, **fit_params):
        is_diagonalize = fit_params.pop("diagonalize", False)
        self.fit(X, y, **fit_params)

        if is_diagonalize:
            return self.eigenvalues_, self.eigenvectors_left_, self.eigenvectors_right_
        else:
            return self.koopman_matrix_

    def _compute_koopman_matrix(self, X: ts.TSCDataFrame):
        raise NotImplementedError("Base class")

    def _diagonalize_koopman_matrix(self):
        """Approximate eigenvalues, -functions and modes of Koopman operator, given the data snapshots."""
        evals, evec_right = np.linalg.eig(self.koopman_matrix_)

        # TODO: instead of solving for the left eigenvectors, check if SVD obtains the left and right eigenvectors
        #  immediately, however, singular values have to be translated into eigenvalues.
        # TODO: speed up eigenvectors * diag
        evec_left = np.linalg.solve(evec_right @ np.diag(evals), self.koopman_matrix_)

        return evals, evec_left, evec_right


class EDMDExact(EDMDBase):

    def __init__(self):
        super(EDMDExact, self).__init__()

    def _compute_koopman_matrix(self, X):
        shift_start, shift_end = X.tsc.shift_matrices(snapshot_orientation="row")
        return np.linalg.lstsq(shift_start, shift_end, rcond=1E-14)[0]  # TODO: check the residual


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
        kernel_base.eliminate_zeros()  # has o be sparse matrix

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
