from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from datafold.dynfold.base import TransformType, TSCTransformerMixin
from datafold.pcfold import PCManifold
from datafold.pcfold.kernels import GaussianKernel, PCManifoldKernel
from datafold.utils.general import mat_dot_diagmat


def get_ending_points(Xs: List[TransformType]):
    ending_point = 0
    ending_points = []
    for X_n in Xs:
        if isinstance(X_n, DataFrame):
            ending_point += np.array(X_n).shape[1]
        else:
            ending_point += X_n.shape[1]
        ending_points.append(ending_point)

    return ending_points


def normalize_csr_matrix(sparse_kernel_matrix: scipy.sparse.csr_matrix):
    sparse_kernel_matrix = (
        1 / 2 * scipy.sparse.csr_matrix(sparse_kernel_matrix + sparse_kernel_matrix.T)
    )
    sparse_kernel_matrix = (
        scipy.sparse.diags(
            np.sqrt(np.array(1 / sparse_kernel_matrix.sum(axis=0))).ravel(), 0
        )
        @ sparse_kernel_matrix
        @ scipy.sparse.diags(
            np.sqrt(np.array(1 / sparse_kernel_matrix.sum(axis=0))).ravel(), 0
        )
    )
    sparse_kernel_matrix = (
        scipy.sparse.diags(np.array(1 / sparse_kernel_matrix.sum(axis=0)).ravel(), 0)
        @ sparse_kernel_matrix
    )
    sparse_kernel_matrix.eliminate_zeros()
    return sparse_kernel_matrix


def sort_eigensystem(eigenvalues, eigenvectors):
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    return sorted_eigenvalues, sorted_eigenvectors


class JointlySmoothFunctions(TSCTransformerMixin, BaseEstimator):
    """Calculate smooth functions on multimodal data/observations.

    Parameters
    ----------
    n_kernel_eigenvectors: int
        The number of eigenvectors to compute from the kernel matrices.

    n_jointly_smooth_functions: int
        The number of jointly smooth functions to compute from the eigenvectors of the
        kernel matrices.

    kernel: Optional[Union[PCManifoldKernel, List[PCManifoldKernel]]]
        The kernel(s) used to describe the proximity between points. You can specify one
        kernel for all observations or one kernel for each observation. Defaults to the
        default :py:class: `.GaussianKernel`.

    kernel_eigenvalue_cut_off: float
        The kernel eigenvectors with a eigenvalue smaller than or equal to
        ``kernel_eigenvalue_cut_off`` will not be included in the calculation of the
        jointly smooth functions.

    eigenvector_tolerance: float
        The relative accuracy for eigenvalues, i.e. the stopping criterion. A value of
        0 implies machine precision.

    Attributes
    ----------
    ending_points_: List[int]
        The ending point of each observation. This is needed, as :py:meth`.fit`,
        :py:meth`.transform`, and :py:meth`.fit_transform` accept a single data array.
        Thus, the multimodal data is passed in as a single array and separated inside the
        methods. :py:meth`get_ending_points` of this module provide a convenience method
        to obtain the ending points of a list of observations.

    observations_: List[PCManifold]
        The :py:class:`PCManifolds` containing the separated observations with the
        specified, corresponding :py:class:`PCManifoldKernel`.

    kernel_matrices_: List[scipy.spars.csr_matrix]
        The computed kernel matrices.

    _cdist_kwargs_: List[Dict]
        The cdist_kwargs returned during the kernel calculation. This is required for the
        out-of-sample extension.

    kernel_eigenvectors_: List[scipy.sparse.csr_matrix]
        The kernel eigenvectors used to calculate the jointly smooth functions.

    kernel_eigenvalues_ List[scipy.sparse.csr_matrix]
        The kernel eigenvalues used to calculate the out-of-sample extension.

    _jointly_smooth_functions_: np.ndarray
        The calculated jointly smooth functions of shape
        `(n_samples, n_jointly_smooth_functions)`.

    _eigenvalues_: np.ndarray
        The eigenvalues of the jointly smooth functions of shape `(n_samples)`

    References
    ----------
    :cite:`TODO enter paper reference`
    """

    def __init__(
        self,
        n_kernel_eigenvectors: int = 100,
        n_jointly_smooth_functions: int = 10,
        kernel: Optional[Union[PCManifoldKernel, List[PCManifoldKernel]]] = None,
        kernel_eigenvalue_cut_off: float = 0,
        eigenvector_tolerance: float = 1e-6,
        **dist_kwargs,
    ) -> None:
        self.n_kernel_eigenvectors = n_kernel_eigenvectors
        self.n_jointly_smooth_functions = n_jointly_smooth_functions
        self.kernel = kernel
        self.kernel_eigenvalue_cut_off = kernel_eigenvalue_cut_off
        self.eigenvector_tolerance = eigenvector_tolerance
        self.dist_kwargs = dist_kwargs  # TODO List

        self.ending_points_: List[int] = []
        self.observations_: List[PCManifold] = []
        self.kernel_matrices_: List[scipy.sparse.csr_matrix] = []
        self._cdist_kwargs_: List[Dict] = []
        self.kernel_eigenvectors_: List[scipy.sparse.csr_matrix] = []
        self.kernel_eigenvalues_: List[scipy.sparse.csr_matrix] = []
        self._jointly_smooth_functions_: np.ndarray
        self._eigenvalues_: np.ndarray

    @property
    def jointly_smooth_functions(self) -> np.ndarray:
        return self._jointly_smooth_functions_

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues_

    def _setup_kernels_for_observations(self, observations):
        self.observations_ = []

        if self.kernel is None:
            self.observations_ = [
                PCManifold(observation, dist_kwargs=self.dist_kwargs)
                for observation in observations
            ]
        elif isinstance(self.kernel, PCManifoldKernel):
            self.observations_ = [
                PCManifold(
                    observation, kernel=self.kernel, dist_kwargs=self.dist_kwargs
                )
                for observation in observations
            ]
        elif isinstance(self.kernel, List):
            if len(self.kernel) == len(observations):
                self.observations_ = [
                    PCManifold(observation, kernel=kernel, dist_kwargs=self.dist_kwargs)
                    for observation, kernel in zip(observations, self.kernel)
                ]
            else:
                raise ValueError(
                    "Kernel list must have the same length as observations list"
                )

        self._optimize_kernels()

    def _optimize_kernels(self):
        for pcm in self.observations_:
            if isinstance(pcm.kernel, GaussianKernel):
                pcm.optimize_parameters(inplace=True)  # TODO Add result_scaling

    def _separate_X(self, X: TransformType) -> List[TransformType]:
        X_separated = [X[:, : self.ending_points_[0]]]
        for i in range(1, len(self.ending_points_)):
            X_separated.append(
                X[:, self.ending_points_[i - 1] : self.ending_points_[i]]
            )
        return X_separated

    def _calculate_kernel_matrices(self):
        self._cdist_kwargs_ = []
        self.kernel_matrices_ = []
        for observation in self.observations_:
            kernel_output = observation.compute_kernel_matrix()
            kernel_matrix, cdist_kwargs, _ = PCManifoldKernel.read_kernel_output(
                kernel_output
            )
            self._cdist_kwargs_.append(cdist_kwargs)
            sparse_kernel_matrix = scipy.sparse.csr_matrix(
                kernel_matrix, dtype=np.float64
            )
            # sparse_kernel_matrix = normalize_csr_matrix(sparse_kernel_matrix)
            self.kernel_matrices_.append(sparse_kernel_matrix)

    def _calculate_kernel_eigensystem(self):
        self.kernel_eigenvectors_ = []
        self.kernel_eigenvalues_ = []
        for kernel_matrix in self.kernel_matrices_:
            kernel_eigenvalues, kernel_eigenvectors = scipy.sparse.linalg.eigsh(
                kernel_matrix,
                k=self.n_kernel_eigenvectors,
                tol=self.eigenvector_tolerance,
                which="LM",
            )
            kernel_eigenvalues, kernel_eigenvectors = sort_eigensystem(
                kernel_eigenvalues, kernel_eigenvectors
            )
            kernel_eigenvectors = kernel_eigenvectors[
                :, kernel_eigenvalues > self.kernel_eigenvalue_cut_off
            ]
            kernel_eigenvalues = kernel_eigenvalues[
                kernel_eigenvalues > self.kernel_eigenvalue_cut_off
            ]
            self.kernel_eigenvectors_.append(kernel_eigenvectors)
            self.kernel_eigenvalues_.append(kernel_eigenvalues)

    def _calculate_jointly_smooth_functions(self) -> Tuple[np.ndarray, np.ndarray]:
        eigenvectors_matrix = scipy.sparse.csr_matrix(
            np.column_stack([eigenvector for eigenvector in self.kernel_eigenvectors_])
        )
        if len(self.kernel_eigenvectors_) == 2:
            ev0 = self.kernel_eigenvectors_[0]
            ev1 = self.kernel_eigenvectors_[1]
            n_jointly_smooth_functions = min(
                [self.n_jointly_smooth_functions, ev0.shape[1] - 1, ev1.shape[1] - 1]
            )
            Q, eigenvalues, R_t = scipy.sparse.linalg.svds(
                ev0.T @ ev1,
                k=n_jointly_smooth_functions,
                which="LM",
                tol=self.eigenvector_tolerance,
            )
            center = np.row_stack(
                [np.column_stack([Q, Q]), np.column_stack([R_t.T, -R_t.T])]
            )
            right = np.diag(
                np.power(np.concatenate([1 + eigenvalues, 1 - eigenvalues]), -1 / 2)
            )
            jointly_smooth_functions = (
                1 / np.sqrt(2) * eigenvectors_matrix @ center @ right
            )
        else:
            n_jointly_smooth_functions = min(
                [self.n_jointly_smooth_functions, eigenvectors_matrix.shape[1]]
            )
            jointly_smooth_functions, eigenvalues, _ = scipy.sparse.linalg.svds(
                eigenvectors_matrix,
                k=n_jointly_smooth_functions,
                which="LM",
                tol=self.eigenvector_tolerance,
            )

        eigenvalues, jointly_smooth_functions = sort_eigensystem(
            eigenvalues, jointly_smooth_functions
        )
        return jointly_smooth_functions, eigenvalues

    def nystrom(self, new_indexed_observations: Dict[int, TransformType]):
        """Embed out-of-sample points with Nyström.

        (see transform of dmap for Nyström documentation)

        Parameters
        ----------
        new_indexed_observations: Dict[int, List[Union[TSCDataFrame, pandas.DataFrame, numpy.ndarray]]
             A dict containing out-of-sample points for (not necessarily all) observations.
             The keys are the indexes of the observations. The values are the observations
             of shape `(n_samples, *n_features_of_observation*)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as the values of shape `(n_samples, n_jointly_smooth_functions)`.
        """
        eigenvectors = []
        alphas = []
        for index, new_observation in new_indexed_observations.items():
            kernel_eigenvector = self.kernel_eigenvectors_[index]
            alpha = kernel_eigenvector.T @ self._jointly_smooth_functions_
            alphas.append(alpha)
            observation = self.observations_[index]
            kernel_output = observation.compute_kernel_matrix(
                new_observation, **self._cdist_kwargs_[index]
            )
            kernel_matrix, _, _ = PCManifoldKernel.read_kernel_output(
                kernel_output=kernel_output
            )
            approx_eigenvectors = kernel_matrix @ mat_dot_diagmat(
                self.kernel_eigenvectors_[index],
                np.reciprocal(self.kernel_eigenvalues_[index]),
            )
            eigenvectors.append(approx_eigenvectors)
        f_m_star = 0.0
        for i in range(len(alphas)):
            f_m_star += eigenvectors[i] @ alphas[i]
        f_m_star /= len(alphas)
        return f_m_star

    def fit(self, X: TransformType, y=None, **fit_params) -> "JointlySmoothFunctions":
        """Compute the jointly smooth functions.

        Parameters
        ----------
        X: TSCDataFrame, pandas.Dataframe, numpy.ndarray
            Training data of shape `(n_samples, n_features)`

        y: None
            ignored

        **fit_params: Dict[str, object]
            - ending_points: ``List[int]``
                The ending points of the observations.

        Returns
        -------
        JointlySmoothFunctions
            self
        """
        X = self._validate_datafold_data(
            X=X,
            array_kwargs=dict(ensure_min_samples=max(2, self.n_kernel_eigenvectors)),
            tsc_kwargs=dict(ensure_min_samples=max(2, self.n_kernel_eigenvectors)),
        )

        self._setup_feature_attrs_fit(
            X=X,
            features_out=[f"jsf{i}" for i in range(self.n_jointly_smooth_functions)],
        )

        self.ending_points_ = self._read_fit_params(
            attrs=[
                ("ending_points", None),
            ],
            fit_params=fit_params,
        )

        if self.ending_points_ is None:
            raise ValueError("Please specify the ending_points of each observation")
        if self.ending_points_[-1] != X.shape[1]:
            raise ValueError("Final endpoint must be the same as X.shape[1]")

        observations = self._separate_X(X)

        self._setup_kernels_for_observations(observations)

        self._calculate_kernel_matrices()

        self._calculate_kernel_eigensystem()

        (
            self._jointly_smooth_functions_,
            self._eigenvalues_,
        ) = self._calculate_jointly_smooth_functions()

        return self

    def transform(self, X: TransformType) -> TransformType:
        """Embed out-of-sample points with the Nyström extension.

        (see transform of dmap for Nyström documentation)

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data points of shape `(n_samples, n_features)` to be embedded.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_jointly_smooth_functions)`
        """
        check_is_fitted(
            self,
            (
                "ending_points_",
                "observations_",
                "kernel_matrices_",
                "_cdist_kwargs_",
                "kernel_eigenvectors_",
                "kernel_eigenvalues_",
                "_jointly_smooth_functions_",
                "_eigenvalues_",
            ),
        )

        X = self._validate_datafold_data(
            X=X,
            array_kwargs=dict(ensure_min_samples=1),
            tsc_kwargs=dict(ensure_min_samples=1),
        )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must have the same number of features as the data with which fit was called."
                "If you want to call it with fewer observations, you have to call nystrom"
            )

        self._validate_feature_input(X, direction="transform")

        new_observations = self._separate_X(X)

        indices = list(range(len(self.observations_)))
        indexed_observations = dict(zip(indices, new_observations))
        f_m_star = self.nystrom(indexed_observations)
        return f_m_star

    def fit_transform(self, X: TransformType, y=None, **fit_params) -> TransformType:
        """Compute jointly smooth functions and return them.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Training data of shape `(n_samples, n_features)`

        y: None
            ignored

        **fit_params: Dict[str, object]
            See `fit` method for additional parameter.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_jointly_smooth_functions)`
        """
        X = self._validate_datafold_data(
            X,
            array_kwargs=dict(ensure_min_samples=max(2, self.n_kernel_eigenvectors)),
            tsc_kwargs=dict(ensure_min_samples=max(2, self.n_kernel_eigenvectors)),
        )
        self.fit(X=X, y=y, **fit_params)

        return self._jointly_smooth_functions_

    def score(self):
        """Compute a score for hyperparameter optimization.

        Returns
        -------
        float
            The sum of the truncated energies.
        """
        return self.calculate_truncated_energies().sum()

    def calculate_truncated_energies(self) -> np.ndarray:
        """Compute the truncated energy for each kernel eigenvector.

        Returns
        -------
        np.ndarray
            The truncated energies of shape `(n_observations, n_jointly_smooth_functions)`.
        """
        truncated_energies = []
        for kernel_eigenvector in self.kernel_eigenvectors_:
            truncated_energy = (
                np.linalg.norm(
                    kernel_eigenvector.T @ self.jointly_smooth_functions, axis=0
                )
                ** 2
            )
            truncated_energies.append(truncated_energy)
        return np.array(truncated_energies)

    def calculate_E0(self) -> float:
        """Compute a threshold for the eigenvalues of the jointly smooth functions.

        Returns
        -------
        float
            The E0 threshold value from :cite:`TODO enter paper reference`
        """
        noisy = self.kernel_eigenvectors_[-1].copy()
        np.random.shuffle(noisy)

        kernel_eigenvectors = self.kernel_eigenvectors_[:-1]
        kernel_eigenvectors.append(noisy)

        eigenvectors_matrix = scipy.sparse.csr_matrix(
            np.column_stack([eigenvector for eigenvector in kernel_eigenvectors])
        )

        if len(kernel_eigenvectors) == 2:
            ev0 = kernel_eigenvectors[0]
            ev1 = kernel_eigenvectors[1]
            _, Gamma, _ = scipy.sparse.linalg.svds(
                ev0.T @ ev1, k=self.n_jointly_smooth_functions, which="LM"
            )
        else:
            _, Gamma, _ = scipy.sparse.linalg.svds(
                eigenvectors_matrix, k=self.n_jointly_smooth_functions, which="LM"
            )

        Gamma.sort()
        gamma2 = Gamma[-2]
        E0 = (1 + gamma2) / 2
        return E0
