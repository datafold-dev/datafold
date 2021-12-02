from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from datafold.dynfold.base import TransformType, TSCTransformerMixin
from datafold.pcfold import PCManifold, TSCDataFrame
from datafold.pcfold.eigsolver import compute_kernel_eigenpairs
from datafold.pcfold.kernels import GaussianKernel, PCManifoldKernel
from datafold.utils.general import mat_dot_diagmat


# TODO: replace with datafold.utils.general.sort_eigenpairs
#  -- Note that one sorts abs. values, the other complex values directly
def sort_eigensystem(eigenvalues, eigenvectors):
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    if isinstance(eigenvectors, pd.DataFrame):
        sorted_eigenvectors = eigenvectors.iloc[:, idx]
    else:
        sorted_eigenvectors = eigenvectors[:, idx]
    return sorted_eigenvalues, sorted_eigenvectors


class JsfDataset:
    """`JsfDataset` does the slicing of multimodal data. This is needed, as `.fit`,
    `.transform`, and `.fit_transform` of `JointlySmoothFunctions` accept a single
    data array `X`. Thus, the multimodal data is passed in as a single array and is
    then separated inside the methods.

    Parameters
    ----------
    name
        The name of the dataset.

    columns
        The columns that correspond to the dataset.

    kernel
        The (optional) kernel for the dataset.

    result_scaling
        The (optional) result scaling for the parameter optimization.

    dist_kwargs
        Keyword arguments passed to the internal distance matrix computation. See
        :py:meth:`datafold.pcfold.distance.compute_distance_matrix` for parameter
        arguments.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        columns: Optional[slice] = None,
        kernel: Optional[PCManifoldKernel] = None,
        result_scaling: float = 1.0,
        **dist_kwargs,
    ):
        self.name = name
        self.columns = columns
        self.kernel = kernel
        self.result_scaling = result_scaling
        self.dist_kwargs = dist_kwargs

    def extract_from(self, X: TransformType) -> Union[TSCDataFrame, PCManifold]:
        if self.columns:
            if isinstance(X, pd.DataFrame) or isinstance(X, TSCDataFrame):
                data = X.iloc[:, self.columns]
            else:
                data = X[:, self.columns]
        else:
            data = X

        if isinstance(data, TSCDataFrame):
            if self.kernel is None:
                self.kernel = GaussianKernel()
            data = TSCDataFrame(data, kernel=self.kernel, dist_kwargs=self.dist_kwargs)
        elif isinstance(data, (np.ndarray, pd.DataFrame)):
            data = PCManifold(
                data=data, kernel=self.kernel, dist_kwargs=self.dist_kwargs
            )
            if self.kernel is None:
                data.optimize_parameters(
                    inplace=True, result_scaling=self.result_scaling
                )

        return data


class _ColumnSplitter:
    """Uses a `JsfDataset` list to split up a single data array X into a `PCManifold` list.

    Parameters
    ----------
    datasets
        The `JsfDataset`s used to split up the array X.
    """

    def __init__(self, datasets: Optional[List[JsfDataset]] = None):
        self.datasets = datasets

    def split(self, X: TransformType, y=None) -> List[Union[TSCDataFrame, PCManifold]]:
        if not self.datasets:
            dataset = JsfDataset()
            return [dataset.extract_from(X)]

        X_split: List[Union[TSCDataFrame, PCManifold]] = []

        for dataset in self.datasets:
            X_split.append(dataset.extract_from(X))

        return X_split


class JointlySmoothFunctions(TSCTransformerMixin, BaseEstimator):
    """Calculate smooth functions on multimodal data/observations.

    Parameters
    ----------
    datasets
        The :py:class:`JsfDataset`s used to split up the multimodal data.

    n_kernel_eigenvectors
        The number of eigenvectors to compute from the kernel matrices.

    n_jointly_smooth_functions
        The number of jointly smooth functions to compute from the eigenvectors of the
        kernel matrices.

    kernel_eigenvalue_cut_off
        The kernel eigenvectors with a eigenvalue smaller than or equal to
        ``kernel_eigenvalue_cut_off`` will not be included in the calculation of the
        jointly smooth functions.

    eigenvector_tolerance
        The relative accuracy for eigenvalues, i.e. the stopping criterion. A value of
        0 implies machine precision.

    Attributes
    ----------
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
        datasets: Optional[List[JsfDataset]] = None,
        n_kernel_eigenvectors: int = 100,
        n_jointly_smooth_functions: int = 10,
        kernel_eigenvalue_cut_off: float = 0,
        eigenvector_tolerance: float = 1e-6,
    ) -> None:
        self.n_kernel_eigenvectors = n_kernel_eigenvectors
        self.n_jointly_smooth_functions = n_jointly_smooth_functions
        self.datasets = datasets
        self.kernel_eigenvalue_cut_off = kernel_eigenvalue_cut_off
        self.eigenvector_tolerance = eigenvector_tolerance

        self.ending_points_: List[int]
        self.observations_: List[Union[TSCDataFrame, PCManifold]]
        self.kernel_matrices_: List[scipy.sparse.csr_matrix]
        self._cdist_kwargs_: List[Dict]
        self.kernel_eigenvectors_: List[scipy.sparse.csr_matrix]
        self.kernel_eigenvalues_: List[scipy.sparse.csr_matrix]
        self._jointly_smooth_functions_: np.ndarray
        self._eigenvalues_: np.ndarray

    @property
    def jointly_smooth_functions(self) -> TransformType:
        return self._jointly_smooth_functions_

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues_

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
            self.kernel_matrices_.append(sparse_kernel_matrix)

    def _calculate_kernel_eigensystem(self):
        self.kernel_eigenvectors_ = []
        self.kernel_eigenvalues_ = []
        for i, kernel_matrix in enumerate(self.kernel_matrices_):
            is_symmetric = np.alltrue(kernel_matrix.A == kernel_matrix.T.A)
            ones_row = np.ones(kernel_matrix.shape[0])
            ones_col = np.ones(kernel_matrix.shape[1])
            is_stochastic = np.alltrue(kernel_matrix @ ones_col == ones_row)
            kernel_eigenvalues, kernel_eigenvectors = compute_kernel_eigenpairs(
                kernel_matrix,
                n_eigenpairs=self.n_kernel_eigenvectors,
                is_symmetric=is_symmetric,
                is_stochastic=is_stochastic,
            )

            if isinstance(kernel_matrix, TSCDataFrame):
                index_from = kernel_matrix
            elif (
                isinstance(self.observations_[i], TSCDataFrame)
                and kernel_matrix.shape[0] == self.observations_[i].shape[0]
            ):
                index_from = self.observations_[i]
            else:
                index_from = None

            if index_from is not None:
                kernel_eigenvectors = TSCDataFrame.from_same_indices_as(
                    index_from,
                    kernel_eigenvectors,
                    except_columns=[
                        f"kev{i}" for i in range(self.n_kernel_eigenvectors)
                    ],
                )

            kernel_eigenvalues, kernel_eigenvectors = sort_eigensystem(
                kernel_eigenvalues, kernel_eigenvectors
            )
            if isinstance(kernel_eigenvectors, TSCDataFrame):
                kernel_eigenvectors = kernel_eigenvectors.iloc[
                    :, kernel_eigenvalues > self.kernel_eigenvalue_cut_off
                ]
            else:
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

        tsc_flag = isinstance(self.kernel_eigenvectors_[0], TSCDataFrame)
        if tsc_flag:
            index_from = self.kernel_eigenvectors_[0]
        else:
            index_from = None

        rng = np.random.default_rng(seed=1)
        if len(self.kernel_eigenvectors_) == 2:
            ev0 = self.kernel_eigenvectors_[0]
            ev1 = self.kernel_eigenvectors_[1]
            n_jointly_smooth_functions = min(
                [self.n_jointly_smooth_functions, ev0.shape[1] - 1, ev1.shape[1] - 1]
            )
            if tsc_flag:
                evs = ev0.to_numpy().T @ ev1.to_numpy()
            else:
                evs = ev0.T @ ev1
            min_ev_shape = min(evs.shape)
            v0 = rng.normal(loc=0, scale=1 / min_ev_shape, size=min_ev_shape)
            Q, eigenvalues, R_t = scipy.sparse.linalg.svds(
                evs,
                k=n_jointly_smooth_functions,
                which="LM",
                tol=self.eigenvector_tolerance,
                v0=v0,
            )
            center = np.row_stack(
                [np.column_stack([Q, Q]), np.column_stack([R_t.T, -R_t.T])]
            )
            right = np.diag(
                np.power(np.concatenate([1 + eigenvalues, 1 - eigenvalues]), -1 / 2)
            )
            jointly_smooth_functions = (
                1 / np.sqrt(2) * eigenvectors_matrix @ center @ right
            )[:, :n_jointly_smooth_functions]
        else:
            n_jointly_smooth_functions = min(
                [self.n_jointly_smooth_functions, eigenvectors_matrix.shape[1]]
            )
            min_ev_shape = min(eigenvectors_matrix.shape)
            v0 = rng.normal(loc=0, scale=1 / min_ev_shape, size=min_ev_shape)
            jointly_smooth_functions, eigenvalues, _ = scipy.sparse.linalg.svds(
                eigenvectors_matrix,
                k=n_jointly_smooth_functions,
                which="LM",
                tol=self.eigenvector_tolerance,
                v0=v0,
            )

        if index_from is not None:
            jointly_smooth_functions = TSCDataFrame.from_same_indices_as(
                index_from,
                jointly_smooth_functions,
                except_columns=[f"jsf{i}" for i in range(n_jointly_smooth_functions)],
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
            kernel_eigenvectors = self.kernel_eigenvectors_[index]
            if isinstance(kernel_eigenvectors, TSCDataFrame):
                kernel_eigenvectors = kernel_eigenvectors.to_numpy()
            if isinstance(self._jointly_smooth_functions_, TSCDataFrame):
                alpha = (
                    kernel_eigenvectors.T @ self._jointly_smooth_functions_.to_numpy()
                )
            else:
                alpha = kernel_eigenvectors.T @ self._jointly_smooth_functions_
            alphas.append(alpha)
            observation = self.observations_[index]
            kernel_output = observation.compute_kernel_matrix(
                new_observation, **self._cdist_kwargs_[index]
            )
            kernel_matrix, _, _ = PCManifoldKernel.read_kernel_output(
                kernel_output=kernel_output
            )
            approx_eigenvectors = kernel_matrix @ mat_dot_diagmat(
                kernel_eigenvectors,
                np.reciprocal(self.kernel_eigenvalues_[index]),
            )

            if isinstance(kernel_matrix, TSCDataFrame):
                index_from: Optional[TSCDataFrame] = kernel_matrix
            elif (
                isinstance(new_observation, TSCDataFrame)
                and kernel_matrix.shape[0] == new_observation.shape[0]
            ):
                index_from = new_observation
            else:
                index_from = None

            if index_from is not None:
                approx_eigenvectors = TSCDataFrame.from_same_indices_as(
                    index_from,
                    approx_eigenvectors,
                    except_columns=[
                        f"aev{i}"
                        for i in range(self.kernel_eigenvectors_[index].shape[1])
                    ],
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
            ignored

        Returns
        -------
        JointlySmoothFunctions
            self
        """
        X = self._validate_datafold_data(
            X=X,
            array_kwargs=dict(
                ensure_min_samples=max(2, self.n_kernel_eigenvectors + 1)
            ),
            tsc_kwargs=dict(ensure_min_samples=max(2, self.n_kernel_eigenvectors + 1)),
        )

        self._setup_feature_attrs_fit(
            X=X,
            features_out=[f"jsf{i}" for i in range(self.n_jointly_smooth_functions)],
        )

        column_splitter = _ColumnSplitter(self.datasets)
        self.observations_ = column_splitter.split(X)

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

        column_splitter = _ColumnSplitter(self.datasets)
        new_observations = column_splitter.split(X)

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

    def score_(self, X, y):
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
