from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from sklearn.base import BaseEstimator
from sklearn.utils import check_scalar
from sklearn.utils.validation import check_is_fitted

from datafold.dynfold.base import TransformType, TSCTransformerMixin
from datafold.pcfold import TSCDataFrame
from datafold.pcfold.eigsolver import compute_kernel_eigenpairs, compute_kernel_svd
from datafold.pcfold.kernels import BaseManifoldKernel
from datafold.utils.general import mat_dot_diagmat


class JointlySmoothFunctions(BaseEstimator, TSCTransformerMixin):
    """Compute jointly smooth functions on multimodal data.

    Parameters
    ----------
    data_splits:
        List of tuples with (name: str, kernel: `BaseManifoldKernel`, indices: slice) to
        describe the splits on the multimodal data in `X`. For each data split a new kernel is
        computed.

    n_kernel_eigenvectors
        The number of eigenvectors to compute from each kernel per split.

    n_jointly_smooth_functions
        The number of jointly smooth functions to compute with a singular value decomposition.

    kernel_eigenvalue_cut_off
        The kernel eigenvectors with a eigenvalue smaller than or equal to
        the cut-off will not be included in the computation of the jointly smooth functions.

    eigenvector_tolerance
        The relative accuracy for eigenvalues, i.e. the stopping criterion. A value of
        zero implies machine precision.

    svd_solver_kwargs
        Keyword arguments passed to the SciPy
        `SVD solver <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html>`__.

    Attributes
    ----------
    X_fit_
        The data `X` passed during `fit`. The data is required to extend the
        ``jointly_smooth_vectors_`` with the Nyström extension for new samples in `transform`.

    kernel_content_: dict
        The computed kernel matrix and data required for the Nyström embedding. The key of
        the dictionary corresponds to the name given in parameter ``data_splits``. (Note that
        the kernel matrix is only stored of ``store_kernel_matrix=True`` during fit).

    kernel_eigenvectors_: Dict
        The kernel eigenvectors for each split. The key of the dictionary corresponds to
        the name given in parameter ``data_splits``. The eigenvectors are used to compute
        and extend the jointly smooth functions.

    kernel_eigenvalues_: Dict
        The kernel eigenvalues for each split. The key of the dictionary corresponds to
        the name given in parameter ``data_splits``. The eigenvalues are used to compute
        and extend the jointly smooth functions.

    jointly_smooth_vectors_: Union[numpy.ndarray, TSCDataFrame]
        The jointly smooth functions evaluated at the training data `X` during `fit`.
        Shape `(n_samples, n_jointly_smooth_functions)`.

    eigenvalues_: numpy.ndarray
        The eigenvalues of the jointly smooth functions.

    References
    ----------
    :cite:t:`dietrich-2022`

    """

    _required_parameters = ["data_splits"]

    def __init__(
        self,
        data_splits: list[tuple],
        *,
        n_kernel_eigenvectors: int = 100,
        n_jointly_smooth_functions: int = 10,
        kernel_eigenvalue_cut_off: float = 0,
        eigenvector_tolerance: float = 1e-6,
        svd_solver_kwargs: Optional[dict] = None,
    ) -> None:
        self.data_splits = data_splits
        self.n_kernel_eigenvectors = n_kernel_eigenvectors
        self.n_jointly_smooth_functions = n_jointly_smooth_functions
        self.kernel_eigenvalue_cut_off = kernel_eigenvalue_cut_off
        self.eigenvector_tolerance = eigenvector_tolerance
        self.svd_solver_kwargs = svd_solver_kwargs

        self.X_fit_: Union[TSCDataFrame, pd.DataFrame, np.ndarray]
        self.kernel_content_: dict
        self.kernel_eigenvectors_: dict
        self.kernel_eigenvalues_: dict
        self.jointly_smooth_vectors_: np.ndarray
        self.eigenvalues_: np.ndarray

    def _validate_parameter(self):
        if (
            not isinstance(self.data_splits, list)
            or not np.array([isinstance(a, tuple) for a in self.data_splits]).all()
        ):
            raise TypeError("parameter 'data_splits' must be a list of tuples")

        if len(self.data_splits) <= 1:
            raise ValueError("parameter 'data_splits' must at least contain two splits")

        if len({n[0] for n in self.data_splits}) != len(self.data_splits):
            raise ValueError("the names in 'data_splits' must be unique")

        for a in self.data_splits:
            if (
                len(a) != 3
                or not isinstance(a[0], str)
                or not isinstance(a[1], BaseManifoldKernel)
                or not isinstance(a[2], (slice, np.ndarray, list))
            ):
                raise TypeError(
                    f"Each tuple must contain three elements with name (type str), "
                    f"kernel (BaseManifoldKernel) and indices (slice or np.ndarray). Got {a}"
                )

        check_scalar(
            self.n_kernel_eigenvectors,
            "n_kernel_eigenvectors",
            target_type=int,
            min_val=1,
        )

        check_scalar(
            self.n_jointly_smooth_functions,
            "n_kernel_eigenvectors",
            target_type=int,
            min_val=1,
        )

        check_scalar(
            self.kernel_eigenvalue_cut_off,
            "kernel_eigenvalue_cut_off",
            target_type=(int, float),
            min_val=0,
        )

    def _compute_kernel_eigenpairs(self, X, kernel_content):
        ret_eigenvectors = dict()
        ret_eigenvalues = dict()

        for name in kernel_content:
            kernel_matrix = kernel_content[name]["kernel_matrix"]
            kernel = kernel_content[name]["kernel"]

            kernel_eigenvalues, kernel_eigenvectors = compute_kernel_eigenpairs(
                kernel=kernel,
                kernel_matrix=kernel_matrix,
                n_eigenpairs=self.n_kernel_eigenvectors,
            )

            if isinstance(X, pd.DataFrame) and kernel_matrix.shape[0] == X.shape[0]:
                kernel_eigenvectors = TSCDataFrame.from_same_indices_as(
                    indices_from=X,
                    values=kernel_eigenvectors,
                    except_columns=[
                        f"ev{i}" for i in range(self.n_kernel_eigenvectors)
                    ],
                )

            bool_evals_cutoff = kernel_eigenvalues > self.kernel_eigenvalue_cut_off
            ret_eigenvalues[name] = kernel_eigenvalues[bool_evals_cutoff]

            if isinstance(kernel_eigenvectors, TSCDataFrame):
                ret_eigenvectors[name] = kernel_eigenvectors.iloc[:, bool_evals_cutoff]
            else:
                ret_eigenvectors[name] = kernel_eigenvectors[:, bool_evals_cutoff]

        return ret_eigenvalues, ret_eigenvectors

    def _compute_jointly_smooth_vectors(
        self, X, kernel_eigenvectors
    ) -> tuple[np.ndarray, np.ndarray]:
        stacked_eigenvectors = np.column_stack(list(kernel_eigenvectors.values()))

        if len(self.kernel_eigenvectors_) == 2:
            # cf. Algorithm 4.1 in https://arxiv.org/pdf/2004.04386.pdf
            W1, W2 = list(kernel_eigenvectors.values())

            # turn to numpy arrays in case of DataFrame
            W1, W2 = np.asarray(W1), np.asarray(W2)

            n_jointly_smooth_functions = min(
                [self.n_jointly_smooth_functions, W1.shape[1] - 1, W2.shape[1] - 1]
            )

            # compute full SVD
            Q, singular_vals, Rt = scipy.linalg.svd(W1.T @ W2, full_matrices=False)

            center_blockmatrix = np.row_stack(
                [np.column_stack([Q, Q]), np.column_stack([Rt.T, -Rt.T])]
            )

            # reduce the elements to the number of functions we actually need
            center_blockmatrix = center_blockmatrix[:, :n_jointly_smooth_functions]

            # diagonal matrix with 1 +/- singular_vals (Gamma in algorithm)
            diagonal_elements = np.concatenate([1 + singular_vals, 1 - singular_vals])
            diagonal_elements = diagonal_elements[:n_jointly_smooth_functions]
            diagonal_elements = np.sqrt(diagonal_elements, out=diagonal_elements)
            diagonal_elements = np.reciprocal(diagonal_elements, out=diagonal_elements)

            jointly_smooth_functions = (
                1.0
                / np.sqrt(2.0)
                * stacked_eigenvectors
                @ mat_dot_diagmat(center_blockmatrix, diagonal_elements)
            )
        else:
            # cf. Algorithm 4.2 in https://arxiv.org/pdf/2004.04386.pdf
            n_jointly_smooth_functions = min(
                [self.n_jointly_smooth_functions, stacked_eigenvectors.shape[1]]
            )

            jointly_smooth_functions, singular_vals, _ = compute_kernel_svd(
                stacked_eigenvectors,
                n_svdtriplet=n_jointly_smooth_functions,
                **(self.svd_solver_kwargs or {}),
            )

        if isinstance(X, TSCDataFrame):
            jointly_smooth_functions = TSCDataFrame.from_same_indices_as(
                X,
                jointly_smooth_functions,
                except_columns=self.get_feature_names_out(),
            )

        return singular_vals, jointly_smooth_functions

    def _nystrom(self, X):
        """Embed out-of-sample points with Nyström extension.

        Parameters
        ----------
        X: Union[TSCDataFrame, pandas.DataFrame]
             The out-of-sample data with shape `(n_samples, n_features_in_)`.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_jointly_smooth_functions)`
        """
        kernel_matrices: dict = self._kernel_content_transform(X)
        eigenvectors = dict()
        alphas = dict()

        for name in kernel_matrices:
            kernel_evec = self.kernel_eigenvectors_[name]

            if isinstance(kernel_evec, TSCDataFrame):
                kernel_evec = kernel_evec.to_numpy()

            if isinstance(self.jointly_smooth_vectors_, TSCDataFrame):
                alpha = kernel_evec.T @ self.jointly_smooth_vectors_.to_numpy()
            else:
                alpha = kernel_evec.T @ self.jointly_smooth_vectors_

            alphas[name] = alpha

            eigenvectors[name] = kernel_matrices[name] @ mat_dot_diagmat(
                kernel_evec,
                np.reciprocal(self.kernel_eigenvalues_[name]),
            )

        _dtype_probe = list(alphas.values())[0].dtype
        f_m_star = np.zeros(
            [X.shape[0], self.n_jointly_smooth_functions], dtype=_dtype_probe
        )
        tmp = np.zeros_like(f_m_star)

        for name in eigenvectors:
            f_m_star += np.dot(eigenvectors[name], alphas[name], out=tmp)

        f_m_star /= len(self.data_splits)  # divide by number of splits

        if isinstance(X, TSCDataFrame) and hasattr(self, "feature_names_in_"):
            f_m_star = TSCDataFrame.from_same_indices_as(
                X, f_m_star, except_columns=self.get_feature_names_out()
            )

        return f_m_star

    def _kernel_content_fit(self, X: TransformType):
        return_content: dict[str, dict] = dict()

        for _, split in enumerate(self.data_splits):
            name, kernel, indices = split

            # copy kernels to not mutate the original attribute(sklearn conform)
            kernel = deepcopy(kernel)

            if isinstance(X, pd.DataFrame):
                kernel_matrix = kernel(X.iloc[:, indices])
            else:
                kernel_matrix = kernel(X[:, indices])

            return_content[name] = dict()
            return_content[name]["kernel_matrix"] = kernel_matrix
            return_content[name]["kernel"] = kernel

        return return_content

    def _kernel_content_transform(self, X):
        return_content = dict()

        for _, split in enumerate(self.data_splits):
            name, _, indices = split
            kernel = self.kernel_content_[name]["kernel"]

            if isinstance(X, pd.DataFrame):
                kernel_matrix = kernel(
                    self.X_fit_.iloc[:, indices],
                    X.iloc[:, indices],
                )
            else:
                kernel_matrix = kernel(
                    self.X_fit_[:, indices],
                    X[:, indices],
                )

            return_content[name] = kernel_matrix

        return return_content

    def get_feature_names_out(self, input_features=None):
        return np.array([f"jsf{i}" for i in range(self.n_jointly_smooth_functions)])

    def fit(self, X: TransformType, y=None, **fit_params) -> "JointlySmoothFunctions":
        """Compute the jointly smooth functions on training data `X`.

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
        self.X_fit_ = self._validate_datafold_data(
            X=X, ensure_min_samples=max(2, self.n_kernel_eigenvectors + 1)
        )

        self._validate_parameter()

        store_kernel_matrix = self._read_fit_params(
            attrs=[("store_kernel_matrix", False)],
            fit_params=fit_params,
        )

        self._setup_feature_attrs_fit(X=X)
        self.kernel_content_ = self._kernel_content_fit(self.X_fit_)
        (
            self.kernel_eigenvalues_,
            self.kernel_eigenvectors_,
        ) = self._compute_kernel_eigenpairs(X, self.kernel_content_)

        if not store_kernel_matrix:
            # release storage by deleting kernel matrix
            for name in self.kernel_content_:
                del self.kernel_content_[name]["kernel_matrix"]

        (
            self.eigenvalues_,
            self.jointly_smooth_vectors_,
        ) = self._compute_jointly_smooth_vectors(X, self.kernel_eigenvectors_)

        return self

    def transform(self, X: TransformType) -> TransformType:
        """Evaluate jointly smooth functions for out-of-sample data with Nyström extension.

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data points of shape `(n_samples, n_features)` to be mapped.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_jointly_smooth_functions)`
        """
        check_is_fitted(
            self,
            (
                "kernel_content_",
                "kernel_eigenvectors_",
                "kernel_eigenvalues_",
                "jointly_smooth_vectors_",
                "eigenvalues_",
            ),
        )

        X = self._validate_datafold_data(X=X)
        self._validate_feature_input(X, direction="transform")

        f_m_star = self._nystrom(X)

        return f_m_star

    def fit_transform(self, X: TransformType, y=None, **fit_params) -> TransformType:
        """Compute and return jointly smooth functions evaluated at training data `X`.

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
            X, ensure_min_samples=max(2, self.n_kernel_eigenvectors)
        )
        self.fit(X=X, y=y, **fit_params)

        return self.jointly_smooth_vectors_

    def score_(self, X, y):
        """Compute a score for hyperparameter optimization.

        Returns
        -------
        float
            The sum of the truncated energies.
        """
        return self._compute_truncated_energies().sum()

    def _compute_truncated_energies(self) -> np.ndarray:
        """Compute the truncated energy for each kernel eigenvector.

        Returns
        -------
        np.ndarray
            The truncated energies of shape `(n_observations, n_jointly_smooth_functions)`.
        """
        truncated_energies = dict()
        for name in self.kernel_eigenvectors_:
            truncated_energy = np.linalg.norm(
                np.asarray(self.kernel_eigenvectors_[name]).T
                @ self.jointly_smooth_vectors_,
                axis=0,
            )
            truncated_energies[name] = truncated_energy**2

        return np.array(truncated_energies.values())

    def compute_E0(self) -> float:
        """Compute a threshold for the eigenvalues of the jointly smooth functions.

        Returns
        -------
        float
            The E0 threshold value from :cite:t:`dietrich-2022`.
        """
        kernel_evecs = list(self.kernel_eigenvectors_.values())

        noisy = kernel_evecs[-1].copy()

        rng = np.random.default_rng(1)  # expose seed to user if required
        rng.shuffle(noisy)

        kernel_eigenvectors = kernel_evecs[:-1]
        kernel_eigenvectors.append(noisy)

        eigenvectors_matrix = np.column_stack(kernel_evecs)

        if len(kernel_eigenvectors) == 2:
            ev0 = kernel_eigenvectors[0]
            ev1 = kernel_eigenvectors[1]

            gamma = scipy.sparse.linalg.svds(
                ev0.T @ ev1,
                k=self.n_jointly_smooth_functions,
                which="LM",
                return_singular_vectors=False,
            )
        else:
            gamma = scipy.sparse.linalg.svds(
                eigenvectors_matrix,
                k=self.n_jointly_smooth_functions,
                which="LM",
                return_singular_vectors=False,
            )

        gamma = np.sort(gamma)[::-1]
        E0 = (1 + gamma[2]) / 2  # page 6 in https://arxiv.org/pdf/2004.04386.pdf
        return E0
