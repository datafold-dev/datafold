#!/usr/bin/env python3
import abc
import inspect
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import scipy.sparse
import scipy.spatial
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree, KNeighborsTransformer, NearestNeighbors
from sklearn.utils.validation import check_scalar

from datafold.utils.general import if1dim_colvec, if1dim_rowvec, is_integer

try:
    # rdist is an optional distance algorithm backend -- an import error is raised only
    # when one attempts to use rdist and the import was not successful
    import rdist

    IS_IMPORTED_RDIST = True
except ImportError:
    rdist = None
    IS_IMPORTED_RDIST = False


class DistanceAlgorithm(metaclass=abc.ABCMeta):
    """Abstract base class for distance matrix algorithms (dense or sparse).

    Important aspects and conventions for the distance algorithms:

    * The terms "pair-wise" (pdist) and "component-wise" (cdist) are
      adapted from the scipy's distance matrix computations
      :class:`scipy.sparse.spatial.pdist` and :class:`scipy.sparse.spatial.cdist`
    * A sparse distance matrix with a distance (either `k`-neighbors or with a radius cut-off)
      does not store all distance pairs. Importantly, this means that the sparse matrix
      **must** store "real distance zeros" (introduced by duplicate points or self-distances
      in case of `pdist`). This sometimes requires a workaround in matrix operations, so that
      stored distance-zeros are not removed.

    Parameters
    ----------
    metric
        distance metric to compute

    is_symmetric
        indicate whether the distance matrix is symmetric (typically the standard
        k-nearest-neighbor is not symmetric)

    k
        * for k-nearest-neighbors the number of neighbors
        * for radius-based distance algorithms, there is a follow-up routine to make sure that
          each sample has at least ``kmin`` neighbors (including distance pairs that are
          larger than radius)
    """

    # distances cannot be negative - an easy to identify negative value
    _invalid_dist_value = -999

    def __init__(
        self,
        metric: str,
        is_symmetric: bool,
        cut_off: Optional[float] = None,
        k: Optional[float] = None,
    ) -> None:
        assert hasattr(
            self, "name"
        ), f"Attribute 'name' is missing in subclass {type(self)}."
        self.name: str

        self.metric = metric
        self.is_symmetric = is_symmetric

        if cut_off is not None:
            self._set_attrs_range(cut_off=cut_off, kmin=k)

        if k is not None:
            self._set_attrs_knn(k)

    def _set_attrs_range(self, cut_off, kmin):
        if kmin is not None and not is_integer(kmin):
            raise TypeError(
                f"parameter 'kmin' must be an integer type or None. Got: {type(kmin)}"
            )

        check_scalar(
            cut_off,
            "cut_off",
            target_type=(float, int),
            min_val=0,
            include_boundaries="neither",
        )

        if self.metric == "sqeuclidean":
            if cut_off is not None:
                # NOTE: this is a special case. Usually the cut_off is represented in the
                # respective metric. However, for the 'sqeuclidean' case we use the
                # 'euclidean' metric for the cut-off.
                cut_off = cut_off**2

        self.cut_off = cut_off
        self.kmin = kmin

    @property
    def dist_type(self):
        if hasattr(self, "cut_off"):
            return "range-nn"
        elif hasattr(self, "k"):
            return "knn"
        else:
            return "full"

    def _set_attrs_knn(self, k):
        if k <= 0:
            raise ValueError(f"parameter '{k=}' must be a positive integer")
        self.k = k

    def __repr__(self):
        _name = self.__class__.__name__

        if self.dist_type == "range-nn":
            return (
                _name + f"({self.metric=}, {self.is_symmetric=}, {self.is_sparse=}, "
                f"{self.cut_off=}, {self.kmin=})".replace("self.", "")
            )
        elif self.dist_type == "knn":
            return (
                _name + f"({self.metric=}, {self.is_symmetric=}, {self.is_sparse=}, "
                f"{self.k=})".replace("self.", "")
            )
        else:
            return _name + f"({self.metric=}, {self.is_symmetric=})".replace(
                "self.", ""
            )

    @property
    def is_sparse(self):
        if self.dist_type == "range-nn":
            return self.cut_off is not None
        elif self.dist_type == "knn":
            return self.k < np.inf
        else:
            return False

    def _validate_cut_off(self, cut_off):
        if cut_off is None or np.isinf(cut_off):
            cut_off = None  # default to None and use dense case if np.isinf(cut_off)
        else:
            try:
                cut_off = float(cut_off)  # make sure to only deal with Python built-in
            except ValueError:
                raise TypeError(f"type(cut_off)={type(cut_off)} must be of type float")

            if cut_off <= 0:
                raise ValueError(
                    f"cut_off={cut_off} must be a positive number number of "
                    f"type 'float'"
                )
        if self.metric == "sqeuclidean":
            if cut_off is not None:
                # NOTE: this is a special case. Usually the cut_off is represented in the
                # respective metric. However, for the 'sqeuclidean' case we use the
                # 'euclidean' metric for the cut-off.
                cut_off = cut_off**2

        return cut_off

    def _validate_X_Y(self, X, Y):
        X = np.asarray(X)
        X = if1dim_colvec(X)

        if X.shape[0] <= 1:
            raise ValueError("Number of samples has to be greater than 1.")

        is_pdist = Y is None

        if not is_pdist:
            Y = np.asarray(Y)
            Y = if1dim_rowvec(Y)

            if X.shape[1] != Y.shape[1]:
                raise ValueError(
                    f"Mismatch in point dimension: "
                    f"X.shape[1]={X.shape[1]} != Y.shape[1]={Y.shape[1]} "
                )
        return X, Y, is_pdist

    def _validate_metric(self, valid, metric):
        if metric not in valid:
            raise ValueError(
                f"Distance algorithm has invalid metric = {metric}. Valid metrics "
                f"are = {valid}."
            )

    def _sparse2dense_matrix(self, distance_matrix):
        if scipy.sparse.issparse(distance_matrix) and self.cut_off is None:
            # dense case stored in a sparse distance matrix -> convert to np.ndarray
            distance_matrix = distance_matrix.toarray()

        return distance_matrix

    def _set_zeros_diagonal_sparse(self, distance_matrix):
        # This function sets the diagonal to zero of a sparse matrix.

        # Some algorithms don't store the zeros on the diagonal for the pdist case.
        # However, this is critical if afterwards the kernel is applied
        # kernel(distance_matrix).
        #   -> kernel(distance)=0 but correct is kernel(distance)=1
        #      (for a stationary kernel)
        # The issue is:
        # * We neglect not zeros but large values (e.g. cut_off=100 we ignore larger
        #   values and do not store them)
        # * The sparse matrix formats see the "not stored values" equal to zero,
        #   however, there are also "true zeros" for duplicates. We HAVE to store these
        #   zero values, otherwise the kernel values are wrong on the opposite extreme
        #   end (i.e. 0 instead of 1, for stationary kernels).

        assert (
            scipy.sparse.issparse(distance_matrix)
            and distance_matrix.shape[0] == distance_matrix.shape[1]
        )

        # in case there are duplicate rows -> set to invalid value
        distance_matrix.data[distance_matrix.data == 0] = self._invalid_dist_value

        # convert to lil-format, because it is more efficient to set the diag=0
        distance_matrix = distance_matrix.tolil()
        distance_matrix.setdiag(self._invalid_dist_value)

        # turn back to csr and set the invalid to "true zeros"
        distance_matrix = distance_matrix.tocsr()
        distance_matrix.data[distance_matrix.data == self._invalid_dist_value] = 0

        return distance_matrix

    def _dense2csr_matrix(self, distance_matrix, cut_off):
        # This is the same issue as described in _set_zeros_sparse_diagonal
        # All true zeros have to be kept in the sparse matrix. This is a workaround as
        # csr_matrix(distance_matrix) removes internally all zeros.

        distance_matrix[distance_matrix == 0] = self._invalid_dist_value
        distance_matrix[distance_matrix >= cut_off] = 0
        distance_matrix = scipy.sparse.csr_matrix(distance_matrix)
        distance_matrix.data[distance_matrix.data == self._invalid_dist_value] = 0

        return distance_matrix

    @staticmethod  # static, so that it can be overwritten
    def _ensure_kmin_nearest_neighbor(
        X: np.ndarray,
        Y: Optional[np.ndarray],
        metric: str,
        kmin: int,
        distance_matrix: scipy.sparse.csr_matrix,
    ) -> scipy.sparse.csr_matrix:
        """Computes `kmin` nearest neighbors for all points that in the current distance
        matrix have not at least `kmin` neighbors, yet.

        This function is especially useful to make sure that a neighborhood graph (
        described by a distance matrix) is fully connected. If outlier have no
        (or only self neighbor), then this can have unwanted side effects in some
        applications.

        Internally, the k-NN query is carried out using :class:`sklearn.neighbors.BallTree`.

        Parameters
        ----------
        X
            Point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Query Point cloud of shape `(n_samples_Y, n_features_Y)`. If not given,
            then `Y=X` (pdist case).

        metric
            distance metric

        kmin
            Minimum number of neighbors. Note, for the `pdist` case, `kmin==1` is already
            fulfilled by the diagonal line (self-distances).

        distance_matrix
            Current distance matrix to which the missin distance pairs are inserted.

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix with shape `(n_samples_Y, n_samples_X)`
        """
        current_nnz = distance_matrix.getnnz(axis=1)
        knn_query_indices = np.where(current_nnz < kmin)[0]
        is_pdist = Y is None

        if len(knn_query_indices) != 0:
            if is_pdist:
                Y = X.view()
            else:
                assert isinstance(Y, np.ndarray)
                if (
                    Y.shape[0] != distance_matrix.shape[0]
                    or X.shape[0] != distance_matrix.shape[1]
                ):
                    raise ValueError("Mismatch between dataset and distance matrix.")

            _ball_tree = BallTree(X, leaf_size=40, metric="euclidean")
            distances, columns_indices = _ball_tree.query(
                Y[knn_query_indices, :],
                k=kmin,
                return_distance=True,
                dualtree=False,
                breadth_first=False,
                sort_results=False,
            )

            distances = np.reshape(
                distances, newshape=np.prod(distances.shape), order="C"
            )

            # Note: duplicates and trivial self-distances in the pdist are assumed to already
            # covered by the DistanceAlgorithm (always contained in the radius!)
            nnz_distance_mask = (distances != 0).astype(bool)
            distances = distances[nnz_distance_mask]

            knn_query_indices = np.repeat(knn_query_indices, kmin)[nnz_distance_mask]

            columns_indices = np.reshape(
                columns_indices, newshape=np.prod(columns_indices.shape), order="C"
            )[nnz_distance_mask]

            if is_pdist:
                knn_query_indices, columns_indices, distances = np.unique(
                    np.vstack(
                        [
                            np.column_stack(
                                [knn_query_indices, columns_indices, distances]
                            ),
                            np.column_stack(
                                [columns_indices, knn_query_indices, distances]
                            ),
                        ]
                    ),
                    axis=0,
                ).T

            kmin_elements_csr = scipy.sparse.csr_matrix(
                (distances, (knn_query_indices, columns_indices)),
                shape=distance_matrix.shape,
            )

            # TODO: This changes the sparsity structure and raises a warning. I am not sure
            #  how to make this right. For this attempt the tests fail:
            #  distance_matrix.tolil(copy=False)[
            #      kmin_elements_csr.nonzero()
            #  ] = kmin_elements_csr.data
            #  maybe the best is to combine the elements of kmin_elements_csr and distance
            #  matrix into one set (sorting out the upper triangle for pdist) and then
            #  create a new sparse matrix...
            distance_matrix[kmin_elements_csr.nonzero()] = kmin_elements_csr.data

        return distance_matrix.tocsr()

    def _handle_kmin(self, X, Y, distance_matrix):
        """Use only for radius/range based algorithms."""
        is_pdist = Y is None

        distance_matrix = self._validate_final_sparse_matrix(
            distance_matrix=distance_matrix
        )

        # only for the sparse case we care about kmin:
        apply_kmin_procedure = self.kmin is not None and (
            (self.kmin > 0 and not is_pdist) or (self.kmin > 1 and is_pdist)
        )

        if apply_kmin_procedure:
            # kmin == 1 and is_pdist does not need treatment because the diagonal is set.
            distance_matrix = self._ensure_kmin_nearest_neighbor(
                X,
                Y,
                metric=self.metric,
                kmin=self.kmin,
                distance_matrix=distance_matrix,
            )

        return distance_matrix

    def _validate_final_sparse_matrix(self, distance_matrix):
        # actually return a dense matrix
        if scipy.sparse.issparse(distance_matrix) and np.isinf(self.cut_off):
            # dense case stored in a sparse distance matrix -> convert to np.ndarray
            distance_matrix = distance_matrix.toarray()
        elif not isinstance(distance_matrix, scipy.sparse.csr_matrix):
            # Currently, we only return a sparse matrix in CSR format.
            distance_matrix = distance_matrix.tocsr()

        if scipy.sparse.issparse(distance_matrix):
            # sort_indices returns immediately if indices are already sorted.
            # If not sorted, the call could be costly (depending on nnz), but is better for
            # follow-up computations.
            distance_matrix.sort_indices()

        return distance_matrix

    @abc.abstractmethod
    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Compute distance matrix.

        If only the reference dataset (`X`) is given, then the distances are pair-wise. From
        this the following distance matrix properties follow:

        * square
        * diagonal contains distance to itself and are therefore zero
        * symmetric

        If an additional query dataset is given, then the distance matrix properties follow:

        * rectangular matrix of shape `(n_samples_Y, n_samples_X)`
        * outlier points can lead to columns / rows of zero
        * duplicated points between `X` and `Y` have zero entries on the diagonal


        Attributes
        ----------
        X
            Reference dataset of shape `(n_samples_X, n_features)`.

        Y
            Query dataset of shape `(n_samples_Y, n_features)`. If set then the computation
            is component-wise and if ``None``, the reference dataset is taken as the query
            points (i.e. `Y=X`).
        """


class BruteForceDist(DistanceAlgorithm):
    """Computes all distance pairs in the distance matrix.

    Based on parameter `metric` and argument `exact_numeric` either backend is used:

        * SciPy with `scipy.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`__ and
          `scipy.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`__
        * scikit-learn with `sklearn.pairwise_distances <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html>`__

    For an explanation of how `exact_numeric = False` is beneficial, see the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html>`__

    Parameters
    ----------
    metric
        Metric to compute, see documentation of backend algorithms what metrics are supported.

    exact_numeric
        If False, computes Euclidean distances more efficiently
        t the cost of introducing numerical noise. Empirically `~1e-14` for "sqeuclidean"
        metric and `~1e-7` for "euclidean" metric.

    cut_off
        distances larger than `cut_off` are set to zero

        .. note::

            Distances with larger distance are removed after a full memory
            allocation of the distance matrix. It is recommended to use
            distance algorithms that directly reduce the number of computations.

    **backend_options
        Keyword arguments handled to the executing backend (depending on ``exact_numeric``
        parameter).
    """  # noqa: E501

    name = "brute"

    def __init__(
        self,
        metric: str,
        exact_numeric: bool = True,
        cut_off: Optional[Union[float, int]] = np.inf,
        **backend_options,
    ):
        self.exact_numeric = exact_numeric
        self.backend_options = backend_options
        super().__init__(metric=metric, is_symmetric=True, cut_off=cut_off or np.inf)

    @classmethod
    def is_symmetric(cls):
        return True

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Compute distance matrix.

        Parameters
        ----------
        X
            Reference dataset of shape `(n_samples_X, n_features)`.

        Y
            Query dataset of shape `(n_samples_Y, n_features)`. If set then the computation
            is component-wise and if ``None``, the reference dataset is taken as the query
            points (i.e. `Y=X`).

        Returns
        -------
        np.ndarray, scipy.sparse.csr_matrix
            distance matrix
        """
        X, Y, is_pdist = self._validate_X_Y(X, Y)

        if is_pdist:
            if self.exact_numeric:
                _pdist = pdist(X, metric=self.metric, **self.backend_options)
                distance_matrix = squareform(_pdist)
            else:
                # sklearn uses a numeric inexact but faster implementation
                distance_matrix = pairwise_distances(
                    X, metric=self.metric, **self.backend_options
                )
        else:
            if self.exact_numeric:
                distance_matrix = cdist(
                    Y, X, metric=self.metric, **self.backend_options
                )
            else:
                distance_matrix = pairwise_distances(
                    Y, X, metric=self.metric, **self.backend_options
                )

        if not np.isinf(self.cut_off):
            distance_matrix = self._dense2csr_matrix(
                distance_matrix, cut_off=self.cut_off
            )

            distance_matrix = self._validate_final_sparse_matrix(distance_matrix)

        return distance_matrix


class RDist(DistanceAlgorithm):
    """Sparse distance matrix algorithm rdist, for point clouds with manifold assumption.

    .. note::
        The dependency on the Python package is optional. The package is currently not
        published.

    Parameters
    ----------
    metric
        "euclidean" or "sqeuclidean"

    cut_off
        Distance values (always Euclidean metric) that are larger are not stored in
        distance matrix.

    Raises
    ------
    ImportError
        if rdist is not installed, but selected as backend

    References
    ----------
    .. todo::
        include own paper if published

    """

    name = "rdist" if IS_IMPORTED_RDIST else None  # type: ignore

    def __init__(self, cut_off, kmin=None, metric="euclidean", **backend_options):
        if not IS_IMPORTED_RDIST:
            raise ImportError("Could not import rdist. Check if it is installed.")
        self.backend_options = backend_options
        self._validate_metric(valid=["euclidean", "sqeuclidean"], metric=metric)
        super().__init__(metric=metric, is_symmetric=True, cut_off=cut_off, k=kmin)

    @classmethod
    def is_symmetric(cls):
        return True

    def _adapt_radius(self):
        # Generally: the cut-off is represented like self.metric. The scipy.kdtree can
        # only compute Euclidean distances. Therefore, undo the squaring of cut-off.
        # For sqeuclidean distance, the squaring has to be done after the
        # distance matrix was computed.

        if self.metric == "sqeuclidean":
            return np.sqrt(self.cut_off)
        else:
            return self.cut_off

    def _get_dist_options(self):
        return {"max_incr_radius": 0, "kmin": 0}

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
    ) -> scipy.sparse.csr_matrix:
        """Compute distance matrix.

        Parameters
        ----------
        X
            Reference dataset of shape `(n_samples_X, n_features)`.

        Y
            Query dataset of shape `(n_samples_Y, n_features)`. If set then the computation
            is component-wise and if ``None``, the reference dataset is taken as the query
            points (i.e. `Y=X`).

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix
        """
        is_pdist = Y is None

        radius = self._adapt_radius()

        if is_pdist or not hasattr(self, "rdist_"):
            self.rdist_ = rdist.Rdist(X, **self.backend_options)

        if is_pdist:
            distance_matrix = self.rdist_.sparse_pdist(
                r=radius, rtype="radius", **self._get_dist_options()
            )
        else:
            distance_matrix = self.rdist_.sparse_cdist(
                req_points=Y, r=radius, rtype="radius", **self._get_dist_options()
            )

        if self.metric == "euclidean":
            distance_matrix.data = np.sqrt(
                distance_matrix.data, out=distance_matrix.data
            )

        return self._handle_kmin(X, Y, distance_matrix)


class ScipyKdTreeDist(DistanceAlgorithm):
    """Sparse distance matrix computation using scipy's kd-tree implementation.

    Parameters
    ----------
    cut_off: float
        Distance values (always Euclidean metric) that are larger are not stored in
        distance matrix.

    metric: str
        "euclidean" or "sqeuclidean"

    kmin: int
        store at least ``kmin`` samples per sample

    backend_options
        key word arguments passed to `cKDTree <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html>`__

    References
    ----------
    :class:`scipy.spatial.cKDTree`
    :meth:`scipy.spatial.KDTree.sparse_distance_matrix`
    """

    name = "scipy.kdtree"

    def __init__(self, cut_off, metric="euclidean", kmin=None, **backend_options):
        self.backend_options = backend_options
        self._validate_metric(valid=["euclidean", "sqeuclidean"], metric=metric)
        super().__init__(
            metric=metric, is_symmetric=True, cut_off=cut_off or np.inf, k=kmin
        )

    @classmethod
    def is_symmetric(cls):
        return True

    def _get_max_distance(self):
        # Generally: the cut-off is represented like self.metric. The scipy.kdtree can
        # only compute Euclidean distances. Therefore, undo the squaring of cut-off.
        # For sqeuclidean distance, the squaring has to be done after the
        # distance matrix was computed.

        if self.metric == "sqeuclidean":
            # note if max_distance==np.inf, it is still np.inf
            return np.sqrt(self.cut_off)
        else:
            return self.cut_off

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
    ) -> scipy.sparse.csr_matrix:
        """Compute distance matrix.

        Parameters
        ----------
        X
            Reference dataset of shape `(n_samples_X, n_features)`.

        Y
            Query dataset of shape `(n_samples_Y, n_features)`. If set then the computation
            is component-wise and if ``None``, the reference dataset is taken as the query
            points (i.e. `Y=X`).

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix
        """
        X, Y, is_pdist = self._validate_X_Y(X, Y)

        if is_pdist or not hasattr(self, "kdtree_x_"):
            self.kdtree_x_ = scipy.spatial.cKDTree(X, **self.backend_options)

        if is_pdist:
            distance_matrix = self.kdtree_x_.sparse_distance_matrix(
                self.kdtree_x_,
                max_distance=self._get_max_distance(),
                output_type="coo_matrix",
            )
        else:
            if not hasattr(self, "kdtree_x_"):
                # else reuse from pdist call
                self.kdtree_x_ = scipy.spatial.cKDTree(X, **self.backend_options)

            kdtree_y = scipy.spatial.cKDTree(Y, **self.backend_options)

            distance_matrix = kdtree_y.sparse_distance_matrix(
                self.kdtree_x_,
                max_distance=self._get_max_distance(),
                output_type="coo_matrix",
            )

        if self.metric == "sqeuclidean":
            distance_matrix.data = np.square(distance_matrix.data)

        return self._handle_kmin(X, Y, distance_matrix=distance_matrix)


class SklearnBalltreeDist(DistanceAlgorithm):
    """Distance matrix using ball tree implementation from scikit-learn.

    Parameters
    ----------
    cut_off
        Distance values (always Euclidean metric) that are larger are not stored in
        distance matrix.

    metric
        see `metric parameter in `sklearn.NearestNeighbor <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html>`__

    kmin
        store at least ``kmin`` samples per sample

    **backend_options
        key word arguments passed to `sklearn.NearestNeighbors <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html>`__

    See Also
    --------
    :class:`sklearn.neighbors.NearestNeighbors`
    """

    name = "sklearn.balltree"

    def __init__(
        self,
        cut_off,
        metric: str = "euclidean",
        kmin: Optional[int] = None,
        **backend_options,
    ):
        self.backend_options = backend_options
        self._validate_metric(valid=["euclidean", "sqeuclidean"], metric=metric)
        super().__init__(
            metric=metric, is_symmetric=True, cut_off=cut_off or np.inf, k=kmin
        )

    @classmethod
    def is_symmetric(cls):
        return True

    def _map_metric_and_radius(self):
        if self.metric == "sqeuclidean":
            return "euclidean", np.sqrt(self.cut_off)
        else:
            return self.metric, self.cut_off

    def _fit_nearest_neighbor(self, X):
        metric, radius = self._map_metric_and_radius()
        nn = NearestNeighbors(
            radius=radius, algorithm="ball_tree", metric=metric, **self.backend_options
        )
        nn.fit(X)

        return nn

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
    ) -> scipy.sparse.csr_matrix:
        """Compute distance matrix.

        Parameters
        ----------
        X
            Reference dataset of shape `(n_samples_X, n_features)`.

        Y
            Query dataset of shape `(n_samples_Y, n_features)`. If set then the computation
            is component-wise and if ``None``, the reference dataset is taken as the query
            points (i.e. `Y=X`).

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix
        """
        X, Y, is_pdist = self._validate_X_Y(X, Y)

        if is_pdist or not hasattr(self, "nn_"):
            self.nn_ = self._fit_nearest_neighbor(X)

        if is_pdist:
            distance_matrix = self.nn_.radius_neighbors_graph(mode="distance")
        else:
            distance_matrix = self.nn_.radius_neighbors_graph(Y, mode="distance")

        if self.metric == "sqeuclidean":
            distance_matrix.data = np.square(
                distance_matrix.data, out=distance_matrix.data
            )

        if is_pdist:
            distance_matrix = self._set_zeros_diagonal_sparse(distance_matrix)

        return self._handle_kmin(X, Y, distance_matrix)


class SklearnKNN(DistanceAlgorithm):
    name = "sklearn.knn"

    def __init__(self, metric, k, **backend_options):
        self.backend_options = backend_options
        super().__init__(metric=metric, is_symmetric=False, k=k)

    @classmethod
    def is_symmetric(cls):
        return False

    def __call__(self, X, Y=None):
        X, Y, is_pdist = self._validate_X_Y(X, Y=Y)

        if is_pdist or not hasattr(self, "nn_"):
            self.nn_ = KNeighborsTransformer(
                mode="distance",
                n_neighbors=self.k,
                metric=self.metric,
                **self.backend_options,
            )
            self.nn_.fit(X)

        if is_pdist:
            # here, the connection to self is not computed (but inserted below)
            distance_matrix = self.nn_.kneighbors_graph(
                n_neighbors=self.k - 1, mode="distance"
            )
            distance_matrix = self._set_zeros_diagonal_sparse(distance_matrix)
        else:
            distance_matrix = self.nn_.kneighbors_graph(
                Y, n_neighbors=self.k, mode="distance"
            )

        return distance_matrix


class GuessOptimalDist(DistanceAlgorithm):
    """Tries to guess a suitable algorithm based on sparsity, metric and installed
    backends algorithms.

    Parameters
    ----------
    metric
        distance metric to compute

    is_symmetric
        Whether it is required to compute a symmetric matrix (if True this excludes
        sparse `k`-nearest-neighbor algorithms)

    cut_off
        If a valid float, then compute a sparse (symmetric) radius-based distance matrix.

    k
        Compute `k`-nearest-neighbor if non-symmetric sparse matrix or provide at least `k`
        neighbors in a radius-based sparse matrix.

    """

    name = "guess_optimal"

    def __new__(cls, metric="euclidean", is_symmetric=True, cut_off=np.inf, kmin=None):
        cut_off = cut_off or np.inf
        cls.is_symmetric = lambda: is_symmetric

        if np.isinf(cut_off):  # dense case
            backend_class = BruteForceDist(cut_off=cut_off, metric=metric)
        else:
            if IS_IMPORTED_RDIST and metric in ["euclidean", "sqeuclidean"]:
                backend_class = RDist(cut_off=cut_off, metric=metric, kmin=kmin)
                assert backend_class is not None

            elif metric in ["euclidean", "sqeuclidean"]:
                backend_class = ScipyKdTreeDist(
                    cut_off=cut_off, metric=metric, kmin=kmin
                )
            else:
                backend_class = BruteForceDist(cut_off=cut_off, metric=metric)

        return backend_class


def _all_available_distance_algorithm(require_symmetric: bool = False):
    """Searches for valid subclasses of :py:class:`DistanceAlgorithm`.

    Parameters
    ----------
    require_symmetric
        Only return ``DistanceAlgorithm`` that compute a symmetric distance matrix.

    Returns
    -------
    list
        all valid subclasses
    """
    all_backends = DistanceAlgorithm.__subclasses__()

    return_backends = list()
    # This is the case if backend is given as a str, now we look for the matching
    # DistanceAlgorithm.name
    for b in all_backends:
        # Tests are only for security that all algorithms have the mandatory (static)
        # attribute set.

        try:
            # Attribute error is raised here if 'backend_name' does not exist
            # If 'backend_name' is set to none, then the implementation is not
            # considered e.g. because dependencies are not met in case of rdist.
            if isinstance(b.name, str):
                if require_symmetric and b.is_symmetric():  # type: ignore
                    return_backends.append(b)
                elif not require_symmetric:
                    return_backends.append(b)
        except AttributeError:
            raise NotImplementedError(
                f"Bug: class {type(b)} has no 'name' attribute or it is not of "
                f"type 'str'. Check implementation."
            )

    return return_backends


def get_backend_distance_algorithm(
    backend, require_symmetric=False
) -> type[DistanceAlgorithm]:
    """Selects and validates the backend class for distance matrix computation.

    Parameters
    ----------
    backend
        * ``str`` - maps to the algorithms
        * ``DistanceAlgorithm`` - returns same object if valid

    require_symmetric
        If True only return ``DistanceAlgorithm`` classes that compute a symmetric distance
        matrix.

    Returns
    -------
    DistanceAlgorithm
    """
    if backend is None:
        raise ValueError("backend cannot be None")

    all_backends = _all_available_distance_algorithm(require_symmetric)

    # This is the case if a user chooses backend by object instead of "name"
    # attribute.
    if backend in all_backends:
        return backend

    selected_backend = None
    for b in all_backends:
        # look up for the backend algorithm with the name implemented
        if b.name == backend:
            selected_backend = b
            break
    else:
        raise ValueError(f"Could not find backend {backend}")

    return selected_backend


def init_distance_algorithm(
    backend="guess_optimal",
    metric="euclidean",
    cut_off=None,
    k=None,
    **backend_options,
) -> DistanceAlgorithm:
    """Initialize a distance matrix by name and keywords.

    Parameters
    ----------
    backend: str
        Backend to compute distance matrix.

    metric
        Distance metric. Needs to be supported by backend.

    cut_off
        Distances larger than `cut_off` are set to zero. The parameter controls the
        degree of sparsity in the distance matrix.

        .. note::
            The pseudo-metric "sqeuclidean" is handled differently in a way that the
            cut-off must be stated in in Eucledian distance (not squared cut-off).

    k
        Minimum number of neighbors per point. Ignored if `cut_off=np.inf` to indicate
        a dense distance matrix, where all distance pairs are computed.

    **backend_options
        Keyword arguments handled to selected backend.

    Returns
    -------
    Union[numpy.ndarray, scipy.sparse.csr_matrix]
        distance matrix of shape `(n_samples_X, n_samples_X)` if `Y=None`, \
        else of shape `(n_samples_Y, n_samples_X)`
    """
    backend_class = get_backend_distance_algorithm(backend)

    backend_init_args = inspect.signature(backend_class).parameters.keys()

    kwargs = {
        "metric": metric,
        "cut_off": cut_off,
        "kmin": k,
        "backend_options": backend_options,
    }
    keys = deepcopy(list(kwargs.keys()))

    for a in keys:
        if a not in backend_init_args:
            kwargs.pop(a)

    backend_options = kwargs.pop("backend_options", {})
    return backend_class(**kwargs, **backend_options)


def compute_distance_matrix(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: str = "euclidean",
    cut_off: Optional[float] = None,
    k: Optional[int] = None,
    backend: Union[str, type[DistanceAlgorithm]] = "guess_optimal",
    **backend_kwargs,
) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """Compute distance matrix with different settings and backends.

    Parameters
    ----------
    X
        Point cloud of shape `(n_samples_X, n_features_X)`.

    Y
        Reference point cloud for component-wise computation of shape \
        `(n_samples_Y, n_features_Y)`. If not given, then `Y=X` (pairwise computation)

    metric
        Distance metric. Needs to be supported by backend.

    cut_off
        Distances larger than `cut_off` are set to zero. The parameter controls the
        degree of sparsity in the distance matrix.

        .. note::
            The pseudo-metric "sqeuclidean" is handled differently in a way that the
            cut-off must be stated in in Eucledian distance (not squared cut-off).

    k
        Minimum number of neighbors per point. Ignored if `cut_off=np.inf` to indicate
        a dense distance matrix, where all distance pairs are computed.

    backend
        Backend to compute distance matrix.

    **backend_kwargs
        Keyword arguments handled to selected backend.

    Returns
    -------
    Union[numpy.ndarray, scipy.sparse.csr_matrix]
        distance matrix of shape `(n_samples_X, n_samples_X)` if `Y=None`, \
        else of shape `(n_samples_Y, n_samples_X)`
    """
    backend_class = init_distance_algorithm(
        backend, metric, cut_off, k, **backend_kwargs
    )
    return backend_class(X, Y)


if __name__ == "__main__":
    print(_all_available_distance_algorithm())
