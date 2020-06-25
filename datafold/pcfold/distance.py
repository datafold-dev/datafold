#!/usr/bin/env python3

import abc
import warnings
from typing import Optional, Sequence, Type, Union

import numexpr as ne
import numpy as np
import scipy.sparse
import scipy.spatial
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree, NearestNeighbors

from datafold.decorators import warn_experimental_function
from datafold.utils.general import if1dim_colvec, if1dim_rowvec, is_symmetric_matrix

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
    * A sparse distance matrix with a distance cut-off value does not store distance
      values *above* the cut-off. Importantly, this means that the sparse matrix
      **must** store real distance zeros (duplicates or self-distances in case of
      `pdist`) and treat the zeros not stored in the sparse matrix as "distance
      values out of range".

    Parameters
    ----------
    metric
        distance metric to compute
    """

    # distances cannot be negative, therefore choose an easy to identify negative value
    _invalid_dist_value = -999

    def __init__(self, metric):
        try:
            getattr(self, "backend_name")
        except AttributeError:
            raise NotImplementedError(
                f"Attribute 'backend_name' is missing in subclass {type(self)}."
            )

        self.metric = metric

    def _check_valid_metric(self, valid, metric):
        if metric not in valid:
            raise ValueError(
                f"Distance algorithm has invalid metric = {metric}. Valid metrics "
                f"are = {valid}."
            )

    def _numeric_cut_off(self, cut_off):
        if cut_off is not None:
            return cut_off
        else:
            return np.inf

    def _set_zeros_sparse_diagonal(self, distance_matrix):
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

    @abc.abstractmethod
    def pdist(
        self, X: np.ndarray, cut_off: Optional[float] = None, **backend_options
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Abstract method to compute the distance matrix pair-wise from the dataset.

        In a pair-wise computation the query and reference data are the same. From this
        the following distance matrix properties follow:

        * square
        * diagonal contains distance to itself and are therefore zero
        * symmetric
        """

    @abc.abstractmethod
    def cdist(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cut_off: Optional[float] = None,
        **backend_options,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Abstract method to compute the distance matrix component-wise.

        The dataset `X` refers to the reference dataset, and the distances are
        computed component-wise for the query dataset `Y`. From this the general
        matrix properties follow:

        * rectangular matrix of shape `(n_samples_Y, n_samples_X)`
        * outlier points can lead to zero columns / rows
        * only duplicated between `X` and `Y` have zero entries
        """


class BruteForceDist(DistanceAlgorithm):
    """Computes all distance pairs in the distance matrix.

    Chooses either

        * `scipy.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_ \
           and
           `scipy.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_
        * `sklearn.pairwise_distances <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html>`_

        Depending on the parameter `metric` and argument `exact_numeric`.

        For an explanation of how `exact_numeric = False` is beneficial, see the
        `scikit-learn` `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html>`_

    Parameters
    ----------
    metric
        Metric to compute, see documentation of backend algorithms what metrics for
        supported options.
    """

    backend_name = "brute"

    def __init__(self, metric: str):
        super(BruteForceDist, self).__init__(metric=metric)

    def pdist(
        self,
        X: np.ndarray,
        cut_off: Optional[float] = None,
        exact_numeric: bool = True,
        **backend_options,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Pair-wise distance matrix computation.
        
        Parameters
        ----------
        X
            Point cloud of shape `(n_samples, n_features)`.

        cut_off
            distances larger than `cut_off` are set to zero

            .. note::
                Distances with larger distance are removed after a full memory
                allocation of the distance matrix. It is recommended to use
                distance algorithms that directly integrate a cut-off sparsity.

        exact_numeric
            If False, computes Euclidean distances more efficiently
            at the cost of introducing numerical noise. Empirically `~1e-14` for
            "sqeuclidean" and `~1e-7` for "euclidean" metric.
            
        **backend_options
            Keyword arguments handled to the executing backend.

        Returns
        -------
        Union[numpy.ndarray, scipy.sparse.csr_matrix]
            distance matrix of shape `(n_samples, n_samples)`
        """
        if exact_numeric:
            X = np.array(X)
            _pdist = pdist(X, metric=self.metric)
            distance_matrix = squareform(_pdist)
        else:
            # sklearn uses an numeric inexact but faster implementation
            distance_matrix = pairwise_distances(X, metric=self.metric)

        if cut_off is not None:
            distance_matrix = self._dense2csr_matrix(distance_matrix, cut_off=cut_off)

        return distance_matrix

    def cdist(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cut_off: Optional[float] = None,
        exact_numeric: bool = True,
        **backend_options,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Component-wise distance matrix computation.

        For undocumented parameters look at :meth:`.pdist`.

        Parameters
        ----------
        X
            Point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Point cloud of shape `(n_samples_Y, n_features_Y)`.

        Returns
        -------
        Union[numpy.ndarray, scipy.sparse.csr_matrix]
            distance matrix of shape `(n_samples_Y, n_samples_X)`
        """

        if exact_numeric:
            distance_matrix = cdist(Y, X, metric=self.metric, **backend_options)
        else:
            distance_matrix = pairwise_distances(
                Y, X, metric=self.metric, **backend_options
            )

        if cut_off is not None:
            # remove distances > cut_off and convert to sparse matrix
            distance_matrix = self._dense2csr_matrix(distance_matrix, cut_off=cut_off)

        return distance_matrix


class RDist(DistanceAlgorithm):
    """Sparse distance matrix algorithm rdist, for point clouds with manifold assumption.

    .. note::
        The dependency on the Python package is optional. The package is currentl not
        published.

    Parameters
    ----------

    metric
        "euclidean" or "sqeuclidean"

    Raises
    ------

    ImportError
        if rdist is not installed, but selected as backend

    References
    ----------

    .. todo::
        include own paper if published

    """

    backend_name = "rdist" if IS_IMPORTED_RDIST else None

    def __init__(self, metric):
        if not IS_IMPORTED_RDIST:
            raise ImportError("Could not import rdist. Check if it is installed.")

        self._check_valid_metric(valid=["euclidean", "sqeuclidean"], metric=metric)
        super(RDist, self).__init__(metric=metric)

    def _adapt_correct_metric_max_distance(self, max_distance):
        # Generally: the cut-off is represented like self.metric. The scipy.kdtree can
        # only compute Euclidean distances. Therefore, undo the squaring of cut-off.
        # For sqeuclidean distance, the squaring has to be done after the
        # distance matrix was computed.

        if self.metric == "sqeuclidean":
            max_distance = np.sqrt(
                max_distance
            )  # note if max_distance==np.inf, the it is still np.inf

        return max_distance

    def _get_dist_options(self):
        return {"max_incr_radius": 0, "kmin": 0}

    def pdist(
        self, X: np.ndarray, cut_off: Optional[float] = None, **backend_options
    ) -> scipy.sparse.csr_matrix:
        """Pair-wise distance matrix computation.

        Parameters
        ----------
        X
            Point cloud of shape `(n_samples, n_features)`.

        cut_off
            Distance values (always Euclidean metric) that are larger are not stored in
            distance matrix.

        **backend_options
            keywords handled to build

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix of shape `(n_samples, n_samples)`
        """

        max_distance = self._numeric_cut_off(cut_off)
        max_distance = self._adapt_correct_metric_max_distance(max_distance)

        assert rdist is not None

        # build tree, currently not stored, backend options are handled to here.
        _rdist = rdist.Rdist(X, **backend_options)

        # compute distance matrix, these options are not accessible from outside at the
        # moment.
        distance_matrix = _rdist.sparse_pdist(
            r=max_distance, rtype="radius", **self._get_dist_options()
        )

        if self.metric == "euclidean":
            distance_matrix.data = np.sqrt(distance_matrix.data)

        return distance_matrix

    def cdist(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cut_off: Optional[float] = None,
        **backend_options,
    ) -> scipy.sparse.csr_matrix:
        """Component-wise distance matrix computation.

        For undocumented parameters look at :meth:`.pdist`.

        Parameters
        ----------
        X
            Point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Point cloud of shape `(n_samples_Y, n_features_Y)`.

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix of shape `(n_samples_Y, n_samples_X)`
        """

        assert rdist is not None

        max_distance = self._numeric_cut_off(cut_off)
        max_distance = self._adapt_correct_metric_max_distance(max_distance)

        _rdist = rdist.Rdist(X, **backend_options)
        distance_matrix = _rdist.sparse_cdist(
            req_points=Y, r=max_distance, rtype="radius", **self._get_dist_options()
        )

        if self.metric == "euclidean":
            distance_matrix.data = np.sqrt(distance_matrix.data)

        return distance_matrix


class ScipyKdTreeDist(DistanceAlgorithm):
    """Sparse distance matrix computation using scipy's kd-tree implementation.

    Parameters
    ----------
    metric
        "euclidean" or "sqeuclidean"
        
    References
    ----------

    :class:`scipy.spatial.cKDTree`
    :meth:`scipy.spatial.KDTree.sparse_distance_matrix`

    """

    backend_name = "scipy.kdtree"

    def __init__(self, metric):
        self._check_valid_metric(valid=["euclidean", "sqeuclidean"], metric=metric)
        super(ScipyKdTreeDist, self).__init__(metric=metric)

    def _adapt_correct_metric_max_distance(self, max_distance):
        # Generally: the cut-off is represented like self.metric. The scipy.kdtree can
        # only compute Euclidean distances. Therefore, undo the squaring of cut-off.
        # For sqeuclidean distance, the squaring has to be done after the
        # distance matrix was computed.

        if self.metric == "sqeuclidean":
            # note if max_distance==np.inf, the it is still np.inf
            max_distance = np.sqrt(max_distance)

        return max_distance

    def pdist(
        self, X: np.ndarray, cut_off: Optional[float] = None, **backend_options
    ) -> scipy.sparse.csr_matrix:
        """Pair-wise distance computation.

        Parameters
        ----------
        X
            Point cloud of shape `(n_samples, n_features)`.

        cut_off
            Distance values (always Euclidean metric) that are larger are not stored in
            distance matrix.

        **backend_options
            key word arguments handled to `cKDTree`

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix of shape `(n_samples, n_samples)`
        """

        # TODO: if necessary there are build_options and compute_options required,
        #  currently no options to sparse_distance_matrix are handed
        max_distance = self._numeric_cut_off(cut_off)
        max_distance = self._adapt_correct_metric_max_distance(max_distance)

        kdtree = scipy.spatial.cKDTree(X, **backend_options)
        dist_matrix = kdtree.sparse_distance_matrix(
            kdtree, max_distance=max_distance, output_type="coo_matrix"
        )

        if self.metric == "sqeuclidean":
            dist_matrix.data = np.square(dist_matrix.data)

        return dist_matrix.tocsr()

    def cdist(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cut_off: Optional[float] = None,
        **backend_options,
    ) -> scipy.sparse.csr_matrix:
        """Component-wise distance matrix computation.

        Parameters
        ----------
        X
            Point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Point cloud of shape `(n_samples_Y, n_features_Y)`.

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix of shape `(n_samples_Y, n_samples_X)`
        """
        max_distance = self._numeric_cut_off(cut_off)
        max_distance = self._adapt_correct_metric_max_distance(max_distance)

        kdtree_x = scipy.spatial.cKDTree(X, **backend_options)
        kdtree_y = scipy.spatial.cKDTree(Y, **backend_options)

        dist_matrix = kdtree_y.sparse_distance_matrix(
            kdtree_x, max_distance=max_distance, output_type="coo_matrix"
        )

        if self.metric == "sqeuclidean":
            dist_matrix.data = np.square(dist_matrix.data)

        return dist_matrix.tocsr()


class SklearnBalltreeDist(DistanceAlgorithm):
    """Distance matrix using ball tree implementation from scikit-learn.

    Parameters
    ----------
    metric
        see `NeaestNeighors` documentation (reference)

    References
    ----------

    :class:`sklearn.neighbors.NearestNeighbors`
    """

    backend_name = "sklearn.balltree"

    def __init__(self, metric):
        super(SklearnBalltreeDist, self).__init__(metric=metric)

    def _map_metric_and_cut_off(self, cut_off):
        if self.metric == "sqeuclidean":
            if cut_off is not None:
                cut_off = np.sqrt(cut_off)
            return "euclidean", cut_off
        else:
            return self.metric, cut_off

    def pdist(
        self, X: np.ndarray, cut_off: Optional[float] = None, **backend_options
    ) -> scipy.sparse.csr_matrix:
        """Pair-wise distance matrix computation.

        Parameters
        ----------
        X
            Point cloud of shape `(n_samples, n_features)`.

        cut_off
            Distance values (always Euclidean metric) that are larger are not stored in
            distance matrix. (see also :class:`sklearn.neighbors.NearestNeighbors`
            documentation)

        **backend_options
            handled to `NearestNeighbor`

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix of shape `(n_samples, n_samples)`
        """
        metric, cut_off = self._map_metric_and_cut_off(cut_off)

        max_distance = self._numeric_cut_off(cut_off)
        nn = NearestNeighbors(
            radius=max_distance, algorithm="ball_tree", metric=metric, **backend_options
        )
        nn.fit(X)
        distance_matrix = nn.radius_neighbors_graph(mode="distance")

        if self.metric == "sqeuclidean":
            distance_matrix.data = np.square(
                distance_matrix.data, out=distance_matrix.data
            )

        distance_matrix = self._set_zeros_sparse_diagonal(distance_matrix)

        return distance_matrix

    def cdist(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cut_off: Optional[float] = None,
        **backend_options,
    ) -> scipy.sparse.csr_matrix:
        """Component-wise distance matrix computation.

        For undocumented parameters look at :meth:`.pdist`.

        Parameters
        ----------
        X
            Point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Point cloud of shape `(n_samples_Y, n_features_Y)`.

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix of shape `(n_samples_Y, n_samples_X)`
        """
        metric, cut_off = self._map_metric_and_cut_off(cut_off)

        max_distance = self._numeric_cut_off(cut_off)
        nn = NearestNeighbors(
            radius=max_distance, algorithm="ball_tree", metric=metric, **backend_options
        )
        nn.fit(X)
        distance_matrix = nn.radius_neighbors_graph(Y, mode="distance")

        if self.metric == "sqeuclidean":
            distance_matrix.data = np.square(
                distance_matrix.data, out=distance_matrix.data
            )

        return distance_matrix


class GuessOptimalDist(DistanceAlgorithm):
    """Tries to guess a suitable algorithm based on sparsity, metric and installed
    backends algorithms.

    Parameters
    ----------
    metric
        distance metric

    """

    backend_name = "guess_optimal"

    def __init__(self, metric):
        super(GuessOptimalDist, self).__init__(metric=metric)

    def _guess_optimal_backend(self, cut_off):

        if cut_off is None:  # dense case
            backend_str = BruteForceDist.backend_name
        else:
            if IS_IMPORTED_RDIST and self.metric in ["euclidean", "sqeuclidean"]:
                backend_str = RDist.backend_name
                # backend_str = ScipyKdTreeDist.backend_name
                assert backend_str is not None

            elif self.metric in ["euclidean", "sqeuclidean"]:
                backend_str = ScipyKdTreeDist.backend_name
            else:
                backend_str = BruteForceDist.backend_name

        backend_class = get_backend_distance_algorithm(backend_str)
        return backend_class(self.metric)  # initialize and return as object

    def pdist(
        self, X: np.ndarray, cut_off: Optional[float] = None, **backend_options
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Pair-wise distance matrix computation.

        Parameters
        ----------
        X
            Point cloud of shape `(n_samples, n_features)`.

        cut_off
            Distance values (always Euclidean metric) that are larger are not stored in
            distance matrix.

        **backend_options
            Keyword arguments passed to :meth:`DistanceAlgorithm.pdist`
        
        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix of shape `(n_samples, n_samples)`
        """
        return self._guess_optimal_backend(cut_off).pdist(X, cut_off, **backend_options)

    def cdist(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cut_off: Optional[float] = None,
        **backend_options,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """Component-wise distance matrix computation.

        For undocumented parameters look at :meth:`.pdist`.

        Parameters
        ----------
        X
            Point cloud of shape `(n_samples_X, n_features_X)`.

        Y
            Point cloud of shape `(n_samples_Y, n_features_Y)`.

        Returns
        -------
        Union[numpy.ndarray, scipy.sparse.csr_matrix]
            distance matrix of shape `(n_samples_Y, n_samples_X)`
        """
        return self._guess_optimal_backend(cut_off).cdist(
            X, Y, cut_off, **backend_options
        )


def _k_smallest_element_value(
    distance_matrix, k: int, ignore_zeros: bool = True, fill_value: float = 0.0
):
    """Compute the k-th smallest element of distance matrix, i.e. the element where only
    k-1 elements are smaller. If `ignore_zeros=True` only positive distances are
    considered.
    """

    if k > distance_matrix.shape[1] or k < 0:
        raise ValueError(
            f"ValueError: kth(={k} out of bounds ({distance_matrix.shape[1]})"
        )

    if scipy.sparse.issparse(distance_matrix):
        k_smallest_values = np.zeros(distance_matrix.shape[0])

        # TODO: This loop is likely slow, improve speed if required
        for row_idx in range(distance_matrix.shape[0]):
            row = distance_matrix.getrow(row_idx).data

            if ignore_zeros:
                # there could still be stored zeros (e.g. on the diagonal of a pdist
                # matrix)
                row = row[row != 0]

                if row.shape[0] <= k:
                    k_smallest_values[row_idx] = fill_value
                else:
                    k_smallest_values[row_idx] = np.partition(row, k)[k]
            else:
                nr_not_stored_zeros = distance_matrix.shape[1] - row.shape[0]
                if k <= nr_not_stored_zeros:
                    k_smallest_values[row_idx] = 0
                else:
                    # if there are still zeros, can still be stored
                    k_smallest_values[row_idx] = np.partition(row, k)[k]
    else:  # dense case
        if ignore_zeros:
            assert not np.isinf(distance_matrix).any()

            # set zeros to inf such that they are ignored in np.partition
            distance_matrix[distance_matrix == 0] = np.inf

            k_smallest_values = np.partition(distance_matrix, k, axis=1)[:, k]
            k_smallest_values[np.isinf(k_smallest_values)] = fill_value

            # set inf values back to zero
            distance_matrix[np.isinf(distance_matrix)] = 0
        else:
            k_smallest_values = np.partition(distance_matrix, k, axis=1)[:, k]

    return k_smallest_values


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

        _ball_tree = BallTree(X, leaf_size=40, metric=metric)
        distances, columns_indices = _ball_tree.query(
            Y[knn_query_indices, :],
            k=kmin,
            return_distance=True,
            dualtree=False,
            breadth_first=False,
            sort_results=False,
        )

        distances = np.reshape(
            distances, newshape=np.product(distances.shape), order="C"
        )

        # Note: duplicates and trivial self-distances in the pdist are assumed to already
        # covered by the DistanceAlgorithm (always contained in the radius!)
        nnz_distance_mask = (distances != 0).astype(np.bool)
        distances = distances[nnz_distance_mask]

        knn_query_indices = np.repeat(knn_query_indices, kmin)[nnz_distance_mask]

        columns_indices = np.reshape(
            columns_indices, newshape=np.product(columns_indices.shape), order="C"
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
        #  how to do make it right. For this attempt the tests fail:
        #  distance_matrix.tolil(copy=False)[
        #      kmin_elements_csr.nonzero()
        #  ] = kmin_elements_csr.data
        #  maybe the best is to combine the elements of kmin_elements_csr and distance
        #  matrix into one set (sorting out the upper triangle for pdist) and then
        #  create a new sparse matrix...
        distance_matrix[kmin_elements_csr.nonzero()] = kmin_elements_csr.data

    return distance_matrix.tocsr()


def _all_available_distance_algorithm():
    """Searches for valid subclasses of :py:class:`DistanceAlgorithm`

    Returns
    -------
    list
        all valid subclasses
    """

    all_backends: Sequence[type] = DistanceAlgorithm.__subclasses__()

    return_backends = list()
    # This is the case if backend is given as a str, now we look for the matching
    # DistanceAlgorithm.backend_name
    for b in all_backends:
        # Tests are only for security that all algorithms have the mandatory (static)
        # attribute set.

        try:
            # Attribute error is raised here if 'backend_name' does not exist
            # If 'backend_name' is set to none, then the implementation is not
            # considered e.g. because dependencies are not met in case of rdist.
            if isinstance(b.backend_name, str):
                return_backends.append(b)
        except AttributeError or AssertionError:
            raise NotImplementedError(
                f"Bug: class {type(b)} has no 'backend_name' attribute or it is not of "
                f"type 'str'. Check implementation."
            )

    return return_backends


@DeprecationWarning
def apply_continuous_nearest_neighbor(distance_matrix, kmin, tol):

    if tol == 0:
        # TODO: check if what are valid tol values and what the tolerance is exactly
        #  used for...
        raise ZeroDivisionError("tol cannot be zero.")

    k_smallest_element_values = _k_smallest_element_value(
        distance_matrix, kmin, ignore_zeros=True, fill_value=1 / tol
    )
    xk = np.reciprocal(k_smallest_element_values)

    epsilon = 0.25  # TODO: magic number, parametrize?

    if scipy.sparse.issparse(distance_matrix):

        xk_inv_sp = scipy.sparse.dia_matrix((xk, 0), (xk.shape[0], xk.shape[0]))
        distance_matrix.data = np.square(distance_matrix.data)
        distance_matrix = xk_inv_sp @ distance_matrix @ xk_inv_sp
        distance_matrix.data = np.sqrt(distance_matrix.data)

        # TODO: 4 is magic number, and the product is currently always 1
        distance_matrix.data[distance_matrix.data > 4 * epsilon] = 0
        # TODO: maybe for pdist matrices need to set the diagonal with zeros again
        distance_matrix.eliminate_zeros()
    else:  # dense case
        xk_inv = np.diag(xk)
        distance_matrix = np.square(distance_matrix)
        distance_matrix = xk_inv @ distance_matrix @ xk_inv
        distance_matrix = np.sqrt(distance_matrix)
        distance_matrix[distance_matrix > 4 * epsilon] = 0  # TODO: see above

    return distance_matrix


def get_backend_distance_algorithm(backend):
    """Selects and validates the backend class for distance matrix computation.

    Parameters
    ----------
    backend
        * ``str`` - maps to the algorithms
        * ``DistanceAlgorithm`` - returns same object if valid

    Returns
    -------
    DistanceAlgorithm
    """

    if backend is None:
        raise ValueError("backend cannot be None")

    all_backends = _all_available_distance_algorithm()

    # This is the case if a user chooses backend by object instead of "backend_name"
    # attribute.
    if backend in all_backends:
        return backend

    for b in all_backends:
        # look up for the backend algorithm with the name implemented
        if b.backend_name == backend:
            return b
    else:
        raise ValueError(f"Could not find backend {backend}")


def compute_distance_matrix(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: str = "euclidean",
    cut_off: Optional[float] = None,
    kmin: int = 0,
    backend: Union[str, Type[DistanceAlgorithm]] = "brute",
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
        Distances larger than `cut_off` are set to zero and controls the degree of
        sparsity in the distance matrix.

        .. note::
            The pseudo-metric "sqeuclidean" is handled differently in a way that the
            cut-off must be stated in in Eucledian distance (not squared cut-off).

    kmin
        Minimum number of neighbors. Ignored if `cut_off=np.inf` to indicate a dense
        distance matrix, where all distance pairs are computed.

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

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    X = if1dim_colvec(X)

    if Y is not None and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    if Y is not None:
        Y = if1dim_rowvec(Y)

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "mismatch of point dimension: "
                f"X.shape[1]={X.shape[1]} != Y.shape[1]={Y.shape[1]} "
            )

    if X.shape[0] <= 1:
        raise ValueError(
            f"number of samples has to be greater than 1. Got {X.shape[0]}"
        )

    if cut_off is not None:
        if cut_off <= 0:
            raise ValueError(f"cut_off={cut_off} must be a positive float.")

        try:
            cut_off = float(cut_off)  # make sure to only deal with Python built-in
        except:
            raise TypeError(f"type(cut_off)={type(cut_off)} must be of type float")

        if np.isinf(cut_off):
            # use dense case if cut_off is infinite
            cut_off = None

    is_pdist = Y is None
    is_sparse = cut_off is not None

    if metric == "sqeuclidean":
        if cut_off is not None:
            # NOTE: this is a special case. Usually the cut_off is represented in the
            # respective metric. However, for the 'sqeuclidean' case we use the
            # 'euclidean' metric for the cut-off.
            cut_off = cut_off ** 2

    backend_class = get_backend_distance_algorithm(backend)
    distance_method = backend_class(metric=metric)

    if is_pdist:
        distance_matrix = distance_method.pdist(X, cut_off, **backend_kwargs)
    else:  # cdist
        distance_matrix = distance_method.cdist(X, Y, cut_off, **backend_kwargs)

    if scipy.sparse.issparse(distance_matrix) and cut_off is None:
        # dense case stored in a sparse distance matrix -> convert to np.ndarray
        distance_matrix = distance_matrix.toarray()

    if is_sparse:

        if not scipy.sparse.issparse(distance_matrix):
            raise RuntimeError(
                "Distance_matrix is expected to be sparse but DistanceAlgorithm "
                f"{backend} returned dense matrix. Please report bug."
            )

        if not isinstance(distance_matrix, scipy.sparse.csr_matrix):
            # Currently, we only return a sparse matrix in CSR format.
            distance_matrix = distance_matrix.tocsr()

        # only for the sparse case we care about kmin:
        if (kmin > 0 and not is_pdist) or (kmin > 1 and is_pdist):
            # kmin == 1 and is_pdist does not need treatment because the diagonal is set.
            distance_matrix = _ensure_kmin_nearest_neighbor(
                X, Y, metric=metric, kmin=kmin, distance_matrix=distance_matrix,
            )

        # sort_indices returns immediately if indices are already sorted.
        # If not sorted, the call could be costly (depending on nnz), but is better for
        # follow-up handling.
        distance_matrix.sort_indices()

        n_elements_stored = (
            distance_matrix.nnz
            + len(distance_matrix.indptr)
            + len(distance_matrix.indices)
        )

        # There are also other reasons than memory savings for sparse matrices --
        # therfore the warning is comment out for now.

        # if n_elements_stored > np.product(distance_matrix.shape):
        #     warnings.warn(
        #         f"cut_off={cut_off} value does not lead to reduced memory requirements "
        #         f"with sparse matrix. The sparse matrix stores {n_elements_stored} "
        #         f"which exceeds a dense matrix by "
        #         f"{n_elements_stored - np.product(distance_matrix.shape)} elements."
        #     )

    return distance_matrix


if __name__ == "__main__":
    print(_all_available_distance_algorithm())
