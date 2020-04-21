#!/usr/bin/env python3

import abc
import logging
from typing import Optional, Sequence, Type, Union

import numexpr as ne
import numpy as np
import scipy.sparse
import scipy.spatial
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from datafold.decorators import warn_experimental_function
from datafold.utils.general import if1dim_colvec

try:
    # rdist is an optional distance algorithm backend -- an import error is raised only
    # when one attempts to use rdist and the import was not successful
    import rdist

    IS_IMPORTED_RDIST = True
except ImportError:
    rdist = None
    IS_IMPORTED_RDIST = False

logger = logging.getLogger(__name__)


class DistanceAlgorithm(metaclass=abc.ABCMeta):
    """Abstract class to warp or implement distance matrix algorithms (dense or sparse).

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
        """Abstract method: Computes the distance matrix pair-wise from the dataset.

        From a pairwise computation follow the matrix properties:

        * square
        * diagonal contains distance to itself and is therefore zero
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
        """Abstract method: Computes the distance matrix component-wise between two
        data sets.

        From a component-wise distance computation follow the general matrix properties:

        * rectangular matrix with shape (n_samples_Y, n_samples_X)
        * outlier points can lead to zero columns / rows
        """


class BruteForceNumexpr(DistanceAlgorithm):
    """
    Source of algorithm:
    https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python
    """

    backend_name = None  #  TODO: algorithm not used at the moment

    def __init__(self, metric):
        if metric not in ["euclidean", "sqeuclidean"]:
            raise ValueError

        super(BruteForceNumexpr, self).__init__(metric=metric)

    def _eval_numexpr_dist(self, A, B, C):

        distance_matrix = ne.evaluate(
            f"A + B - 2. * C",
            {"A": A, "B": B, "C": C,},
            optimization="aggressive",
            order="C",
        )

        # For some reason actual zero values can be slightly negative (in range of
        # numerical noise ~1e-14) --> this then results into nan values when applying
        # the sqrt() function for the euclidean
        distance_matrix[distance_matrix < 0] = 0
        return distance_matrix

    def pdist(
        self, X: np.ndarray, cut_off: Optional[float] = None, **backend_options
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:

        X_norm = ne.evaluate("sum(X ** 2, axis=1)", {"X": X})
        # X_norm = np.linalg.norm(X, axis=1)
        # X_norm = np.square(X_norm, X_norm)

        distance_matrix = self._eval_numexpr_dist(
            A=X_norm[np.newaxis, :], B=X_norm[:, np.newaxis], C=np.dot(X, X.T),
        )

        if self.metric == "euclidean":
            distance_matrix = np.sqrt(distance_matrix, out=distance_matrix)

        # Somehow, zero valuess are often imprecise,
        # For these zero values are easy to set:
        np.fill_diagonal(distance_matrix, 0)

        # Brute force algorithms can only sparsify the distance matrix, after everything
        # is computed:
        if cut_off is not None:
            distance_matrix = self._dense2csr_matrix(distance_matrix, cut_off)

        return distance_matrix

    def cdist(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cut_off: Optional[float] = None,
        **backend_options,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:

        X_norm = ne.evaluate("sum(X ** 2, axis=1)", {"X": X})
        Y_norm = ne.evaluate("sum(Y ** 2, axis=1)", {"Y": Y})

        distance_matrix = self._eval_numexpr_dist(
            A=Y_norm[:, np.newaxis], B=X_norm[np.newaxis, :], C=np.dot(Y, X.T)
        )

        if self.metric == "euclidean":
            distance_matrix = np.sqrt(distance_matrix, out=distance_matrix)

        if cut_off is not None:
            distance_matrix = self._dense2csr_matrix(distance_matrix, cut_off)

        return distance_matrix


class BruteForceDist(DistanceAlgorithm):
    """Computes the full distance matrix.

    Chooses either

        * `scipy.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>_` \
           and
           `scipy.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_
        * `sklearn.pairwise_distances <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html>`_

        Depending on the parameter `exact_numeric`.

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
            point cloud with shape `(n_samples, n_features)`

        cut_off
            distances larger than `cut_off` are set to zero

            .. note::
                Distances with larger distance are removed after they memory was
                allocated and distances computed. It is recommended to use other
                distance algorithms that directly integrates cut off sparsity .

        exact_numeric
            If False, computes (also depending from metric) distances more efficiently
            at the cost of introducing numerical noise. (empirically: `~1e-14` for
            "sqeuclidean" and `~1e-7` for "euclidean" metric).
            
        **backend_options
            Keyword arguments handled to the executing backend.

        Returns
        -------
        Union[numpy.ndarray, scipy.sparse.csr_matrix]
            distance matrix with shape `(n_samples, n_samples)`
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

        For not documented parameters look in ``pdist``

        Parameters
        ----------
        X
            point cloud with shape `(n_samples_X, n_features_X)`

        Y
            point cloud with shape `(n_samples_Y, n_features_Y)`

        Returns
        -------
        Union[numpy.ndarray, scipy.sparse.csr_matrix]
            distance matrix with shape `(n_samples_Y, n_samples_X)`
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
    """Sparse distance matrix algorithm rdist, targeting point clouds with
    manifold assumption.

    The dependency on Python package "rdist" is optional.

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
        return {"strict_distance": True, "max_incr_radius": 0, "kmin": 0}

    def pdist(
        self, X: np.ndarray, cut_off: Optional[float] = None, **backend_options
    ) -> scipy.sparse.csr_matrix:
        """Pair-wise distance matrix computation.

        Parameters
        ----------
        X
            point cloud with shape `(n_samples, n_features)`

        cut_off
            distances (always Euclidean) larger than `cut_off` are set to zero

        **backend_options
            keywords handled to build

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix with shape `(n_samples, n_samples)`
        """

        max_distance = self._numeric_cut_off(cut_off)
        max_distance = self._adapt_correct_metric_max_distance(max_distance)

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

        For not documented parameters look in ``pdist``.

        Parameters
        ----------
        X
            point cloud with shape `(n_samples_X, n_features_X)`

        Y
            point cloud with shape `(n_samples_Y, n_features_Y)`

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix with shape `(n_samples_Y, n_samples_X)`
        """

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
    """Sparse distance matrix computation using scipy's KD-tree implementation.

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
            point cloud with shape `(n_samples, n_features)`

        cut_off
            larger distances (in Euclidean metric) are set to zero

        **backend_options
            key word arguments handled to `cKDTree`

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix with shape `(n_samples, n_samples)`
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

        For not documented parameters look in ``pdist``.

        Parameters
        ----------
        X
            point cloud with shape `(n_samples_X, n_features_X)`

        Y
            point cloud with shape `(n_samples_Y, n_features_Y)`

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix with shape `(n_samples_Y, n_samples_X)`
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
    """Ball tree implementation from scikit-learn.

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
            point cloud with shape `(n_samples, n_features)`

        cut_off
            larger distances are set to zero (see
            :class:`sklearn.neighbors.NearestNeighbors` documentation)

        **backend_options
            handled to `NearestNeighbor`

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix with shape `(n_samples, n_samples)`
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

        For not documented parameters look in ``pdist``.

        Parameters
        ----------
        X
            point cloud with shape `(n_samples_X, n_features_X)`

        Y
            point cloud with shape `(n_samples_Y, n_features_Y)`

        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix with shape `(n_samples_Y, n_samples_X)`
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
            point cloud with shape `(n_samples, n_features)`

        cut_off
            larger distances are set to zero

        **backend_options
            keyword arguments handled to guessed optimal :meth:`DistanceAlgorithm.pdist`
        
        Returns
        -------
        scipy.sparse.csr_matrix
            distance matrix with shape `(n_samples, n_samples)`
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

        For not documented parameters look in ``pdist``.

        Parameters
        ----------
        X
            point cloud with shape `(n_samples_X, n_features_X)`

        Y
            point cloud with shape `(n_samples_Y, n_features_Y)`

        Returns
        -------
        Union[numpy.ndarray, scipy.sparse.csr_matrix]
            distance matrix with shape `(n_samples_Y, n_samples_X)`
        """
        return self._guess_optimal_backend(cut_off).cdist(
            X, Y, cut_off, **backend_options
        )


def _k_smallest_element_value(
    distance_matrix, k: int, ignore_zeros: bool = True, fill_value: float = 0.0
):
    """Compute the k-th smallest element of distance matrix, i.e. the element where only
    k-1 elements are smaller. If ignore_zeros=True only positive distances are considered.
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


@warn_experimental_function
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
    """Selects and validates the backend class to compute a distance matrix.

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
    tol: float = 1.0,
    backend: Union[str, Type[DistanceAlgorithm]] = "brute",
    **backend_kwargs,
) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """Function to compute distance matrix with different settings and backends.

    Parameters
    ----------

    X
        point cloud with shape `(n_samples_X, n_features_X)`

    Y
        reference point cloud for component-wise computation with shape \
        `(n_samples_Y, n_features_Y)`
    
    metric
        distance metric (needs to be supported by backend)

    cut_off
        Distances larger than `cut_off` are set to zero and controls the degree of
        sparsity in the distance matrix.

        .. note::
            The pseudo-metric "sqeuclidean" is handled differently in a way that the
            cut off must be stated in in Eucledian distance (not squared cut off).

    kmin
        input for continuous nearest neighbor

    tol
        tolerance used in continuous nearest neighbors

    backend
        backend to compute distance matrix

    **backend_kwargs
        keyword agruments handled to selected backend

    Returns
    -------
    Union[numpy.ndarray, scipy.sparse.csr_matrix]
        distance matrix with shape `(n_samples_X, n_samples_X)` if `Y=None`, \
        else with shape `(n_samples_Y, n_samples_X)`
    """

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    X = if1dim_colvec(X)

    if X.shape[0] <= 1:
        raise ValueError(
            f"number of samples has to be greater than 1. Got {X.shape[0]}"
        )

    logger.info("Setting up computation of distance matrix.")

    if cut_off is not None and np.isinf(cut_off):
        # use dense case if cut_off is infinite
        cut_off = None

    is_pdist = Y is None
    is_sparse = cut_off is not None

    if not is_sparse and kmin > 0:
        # TODO: (raise error early, before expensive distance matrix is computed)
        raise NotImplementedError(
            "fix_kmin_rows currently only works for sparse distance matrices (i.e. with "
            "set cut_off)."
        )

    if metric == "sqeuclidean":
        if cut_off is not None:
            # NOTE: this is a special case. Usually the cut_off is represented in the
            # respective metric. However, for the 'sqeuclidean' case we use the
            # 'euclidean' metric for the cut off.
            cut_off = cut_off ** 2

    backend_class = get_backend_distance_algorithm(backend)
    distance_method = backend_class(metric=metric)

    logger.info(f"Start computing distance matrix.")

    if is_pdist:
        distance_matrix = distance_method.pdist(X, cut_off, **backend_kwargs)
    else:  # cdist
        distance_matrix = distance_method.cdist(X, Y, cut_off, **backend_kwargs)

    if scipy.sparse.issparse(distance_matrix) and cut_off is None:
        # dense case stored in a sparse distance matrix -> convert to np.ndarray
        distance_matrix = distance_matrix.toarray()

    if is_sparse and not scipy.sparse.issparse(distance_matrix):
        raise RuntimeError(
            "Distance_matrix is expected to be sparse but returned "
            "distance matrix is dense. Please report bug. "
        )

    if is_sparse and distance_matrix.nnz == np.product(distance_matrix.shape):
        logger.info(
            f"cut_off={cut_off} value has no effect on sparsity of distance matrix, "
            f"the sparse matrix is effectively dense."
        )

    if kmin > 0:
        logger.info("apply continuous nearest neighbor on distance matrix")
        distance_matrix = apply_continuous_nearest_neighbor(distance_matrix, kmin, tol)

    if scipy.sparse.issparse(distance_matrix):
        if not isinstance(distance_matrix, scipy.sparse.csr_matrix):
            # For now only return CSR format.
            distance_matrix = distance_matrix.tocsr()

        # Sort_indices return immediately if indices are already sorted.
        # If not sorted, the call can be costly (depending on nnz), but is better for
        # later usage
        distance_matrix.sort_indices()

    return distance_matrix


if __name__ == "__main__":
    print(_all_available_distance_algorithm())
