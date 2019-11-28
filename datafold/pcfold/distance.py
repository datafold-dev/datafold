#!/usr/bin/env python3

import abc
import logging

import numpy as np
import scipy.sparse
import scipy.spatial
from scipy.spatial.distance import _METRICS, pdist, cdist, squareform
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

# rdist is an optional distance algorithm backend -- an import error is raised only when one attempts to use rdist and
# the import was not successful
try:
    import rdist

    IS_IMPORTED_RDIST = True
except ImportError:
    rdist = None
    IS_IMPORTED_RDIST = False

logger = logging.getLogger(__name__)


def get_k_smallest_element_value(
    distance_matrix, k: int, ignore_zeros=True, fill_value=0
):
    """Return the k-th smallest element, i.e. the element where only k-1 elements are smaller.
    if ignore_zeros=True only positive distances are considered.
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
                # there could still be stored zeros (e.g. on the diagonal of a pdist matrix)
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


class DistanceAlgorithm(metaclass=abc.ABCMeta):
    """Provides same interface for different distance matrix computations."""

    def __init__(self, metric):
        try:
            getattr(self, "NAME")
        except AttributeError:
            raise NotImplementedError(
                f"Attribute NAME is missing in subclass {type(self)}."
            )

        self.metric = metric

    def _check_valid_metric(self, valid, metric):
        if metric not in valid:
            raise ValueError(
                f"Distance algorithm has invalid metric = {metric}. Valid metrics are = {valid}."
            )

    def _numeric_cut_off(self, cut_off):
        if cut_off is not None:
            return cut_off
        else:
            return np.inf

    def _set_zeros_sparse_diagonal(self, distance_matrix):
        # This function sets the diagonal to zero of a sparse matrix.

        # Some algorithms don't store the zeros on the diagonal for the pdist case. However, this is critical if
        # afterwards the kernel is applied kernel(distance_matrix).
        #   -> kernel(distance)=0 but correct is kernel(distance)=1 (for a stationary kernel)
        # The issue is:
        # * We neglect not zeros but large values (e.g. cut_off=100 we ignore larger values and do not store them)
        # * The sparse matrix formats see the "not stored values" equal to zero, however, there are also "true zeros"
        #   for duplicates. We HAVE to store these zero values, otherwise the kernel values are wrong on the
        #   opposite extreme end (i.e. 0 instead of 1, for stationary kernels).

        assert (
            scipy.sparse.issparse(distance_matrix)
            and distance_matrix.shape[0] == distance_matrix.shape[1]
        )

        # distances cannot be negative, therefore choose an easy to identify negative value
        invalid_value = -999.0

        # in case there are duplicate rows -> set to invalid_value
        distance_matrix.data[distance_matrix.data == 0] = invalid_value

        # convert to lil-format, because it is more efficient to set the diag=0
        distance_matrix = distance_matrix.tolil()
        distance_matrix.setdiag(invalid_value)

        # turn back to csr and set the invalid to "true zeros"
        distance_matrix = distance_matrix.tocsr()
        distance_matrix.data[distance_matrix.data == invalid_value] = 0

        return distance_matrix

    @abc.abstractmethod
    def pdist(self, X, cut_off=None, **backend_options):
        """Computes the distance matrix pairwise from the dataset. From this follows always that the matrix is square
        and the diagonal is zero (distance of self points)."""

    @abc.abstractmethod
    def cdist(self, X, Y, cut_off=None, **backend_options):
        """Computes the distance matrix componentwise between two point clouds X and Y. Important:
        the query points are in rows (i) and the reference points in columns (j)."""


class BruteForceDist(DistanceAlgorithm):

    NAME = "brute"

    def __init__(self, metric):
        super(BruteForceDist, self).__init__(metric=metric)

    def pdist(self, X, cut_off=None, **backend_options):
        radius = self._numeric_cut_off(cut_off)

        metric_params = {}

        if self.metric == "mahalanobis":
            # TODO: also allow the user to handle metric_params?
            # TODO: can also compute and handle VI = inverse covariance matrix
            # TODO: sklearn also provides to approximate the covariance for large metrics
            #  https://scikit-learn.org/stable/modules/covariance.html
            metric_params = {
                "V": np.cov(X, rowvar=False)
            }  # add inverse covariance matrix??

        if (
            self.metric in _METRICS.keys()
            and np.isinf(radius)
            # in pdist are no more parameters, and no parallism is supported
            and backend_options == {}
        ):
            distance_matrix = squareform(pdist(X, metric=self.metric))
        else:
            distance_matrix = radius_neighbors_graph(
                X,
                radius=radius,
                mode="distance",
                metric=self.metric,
                metric_params=metric_params,
                include_self=True,
                **backend_options,
            )
            distance_matrix = self._set_zeros_sparse_diagonal(distance_matrix)

        return distance_matrix

    def cdist(self, X, Y, cut_off=None, **backend_options):
        radius = self._numeric_cut_off(cut_off)

        metric_params = {}
        if self.metric == "mahalanobis":
            # TODO: also allow the user to handle metric_params?
            # TODO: can also compute and handle VI = inverse covariance matrix
            # TODO: sklearn also provides to approximate the covariance for large metrics
            #  https://scikit-learn.org/stable/modules/covariance.html
            metric_params["V"] = np.cov(
                X, rowvar=False
            )  # add inverse covariance matrix

        if (
            self.metric in _METRICS.keys()
            and np.isinf(radius)
            and backend_options == {}
        ):
            distance_matrix = cdist(Y, X, metric=self.metric)
        else:
            method = NearestNeighbors(
                radius=radius, algorithm="auto", metric=self.metric, **backend_options
            )

            # Fit to Y first, so that in the rows are the query and reference are in columns
            method = method.fit(X)
            distance_matrix = method.radius_neighbors_graph(Y, mode="distance")

        return distance_matrix


class RDist(DistanceAlgorithm):

    NAME = "rdist" if IS_IMPORTED_RDIST else None

    def __init__(self, metric):
        if not IS_IMPORTED_RDIST:
            raise ImportError("Could not import rdist. Check if it is installed.")

        self._check_valid_metric(valid=["euclidean", "sqeuclidean"], metric=metric)
        super(RDist, self).__init__(metric=metric)

    def _adapt_correct_metric_max_distance(self, max_distance):
        # Generally: the cut-off is represented like self.metric. The scipy.kdtree can only compute Euclidean distances.
        # Therefore, undo the squaring of cut-off. For sqeuclidean distance, the squaring has to be done after the
        # distance matrix was computed.

        if self.metric == "sqeuclidean":
            max_distance = np.sqrt(
                max_distance
            )  # note if max_distance==np.inf, the it is still np.inf

        return max_distance

    def _get_dist_options(self):
        return {"strict_distance": True, "max_incr_radius": 0, "kmin": 0}

    def pdist(self, X, cut_off=None, **backend_options):

        max_distance = self._numeric_cut_off(cut_off)
        max_distance = self._adapt_correct_metric_max_distance(max_distance)

        # build tree, currently not stored, backend options are handled to here.
        _rdist = rdist.Rdist(X, **backend_options)

        # compute distance matrix, these options are not accessible from outside at the moment.
        distance_matrix = _rdist.sparse_pdist(
            r=max_distance, rtype="radius", **self._get_dist_options()
        )

        if self.metric == "euclidean":
            distance_matrix.data = np.sqrt(distance_matrix.data)

        return distance_matrix

    def cdist(self, X, Y, cut_off=None, **backend_options):

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

    NAME = "scipy.kdtree"

    def __init__(self, metric):
        self._check_valid_metric(valid=["euclidean", "sqeuclidean"], metric=metric)
        super(ScipyKdTreeDist, self).__init__(metric=metric)

    def _adapt_correct_metric_max_distance(self, max_distance):
        # Generally: the cut-off is represented like self.metric. The scipy.kdtree can only compute Euclidean distances.
        # Therefore, undo the squaring of cut-off. For sqeuclidean distance, the squaring has to be done after the
        # distance matrix was computed.

        if self.metric == "sqeuclidean":
            max_distance = np.sqrt(
                max_distance
            )  # note if max_distance==np.inf, the it is still np.inf

        return max_distance

    def pdist(self, X, cut_off=None, **backend_options):

        # TODO: if necessary there are build_options and compute_options required, currently no options to
        #  sparse_distance_matrix are handed
        max_distance = self._numeric_cut_off(cut_off)
        max_distance = self._adapt_correct_metric_max_distance(max_distance)

        kdtree = scipy.spatial.cKDTree(X, **backend_options)
        dist_matrix = kdtree.sparse_distance_matrix(
            kdtree, max_distance=max_distance, output_type="coo_matrix"
        )

        if self.metric == "sqeuclidean":
            dist_matrix.data = np.square(dist_matrix.data)

        return dist_matrix.tocsr()

    def cdist(self, X, Y, cut_off=None, **backend_options):

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

    NAME = "sklearn.balltree"

    def __init__(self, metric):
        super(SklearnBalltreeDist, self).__init__(metric=metric)

    def _map_metric_and_cut_off(self, cut_off):
        if self.metric == "sqeuclidean":
            if cut_off is not None:
                cut_off = np.sqrt(cut_off)
            return "euclidean", cut_off
        else:
            return self.metric, cut_off

    def pdist(self, X, cut_off=None, **backend_options):

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

    def cdist(self, X, Y, cut_off=None, **backend_options):

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

    NAME = "guess_optimal"

    def __init__(self, metric):
        super(GuessOptimalDist, self).__init__(metric=metric)

    def _guess_optimal_backend(self, cut_off):

        if cut_off is None:  # dense case
            backend_str = BruteForceDist.NAME
        else:
            if IS_IMPORTED_RDIST and self.metric in ["euclidean", "sqeuclidean"]:
                backend_str = RDist.NAME
                # backend_str = ScipyKdTreeDist.NAME
                assert backend_str is not None

            elif self.metric in ["euclidean", "sqeuclidean"]:
                backend_str = ScipyKdTreeDist.NAME
            else:
                backend_str = BruteForceDist.NAME

        backend_class = get_backend_distance_algorithm(backend_str)
        return backend_class(self.metric)  # initialize and return as object

    def pdist(self, X, cut_off=None, **backend_options):
        return self._guess_optimal_backend(cut_off).pdist(X, cut_off, **backend_options)

    def cdist(self, X, Y, cut_off=None, **backend_options):
        return self._guess_optimal_backend(cut_off).cdist(
            X, Y, cut_off, **backend_options
        )


def apply_continuous_nearest_neighbor(distance_matrix, kmin, tol):

    if tol == 0:
        # TODO: check if what are valid tol values and what the tolerance is exactly used for...
        raise ZeroDivisionError("tol cannot be zero.")

    k_smallest_element_values = get_k_smallest_element_value(
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
        distance_matrix.eliminate_zeros()  # TODO: maybe for pdist matrices need to set the diagonal with zeros again
    else:  # dense case
        xk_inv = np.diag(xk)
        distance_matrix = np.square(distance_matrix)
        distance_matrix = xk_inv @ distance_matrix @ xk_inv
        distance_matrix = np.sqrt(distance_matrix)
        distance_matrix[distance_matrix > 4 * epsilon] = 0  # TODO: see above

    return distance_matrix


def all_available_distance_algorithm():
    all_backends = DistanceAlgorithm.__subclasses__()

    # This is the case if backend is given as a str, now we look for the matching DistanceAlgorithm.NAME
    for b in all_backends:
        # Tests are only for security that all algorithms have the mandatory (static) attribute set.
        try:
            if b.NAME is None:  # Attribute error is raised here if NAME does not exist
                # If NAME is set to none, then the implementation is not considered
                # e.g. because dependencies are not met in case of rdist.
                all_backends.remove(b)
            else:
                assert isinstance(b.NAME, str)
        except AttributeError or AssertionError:
            raise NotImplementedError(
                f"Bug: class {type(b)} has no NAME attribute or it is not of type 'str'. "
                f"Check implementation."
            )

    return all_backends


def get_backend_distance_algorithm(backend):

    if backend is None:
        raise ValueError("backend cannot be None")

    all_backends = all_available_distance_algorithm()

    # This is the case if a user chose the backend by object instead of "NAME"
    if backend in all_backends:
        return backend

    for b in all_backends:
        # look up for the backend algorithm with the name implemented
        if b.NAME == backend:
            return b

    raise ValueError(f"Could not find backend {backend}")


def compute_distance_matrix(
    X,
    Y=None,
    metric="euclidean",
    cut_off=None,
    kmin=0,
    tol=1,
    backend="brute",
    **backend_options,
):
    """
    :param X: np.ndarray - point cloud
    :param Y: np.ndarray [Optional] - point cloud for componentwise (cdist)
    :param metric: str - metric to compute
    :param cut_off: float - cut off for sparsity
    :param kmin: int input for continuous nearest neighbor TODO: check again this parameter
    :param tol: float tolerance used in continuous nearest neighbors TODO: I don't quite get this parameter
    :param backend: str - algorithm to choose
    :param backend_options: dict - handling for specific options to the selected algorithm
    :return: distance matrix
    """

    logger.info("Setting up computation of distance matrix.")

    if cut_off is not None and np.isinf(cut_off):
        # use dense case if cut_off is infinite
        cut_off = None

    is_pdist = Y is None
    is_sparse = cut_off is not None

    if not is_sparse and kmin > 0:
        # TODO: (raise error early, before expensive distance matrix is computed)
        raise NotImplementedError(
            "fix_kmin_rows currently only works for sparse distance matrices (i.e. with set "
            "cut_off)."
        )

    if metric == "sqeuclidean":
        # TODO: discuss this if this is okay
        if cut_off is not None:
            # NOTE: this is a special case. Usually the cut_off is represented in the respective metric. However, for
            # the 'sqeuclidean' case we use the 'euclidean' metric for the cut off.
            cut_off = cut_off ** 2

    backend_class = get_backend_distance_algorithm(backend)
    distance_method = backend_class(metric=metric)

    logger.info(f"Start computing distance matrix.")

    if is_pdist:
        distance_matrix = distance_method.pdist(X, cut_off, **backend_options)
    else:  # cdist
        distance_matrix = distance_method.cdist(X, Y, cut_off, **backend_options)

    if scipy.sparse.issparse(distance_matrix) and cut_off is None:
        # dense case stored in a sparse distance matrix -> convert to np.ndarray
        distance_matrix = distance_matrix.toarray()

    if is_sparse and distance_matrix.nnz == np.product(distance_matrix.shape):
        logger.warning(
            f"cut_off={cut_off} value has no effect on sparsity of distance matrix, the sparse matrix is "
            f"effectively dense."
        )

    if kmin > 0:
        logger.info("apply continuous nearest neighbor on distance matrix")
        distance_matrix = apply_continuous_nearest_neighbor(distance_matrix, kmin, tol)

    if scipy.sparse.issparse(distance_matrix):
        if not isinstance(distance_matrix, scipy.sparse.csr_matrix):
            # For now only return CSR format.
            distance_matrix = distance_matrix.tocsr()

        # Sort_indices return immediately if indices are already sorted.
        # If not sorted, the call can be costly (depending on nnz), but is better for later usage
        distance_matrix.sort_indices()

    return distance_matrix


if __name__ == "__main__":
    all_available_distance_algorithm()
