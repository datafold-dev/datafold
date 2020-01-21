import abc

import numexpr as ne
import numpy as np
import scipy.sparse
import scipy.spatial
from sklearn.gaussian_process.kernels import (
    RBF,
    Kernel,
    NormalizedKernelMixin,
    StationaryKernelMixin,
    _check_length_scale,
)

from datafold.pcfold.distance import compute_distance_matrix


def apply_kernel_function(distance_matrix, kernel_function):
    if scipy.sparse.issparse(distance_matrix):
        kernel = distance_matrix
        # NOTE: applies on stored data, it is VERY important, that real distance zeros are
        # included in 'distance_matrix' (E.g. normalized kernels have to have a 1.0 on
        # the diagonal) are included in the sparse matrix!
        kernel.data = kernel_function(kernel.data)
    else:
        kernel = kernel_function(distance_matrix)

    return kernel


def apply_kernel_function_numexpr(distance_matrix, expr, expr_dict):

    assert "D" not in expr_dict.keys()

    if scipy.sparse.issparse(distance_matrix):
        # copy because the distance matrix may be used further by the user
        distance_matrix = distance_matrix.copy()
        expr_dict["D"] = distance_matrix.data
        ne.evaluate(expr, expr_dict, out=distance_matrix.data)
        return distance_matrix
    else:
        expr_dict["D"] = distance_matrix
        return ne.evaluate(expr, expr_dict)


class PCManifoldKernelMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):
        """Evaluation calls for kernels used in PCManifold"""

    @abc.abstractmethod
    def eval(self, distance_matrix):
        """Evaluate kernel on an already computed distance matrix. Note: there are no
        checks whether the correct kernel metric was used. 'distance_matrix' may be
        sparse or dense. For the sparse case note that it acts on all stored
        data, i.e. "real zeros" by distance have to be stored."""


class RadialBasisKernel(RBF, PCManifoldKernelMixin):
    """Overwrites selected functions of sklearn.RBF in order to use sparse distance
    matrix computations."""

    def __init__(self, epsilon=1.0, length_scale_bounds=(1e-5, 1e5)):
        self._epsilon = epsilon

        # sqrt because of slightly different notation of the kernel
        super(RadialBasisKernel, self).__init__(
            length_scale=np.sqrt(epsilon), length_scale_bounds=length_scale_bounds
        )

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value
        # keep aligned with super class from scikit learn
        self.length_scale = np.sqrt(self.epsilon)

    def __call__(
        self,
        X,
        Y=None,
        eval_gradient=False,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):
        # TODO: only this kernel so far has "eval_gradient" -- should that go into the
        #  general interface?
        # TODO: maybe remove all default parameters from the kernel interface!

        X = np.atleast_2d(X)

        if Y is not None:
            Y = np.atleast_2d(Y)

        if dist_backend_kwargs is None:
            dist_backend_kwargs = {}

        _check_length_scale(X, self.length_scale)  # sklearn function

        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="sqeuclidean",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **dist_backend_kwargs
        )

        if eval_gradient and scipy.sparse.issparse(distance_matrix):
            raise NotImplementedError(
                "Gradient is not implemented for sparse distance matrix"
            )

        kernel_matrix = self.eval(distance_matrix)

        if not eval_gradient:
            return kernel_matrix
        else:

            # TODO: check gradient code (are both correct, what are the differences, link
            #  to comp. explanations, ...):
            if Y is None:
                # TODO: used from super class sklean.RBF kernel
                return kernel_matrix, self._gradient(distance_matrix, kernel_matrix)
            else:
                # TODO: used from kernels.py code (and adapted)
                return kernel_matrix, self._gradient_given_Y(X, Y, kernel_matrix)

    def eval(self, distance_matrix):
        # Security copy, the distance matrix is maybe required again (for gradient,
        # or other computations...)

        return apply_kernel_function_numexpr(
            distance_matrix,
            expr="exp((- 1 / (2*eps)) * D)",
            expr_dict={"eps": self._epsilon},
        )

    def _gradient(self, distance_matrix, kernel_matrix=None):

        # NOTE: Copied code from the super class RBF
        if self.hyperparameter_length_scale.fixed:
            # Hyperparameter l kept fixed
            return np.empty((kernel_matrix.shape[0], kernel_matrix.shape[0], 0))
        elif not self.anisotropic or self.length_scale.shape[0] == 1:
            kernel_gradient = kernel_matrix * distance_matrix[:, :, np.newaxis]
            return kernel_gradient
        elif self.anisotropic:

            # We need to recompute the pairwise dimension-wise distances
            kernel_gradient = (
                kernel_matrix[:, np.newaxis, :] - kernel_matrix[np.newaxis, :, :]
            ) ** 2 / (self.length_scale ** 2)
            kernel_gradient *= kernel_matrix[..., np.newaxis]
            return kernel_gradient

    def _gradient_given_Y(self, X, Y, kernel_matrix):
        """ computes the gradient of the kernel w.r.t. the x argument """

        if scipy.sparse.issparse(kernel_matrix):
            raise NotImplementedError("Not implemented sparse kernel matrix version.")

        # this is very heavy on the memory. TODO: improve it.
        xmy = np.zeros((Y.shape[1], Y.shape[0], X.shape[0]))

        if Y.shape[1] == 1:
            raise ValueError("one dim. grad not implemented yet")
        else:
            for k in range(
                Y.shape[1]
            ):  # TODO: check if this loop can be optimized, probably slow
                xm = Y[:, k].T @ np.ones((1, X.shape[0]))
                ym = X[:, k].T @ np.ones((1, Y.shape[0])).T
                xmy[k, :, :] = xm - ym

        res = np.zeros((Y.shape[1], kernel_matrix.shape[0], kernel_matrix.shape[1]))

        for k in range(res.shape[0]):
            res[k, :, :] = -2 / self._epsilon * kernel_matrix * np.array(xmy[k, :])

        return res


@NotImplementedError
class PerceptronKernel:
    """""" ""


class MultiquadraticKernel(
    StationaryKernelMixin, NormalizedKernelMixin, PCManifoldKernelMixin, Kernel
):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(MultiquadraticKernel, self).__init__()

    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):

        # TODO: check metric!
        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="XXXX",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **dist_backend_kwargs
        )

        kernel_matrix = self.eval(distance_matrix)
        return kernel_matrix

    def eval(self, distance_matrix):
        kernel_func = lambda dist: np.sqrt(
            np.square(np.multiply(dist, 1.0 / self.epsilon)) + 1
        )
        kernel_matrix = apply_kernel_function(distance_matrix, kernel_func)
        return kernel_matrix


class InverseMultiQuadraticKernel(
    StationaryKernelMixin, NormalizedKernelMixin, PCManifoldKernelMixin, Kernel
):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(InverseMultiQuadraticKernel, self).__init__()

    def __call__(
        self, X, Y=None, dist_cut_off=None, dist_backend="brute", **backend_options
    ):

        # TODO: check metric
        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="XXXX",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **backend_options
        )

        kernel_matrix = self.eval(distance_matrix)
        return kernel_matrix

    def eval(self, distance_matrix):
        kernel_func = lambda dist: np.reciprocal(
            np.sqrt(np.square(np.multiply(dist, 1 / self.epsilon)) + 1)
        )
        kernel_matrix = apply_kernel_function(distance_matrix, kernel_func)
        return kernel_matrix


class InverseQuadraticKernel(PCManifoldKernelMixin, Kernel):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(InverseQuadraticKernel, self).__init__()

    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):

        # TODO: check metric
        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="XXXX",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **dist_backend_kwargs
        )

        kernel_matrix = self.eval(distance_matrix)
        return kernel_matrix

    def eval(self, distance_matrix):
        kernel_func = lambda dist: np.clip(
            1 - np.square(1.0 / self.epsilon * dist), 0, 1
        )
        kernel_matrix = apply_kernel_function(distance_matrix, kernel_func)
        return kernel_matrix


class OUKernel(
    StationaryKernelMixin, NormalizedKernelMixin, PCManifoldKernelMixin, Kernel
):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(OUKernel, self).__init__()

    def __call__(
        self,
        X,
        Y=None,
        dist_cut_off=None,
        dist_backend="brute",
        kernel_kwargs=None,
        dist_backend_kwargs=None,
    ):
        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="euclidean",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **dist_backend_kwargs
        )

        kernel_matrix = self.eval(distance_matrix)
        return kernel_matrix

    def eval(self, distance_matrix):
        kernel_matrix = apply_kernel_function(
            distance_matrix, lambda x: np.exp(-x / self.epsilon)
        )
        return kernel_matrix


class LanczosKernel(NormalizedKernelMixin, PCManifoldKernelMixin, Kernel):
    # TODO: check other kernels

    def __init__(self, epsilon=1, width=2):
        self.epsilon = epsilon
        self.width = width
        super(LanczosKernel, self).__init__()

    def __call__(
        self, X, Y=None, dist_cut_off=None, dist_backend="brute", **backend_options
    ):

        # TODO: check if the metric is correct
        distance_matrix = compute_distance_matrix(
            X,
            Y,
            metric="euclidean",
            cut_off=dist_cut_off,
            backend=dist_backend,
            **backend_options
        )

        kernel_matrix = self.eval(distance_matrix)
        return kernel_matrix

    def eval(self, distance_matrix):
        bool_idx = distance_matrix < self.width
        kernel_matrix = bool_idx * (
            np.sinc(distance_matrix / self.epsilon)
            * np.sinc(distance_matrix / (self.epsilon * self.width))
        )
        return kernel_matrix

    def is_stationary(self):
        return False  # TODO: check if the kernel is really not stationary
