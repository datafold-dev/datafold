#!/usr/bin/env python3

import numpy as np
import scipy

from pcmanifold.distance import compute_distance_matrix

# --------------------------------------------------
# people who contributed code
__authors__ = "Felix Dietrich"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


# TODO: make GaussienRegression also a KernelMethod and adapt to scikit-learn interface

class PCMGPRegression(object):
    """
    Gaussian Process Regression on the point cloud manifold.
    """

    def __init__(self, pcm, rcond=1e-10):
        """
        pcm: PCManifold object
        """

        # TODO: enforce RBF kernel?
        self._pcm = pcm
        self._rcond = rcond

    @property
    def fx(self):
        return self._fx

    @staticmethod
    def sample(pcm, n_functions=1, output_noise_std=1e-5):
        """Samples random, smooth functions on the point cloud. does NOT use sparsity!!! """

        gp_kernel = pcm.compute_kernel_matrix()
        sgpk = gp_kernel  # scipy.linalg.fractional_matrix_power(gp_kernel @ gp_kernel.T, 1/2) # make it spd

        gp_chol = np.linalg.cholesky(sgpk + output_noise_std * np.identity(gp_kernel.shape[0]))

        u = np.random.randn(pcm.shape[0], n_functions)  # TODO: look this up, what is "u"?
        function_samples = gp_chol @ u
        return function_samples

    def regression(self, fx, regression_scale=1, solver_tolerance=1e-6, sigma=1e-5, tikhonov_regularization=False):
        """
        regresses the function values in (N x d),
        and returns the vector (k(D,D')^{-1}fx).
        """

        self._regression_scale = regression_scale
        self._pcm.kernel.epsilon = self._regression_scale

        if tikhonov_regularization:
            self._kernel_matrix = self._pcm.compute_kernel_matrix()  # TODO: could be dense/sparse?

            self._kernel_matrix = self._kernel_matrix.diagonal() + \
                                  sigma ** 2 * scipy.sparse.identity(self._kernel_matrix.shape[0])

            self.__kernel_matrix_inverse = None  # scipy.sparse.linalg.splu(kernel_matrix)#, \
            # permc_spec = "NATURAL", \
            # diag_pivot_thresh=0, \
            # options={"SymmetricMode":True})

            # __L = np.linalg.cholesky(kernel_matrix.todense())
            # alpha0,_,_,_ = np.linalg.lstsq(__L, fx, rcond=self.__rcond)
            # alpha,_,_,_ = np.linalg.lstsq(__L.T, alpha0, rcond=self.__rcond)
            # self.__kfx = alpha

            # self.__kfx = scipy.sparse.linalg.spsolve(kernel_matrix, fx)# self.__kernel_matrix_inverse.solve(fx)

            # TODO: maybe write this with a full loop for code-readability, also preallocate the _kfx
            self._kfx = np.column_stack([scipy.sparse.linalg.lsmr(self._kernel_matrix, fx[:, k], atol=solver_tolerance,
                                                                  btol=solver_tolerance)[0]
                                         for k in range(fx.shape[1])])  # self.__kernel_matrix_inverse.solve(fx)
        else:
            self._kernel_matrix = self._pcm.compute_kernel_matrix()
            # kernel_matrix = scipy.sparse.csc_matrix(kernel_matrix + sigma**2 * scipy.sparse.identity(kernel_matrix.shape[0]))

            self._invdiag = scipy.sparse.diags(1.0 / (self._rcond + self._kernel_matrix.sum(axis=1).A.ravel()))
            self._kernel_matrix = self._invdiag @ self._kernel_matrix
            # invdiag = scipy.sparse.diags(1.0/(self.__rcond+kernel_matrix.sum(axis=1).A.ravel()))
            # kernel_matrix = invdiag @ kernel_matrix

            # TODO: the same as above?
            self._kfx = np.column_stack([scipy.sparse.linalg.lsmr(self._kernel_matrix, fx[:, k],
                                                                  atol=solver_tolerance, btol=solver_tolerance)[0]
                                         for k in range(fx.shape[1])])

        self._fx = fx  # store the values for later use

        return self._kfx

    def evaluate(self, new_points, use_distances=None, use_kfx=None, return_cov=False, metric=None,
                 tikhonov_regularization=False):

        if use_kfx is None and self._kfx is None:
            raise ValueError("must run regression first")

        if use_kfx is None:
            use_kfx = self._kfx

        if metric is None:
            metric = "euclidean"

        if use_distances is None:
            # add a little bit of noise to avoid zero distances to points that are on top of points in self.__points
            noise = (np.random.rand(new_points.shape[0], self._pcm.shape[1]) - .5) * self._rcond

            distance_matrix = compute_distance_matrix(X=self._pcm, Y=new_points + noise,
                                                      metric=metric,
                                                      cut_off=self._pcm.cut_off,
                                                      backend=self._pcm.dist_backend,
                                                      njobs=1)

            # distance_matrix = self._pcm.sparse_distance_matrix(Y=new_points + noise, metric=metric)
        else:
            # TODO: this does not work at the moment, the kernel matrix is not using it.
            distance_matrix = use_distances

        if tikhonov_regularization:
            kyx = self._pcm.kernel.eval(distance_matrix)
            # kyx = self._pcm.sparse_kernel_matrix(distance_matrix=distance_matrix, scale=self._regression_scale, add_identity=False)
            mean = np.array(kyx.T.dot(scipy.sparse.csr_matrix(use_kfx)).todense())
        else:
            # add a little bit of noise to avoid zero distances to points that are on top of points in self.__points
            noise = (np.random.rand(new_points.shape[0], self._pcm.shape[1]) - .5) * self._rcond
            kyx = self._pcm.compute_kernel_matrix(Y=new_points + noise)

            # invdiag1 = scipy.sparse.diags(1.0 / (self._rcond + kyx.sum(axis=1).T.ravel()))
            invdiag2 = scipy.sparse.diags(1.0 / (self._rcond + np.array(kyx.sum(axis=0)).ravel()), offsets=0)

            kyx = kyx @ invdiag2
            # invdiag0 = scipy.sparse.diags(1.0/(self.__rcond+kyx.sum(axis=1).A.ravel()))
            # kyx = invdiag0 @ kyx
            mean = np.array(kyx.T.dot(scipy.sparse.csr_matrix(use_kfx)).todense())

        if return_cov:

            distance_matrix = compute_distance_matrix(X=new_points, metric=metric, cut_off=None, backend="brute",
                                                      njobs=1)
            kernel_matrix = self._pcm.kernel.eval(distance_matrix).toarray()

            v = self.regression(kyx.T)
            # v,_,_,_ = np.linalg.lstsq(self.__L, k_xxs, rcond=self.__rcond)
            std = kernel_matrix - np.dot(v.T, v)
            return mean, std

        else:
            return mean

    def optimize_regression(self, fx, lim_scales=None, n_samples=15, verbosity_level=0):

        if lim_scales is None:
            lim_scales = [1e-7, 1]

        scales = np.exp(np.linspace(np.log(lim_scales[0]), np.log(lim_scales[1]), n_samples))
        lls = np.zeros((n_samples,))
        testpoints = self._pcm + (np.random.rand(self._pcm.shape[0], self._pcm.shape[1]) - .5) * 1e-4

        for k in range(n_samples):
            self.regression(fx=fx, regression_scale=scales[k])
            test_fx = self.evaluate(new_points=testpoints)
            # ll = self.log_marginal_likelihood(fx=fx, regression_scale = scales[k])
            lls[k] = np.linalg.norm(test_fx - fx)
            if verbosity_level > 0:
                print(str(k) + " ", end="")
        return scales, lls

    def log_marginal_likelihood(self, fx, regression_scale=None):
        # compute the decomposition for fx

        if regression_scale is None:
            regression_scale = 1  # TODO: check, the next line would fail if regression_scale is None

        self.regression(fx, regression_scale * 4)
        # compute the functional part of the log likelihood

        if len(fx.shape) == 1:
            fx = fx.reshape(-1, 1)
            kfx = self._kfx.reshape(-1, 1)
        else:
            kfx = self._kfx

        part1 = 0
        for k in range(fx.shape[1]):
            part1 += -1 / 2 * np.dot(fx[:, k].T, kfx[:, k])
        # compute the kernel part of the log likelihood
        # k = self.__kernel_matrix.todense()
        # L = np.linalg.cholesky(k)

        part2 = -1 / 2 * np.log(np.sum(self._kernel_matrix.diagonal()))
        part3 = -self._kernel_matrix.shape[0] / 2 * np.log(2 * np.pi)

        return part1 / fx.shape[1] + part2 + part3
