""" Unit test for the diffusion_maps module.

"""

import os
import unittest

import matplotlib.pyplot as plt
import scipy.sparse.linalg.eigen.arpack
from scipy.stats import norm
from sklearn.datasets import make_swiss_roll

from datafold.dynfold.utils import downsample
from datafold.dynfold.diffusion_maps import DiffusionMapsVariable
from datafold.dynfold.kernel import DmapKernelFixed
from datafold.dynfold.tests.helper import *


class DiffusionMapsTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.xmin = 0.0
        self.ymin = 0.0
        self.width = 1.0
        self.height = 1e-1
        self.num_samples = 50000
        self.data = make_strip(self.xmin, self.ymin,
                               self.width, self.height,
                               self.num_samples)

    @staticmethod
    def _compute_rayleigh_quotients(matrix, eigenvectors):
        """Compute Rayleigh quotients."""
        n = eigenvectors.shape[0]
        rayleigh_quotients = np.zeros(n)
        for n in range(n):
            v = eigenvectors[n, :]
            rayleigh_quotients[n] = np.dot(v, matrix @ v) / np.dot(v, v)
        rayleigh_quotients = np.sort(np.abs(rayleigh_quotients))
        return rayleigh_quotients[::-1]

    def test_accuracy(self):
        num_samples = 5000
        logging.debug(f'Computing diffusion maps on a matrix of size {num_samples}')
        num_eigenpairs = 10
        epsilon = 5e-1
        downsampled_data = downsample(self.data, num_samples)

        # symmetrize_kernel=False, because the rayleigh_quotient requires the kernel_matrix_
        dm = DiffusionMaps(epsilon, symmetrize_kernel=False, num_eigenpairs=num_eigenpairs).fit(downsampled_data)

        actual_ew = dm.eigenvalues_
        expected_ew = self._compute_rayleigh_quotients(dm.kernel_matrix_, dm.eigenvectors_)

        logging.debug(f"Eigenvalues: {actual_ew}")
        logging.debug(f"Rayleigh quotients: {expected_ew}")

        nptest.assert_allclose(np.abs(actual_ew), np.abs(expected_ew))

    def test_multiple_epsilon_values(self):
        num_samples = 5000
        num_maps = 10
        num_eigenpairs = 10
        epsilon_min, epsilon_max = 1e-1, 1e1
        epsilons = np.logspace(np.log10(epsilon_min),
                               np.log10(epsilon_max), num_maps)

        downsampled_data = downsample(self.data, num_samples)

        evs = np.zeros((num_maps, num_eigenpairs, downsampled_data.shape[0]))
        ews = np.zeros((num_maps, num_eigenpairs))

        logging.basicConfig(level=logging.WARNING)

        for i, epsilon in enumerate(reversed(epsilons)):
            dm = DiffusionMaps(epsilon, num_eigenpairs, symmetrize_kernel=False).fit(downsampled_data)

            evs[i, :, :] = dm.eigenvectors_
            ews[i, :] = dm.eigenvalues_

            ew = dm.eigenvalues_
            rq = self._compute_rayleigh_quotients(dm.kernel_matrix_, dm.eigenvectors_)
            nptest.assert_allclose(np.abs(ew), np.abs(rq))

            # plt.title('$\\epsilon$ = {:.3f}'.format(epsilon))
            # for k in range(1, 10):
            #     plt.subplot(2, 5, k)
            #     plt.scatter(downsampled_data[:, 0], downsampled_data[:, 1],
            #                 c=evs[i, k, :])
            #     plt.xlim([self.xmin, self.xmin + self.width])
            #     plt.ylim([self.ymin, self.ymin + self.height])
            #     plt.tight_layout()
            #     plt.gca().set_title('$\\psi_{}$'.format(k))
            # plt.subplot(2, 5, 10)
            # plt.step(range(ews[i, :].shape[0]), np.abs(ews[i, :]))
            # plt.title('epsilon = {:.2f}'.format(epsilon))
            # plt.show()

    def test_sanity_dense_sparse(self):

        data, _ = make_swiss_roll(1000, random_state=1)

        dense_case = DiffusionMaps(epsilon=1.25, num_eigenpairs=11).fit(data)
        sparse_case = DiffusionMaps(epsilon=1.25, num_eigenpairs=11, cut_off=1E100).fit(data)

        nptest.assert_allclose(dense_case.kernel_matrix_, sparse_case.kernel_matrix_.toarray(), rtol=1E-13, atol=1E-14)
        nptest.assert_allclose(dense_case.eigenvalues_, sparse_case.eigenvalues_, rtol=1E-13, atol=1E-14)

        # TODO: due to the sparse component, it is a bit tricky to compare eigenvectors (this requires more work),
        #  things that can be checked, is eigenvec1 = -eigenvec2? are they self orthogonal eigenvec @ eigenvec = 1, etc.
        # self.assertTrue(np.allclose(dense_case.eigenvectors, sparse_case.eigenvectors,
        #                             rtol=1E-13, atol=1E-14))

    def test_symmetric_dense(self):
        data, _ = make_swiss_roll(1000)

        dmap1 = DiffusionMaps(epsilon=1.5, num_eigenpairs=5, symmetrize_kernel=True).fit(data)
        dmap2 = DiffusionMaps(epsilon=1.5, num_eigenpairs=5, symmetrize_kernel=False).fit(data)

        # make sure that the symmetric transformation is really used
        self.assertTrue(dmap1.kernel_.is_symmetric_transform(is_pdist=True))

        # Note: cannot compare kernel matrices, because they are only similar (sharing same eigenvalues and
        # eigenvectors [after transformation] not equal
        nptest.assert_allclose(dmap1.eigenvalues_, dmap2.eigenvalues_, rtol=1E-14, atol=1E-14)
        self.assertTrue(cmp_eigenvectors(dmap1.eigenvectors_, dmap2.eigenvectors_))

    def test_symmetric_sparse(self):
        data, _ = make_swiss_roll(1000)

        dmap1 = DiffusionMaps(epsilon=1.5, num_eigenpairs=5, cut_off=1E100, symmetrize_kernel=True).fit(data)
        dmap2 = DiffusionMaps(epsilon=1.5, num_eigenpairs=5, cut_off=1E100, symmetrize_kernel=False).fit(data)

        # make sure that the symmetric transformation is really used
        self.assertTrue(dmap1.kernel_.is_symmetric_transform(is_pdist=True))

        # Note: cannot compare kernel matrices, because they are only similar (sharing same eigenvalues and
        # eigenvectors [after transformation] not equal
        nptest.assert_allclose(dmap1.eigenvalues_, dmap2.eigenvalues_, rtol=1E-14, atol=1E-14)
        self.assertTrue(cmp_eigenvectors(dmap1.eigenvectors_, dmap2.eigenvectors_))


class DiffusionMapsLegacyTest(unittest.TestCase):
    """We want to produce exactly the same results as the forked DMAP repository. These are test to make sure this is
    the case. All dmaps have symmetrize_kernel=False to be able to compare the kernel."""

    def test_simple_dataset(self):
        """Taken from method_examples(/diffusion_maps/diffusion_maps.ipynb) repository."""
        data, epsilon = circle_data()

        actual = DiffusionMaps(epsilon=epsilon, num_eigenpairs=11, symmetrize_kernel=False).fit(data)
        expected = legacy_dmap.DiffusionMaps(points=data, epsilon=epsilon)

        cmp_eigenpairs(actual, expected)

    def test_kernel_matrix_simple_dense(self):
        data, epsilon = circle_data()

        actual = DiffusionMaps(epsilon=epsilon, num_eigenpairs=11, symmetrize_kernel=False).fit(data)
        expected = legacy_dmap.DenseDiffusionMaps(points=data, epsilon=epsilon)

        cmp_kernel_matrix(actual, expected, rtol=1E-14, atol=1E-15)

    def test_kernel_matrix_simple_sparse(self):
        data, epsilon = circle_data(nsamples=1000)

        actual = DiffusionMaps(epsilon=epsilon, num_eigenpairs=11, cut_off=1E100, symmetrize_kernel=False).fit(data)
        expected = legacy_dmap.SparseDiffusionMaps(points=data, epsilon=epsilon)

        nptest.assert_allclose(actual.kernel_matrix_.toarray(), expected.kernel_matrix.toarray(), rtol=1E-14, atol=1E-15)

    def test_swiss_roll_dataset(self):
        """Taken from method_examples(/diffusion_maps/diffusion_maps.ipynb) repository."""
        data, _ = make_swiss_roll(n_samples=1000, noise=0.01, random_state=1)

        actual = DiffusionMaps(epsilon=1.25, num_eigenpairs=11, symmetrize_kernel=False).fit(data)
        expected = legacy_dmap.DenseDiffusionMaps(points=data, epsilon=1.25)

        cmp_eigenpairs(actual, expected)

    def test_multiple_cutoff(self):
        data1, _ = make_swiss_roll(1000, random_state=0)
        data1 *= 0.01  # scale data down to allow for smaller cut_offs

        data2, epsilon2 = circle_data()

        cut_off = np.append(np.linspace(0.1, 1, 5), np.linspace(1, 10, 5))

        ne = 5
        for co in cut_off:
            actual1 = DiffusionMaps(epsilon=1E-3, num_eigenpairs=ne, cut_off=co, symmetrize_kernel=False).fit(data1)
            expected1 = legacy_dmap.DiffusionMaps(points=data1, epsilon=1E-3, num_eigenpairs=ne, cut_off=co)

            cmp_eigenpairs(actual1, expected1)

            actual2 = DiffusionMaps(epsilon=epsilon2, num_eigenpairs=ne, cut_off=co, symmetrize_kernel=False).fit(data2)
            expected2 = legacy_dmap.DiffusionMaps(points=data2, epsilon=epsilon2, num_eigenpairs=ne, cut_off=co)

            cmp_eigenpairs(actual2, expected2)

    def test_normalized_kernel(self):

        data, _ = make_swiss_roll(1000, random_state=123)
        epsilon = 1.25

        # actual = DiffusionMaps(epsilon=epsilon, num_eigenpairs=11, is_stochastic=False).fit(data)
        # expected = legacy_dmap.DenseDiffusionMaps(data, epsilon=1.25, normalize_kernel=False)
        # cmp_kernel_matrix(actual, expected, rtol=1E-14, atol=1E-14)
        #
        # actual = DiffusionMaps(epsilon=epsilon, num_eigenpairs=11, is_stochastic=True).fit(data)
        # expected = legacy_dmap.DenseDiffusionMaps(data, epsilon=1.25, normalize_kernel=True)
        # cmp_kernel_matrix(actual, expected, rtol=1E-15, atol=1E-15)

        # Sparse case
        actual = DiffusionMaps(epsilon=epsilon, num_eigenpairs=11, cut_off=3, is_stochastic=False).fit(data)

        expected = legacy_dmap.SparseDiffusionMaps(
            data, epsilon=1.25, num_eigenpairs=11, cut_off=3, normalize_kernel=False)

        cmp_kernel_matrix(actual, expected, rtol=1E-15, atol=1E-15)

        actual = DiffusionMaps(epsilon=epsilon, num_eigenpairs=11, cut_off=3, is_stochastic=True, symmetrize_kernel=False).fit(data)
        expected = legacy_dmap.SparseDiffusionMaps(
            data, epsilon=1.25, num_eigenpairs=11, cut_off=3, normalize_kernel=True)
        cmp_kernel_matrix(actual, expected, rtol=1E-15, atol=1E-15)

    def test_renormalization_factor(self):
        data, _ = make_swiss_roll(1000, random_state=1)
        nfactor = np.linspace(0, 1, 5)

        for factor in nfactor:
            actual = DiffusionMaps(epsilon=1.25, num_eigenpairs=11, symmetrize_kernel=False, alpha=factor).fit(data)
            expected = legacy_dmap.DenseDiffusionMaps(data, epsilon=1.25, renormalization=factor)
            cmp_dmap_legacy(actual, expected, rtol=1E-15, atol=1E-15)

            actual = DiffusionMaps(epsilon=1.25, num_eigenpairs=11, cut_off=3, symmetrize_kernel=False, alpha=factor).fit(data)
            expected = legacy_dmap.SparseDiffusionMaps(data, epsilon=1.25, cut_off=3, renormalization=factor)
            cmp_dmap_legacy(actual, expected, rtol=1E-15, atol=1E-15)

    def test_multiple_epsilon(self):

        data, _ = make_swiss_roll(1000, random_state=123)
        epsilons = np.linspace(1.2, 1.7, 5)[1:]
        ne = 5
        for eps in epsilons:
            actual_dense = DiffusionMaps(epsilon=eps, num_eigenpairs=ne, symmetrize_kernel=False, ).fit(data)
            expected_dense = legacy_dmap.DenseDiffusionMaps(points=data, num_eigenpairs=ne, epsilon=eps)

            try:
                actual_sparse = DiffusionMaps(epsilon=eps, num_eigenpairs=ne, cut_off=1, symmetrize_kernel=False).fit(data)
                expected_sparse = legacy_dmap.SparseDiffusionMaps(
                    points=data, epsilon=eps, num_eigenpairs=ne, cut_off=1)

            except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence as e:
                print(f"Did not converge for epsilon={eps}. This can happen due to random effects of the sparse "
                      "eigenproblem solver (and usually a bad conditioned matrix).")
                raise e

            cmp_dmap_legacy(actual_dense, expected_dense, rtol=1E-15, atol=1E-15)
            cmp_dmap_legacy(actual_sparse, expected_sparse, rtol=1E-14, atol=1E-14)

    def test_num_eigenpairs(self):

        data, _ = make_swiss_roll(1000)
        num_eigenpairs = np.linspace(10, 50, 5).astype(np.int)

        for ne in num_eigenpairs:
            actual = DiffusionMaps(epsilon=1.25, num_eigenpairs=ne, symmetrize_kernel=False).fit(data)
            expected = legacy_dmap.DenseDiffusionMaps(data, epsilon=1.25, num_eigenpairs=ne)

            cmp_dmap_legacy(actual, expected, rtol=1E-15, atol=1E-15)

            actual = DiffusionMaps(epsilon=1.25, num_eigenpairs=ne, cut_off=3, symmetrize_kernel=False).fit(data)
            expected = legacy_dmap.SparseDiffusionMaps(data, epsilon=1.25, cut_off=3, num_eigenpairs=ne)

            cmp_dmap_legacy(actual, expected, rtol=1E-15, atol=1E-15)


class DiffusionMapsVariableTest(unittest.TestCase):

    @staticmethod
    def eig_neg_factor(exact, approx):
        # chooses the right "direction" of eigenvector to compare with exact solution
        n1 = exact.flatten() @ approx.flatten()
        n2 = exact.flatten() @ (-1 * approx.flatten())

        if n2 > n1:
            return -1
        else:
            return 1

    @staticmethod
    def plot_quantities(data, dmap):

        h3 = lambda x: 1 / np.sqrt(6) * (x ** 3 - 3 * x)  # 3rd Hermetian polynomial
        assert data.ndim == 2 and data.shape[1] == 1

        eigvec = dmap.eigenvectors_.T  #TODO #44

        f, ax = plt.subplots(ncols=3, nrows=3)
        f.suptitle(f"N={data.shape[0]}, eps={dmap.epsilon}, beta={dmap.beta}, expected_dim={dmap.expected_dim}, "
                   f"k={dmap.k}")

        ax[0][0].plot(data, dmap.rho0_, '-')
        ax[0][0].set_title("rho0 - ad hoc bandwidth function")

        ax[0][1].hist(data, density=True, bins=100, color="grey", edgecolor="black")
        ax[0][1].plot(data, dmap.peq_est_, '*', color="#1f77b4", label="estimate")
        ax[0][1].set_title("hist distribution data")

        factor = DiffusionMapsVariableTest.eig_neg_factor(h3(data), eigvec[:, 3])  # TODO #44
        ax[1][0].plot(np.linspace(-3, 3, 200), h3(np.linspace(-3, 3, 200)), label="exact, H3")
        ax[1][0].plot(data[:, 0], factor * eigvec[:, 3], '-', label=f"dmap_variable_kernel, ev_idx=3")

        ax[1][0].legend()

        ax[0][2].plot(data, dmap.rho_, '*')
        ax[0][2].set_title("rho - bandwidth function")
        ax[1][1].plot(data, dmap.q0_, '*')
        ax[1][1].set_title("q0 - sampling density - estimate")
        ax[1][2].plot(data, dmap.peq_est_, '*', label="estimate")

        ax[1][2].plot(np.linspace(-3, 3, 200), norm.pdf(np.linspace(-3, 3, 200), 0, 1), label="exact")
        ax[1][2].legend()
        ax[1][2].set_title("peq - invariant measure, estimate")

        M = dmap.eigenvalues_.shape[0]
        ax[2][0].plot(np.arange(M), dmap.eigenvalues_, '*-')
        ax[2][0].set_xlabel("idx")
        ax[2][0].set_ylabel("eigval")

        im = ax[2][1].imshow(np.abs((eigvec.T @ eigvec)) / eigvec.shape[0])
        ax[2][1].set_title("inner products of EV (abs and rel)")
        f.colorbar(im, ax=ax[2][1])

    def test_ornstein_uhlenbeck(self):
        from scipy.special import erfinv

        nr_samples = 5000
        num_eigenpairs = 20

        def compute_nice_ou(N):
            # non-random sampling
            delta = 1 / (N + 1)
            xtilde = delta * np.arange(1, N + 1)
            x = np.sqrt(2) * erfinv(2 * xtilde - 1)

            # bool_idx = np.logical_and(x >= -3, x <=3)
            return x[:, np.newaxis]

        X = compute_nice_ou(nr_samples)

        dmap = DiffusionMapsVariable(epsilon=0.001, num_eigenpairs=num_eigenpairs, k=100, expected_dim=1, beta=-0.5,
                                     symmetrize_kernel=True).fit(X)

        # PLOT:
        plot = False

        if plot:
            DiffusionMapsVariableTest.plot_quantities(X, dmap)
            plt.show()

        # TESTS:
        h3 = lambda x: 1 / np.sqrt(6) * (x ** 3 - 3 * x)  # 3rd Hermetian polynomial
        factor = DiffusionMapsVariableTest.eig_neg_factor(h3(X), dmap.eigenvectors_.T[:, 3])

        actual = factor * dmap.eigenvectors_.T[:, 3]  # TODO #44
        expected = h3(X)

        # using only a reference computation (fails if quality gets worse)
        self.assertTrue(np.abs(actual-expected.ravel()).max() <= 1.5943698803387)

        actual = dmap.peq_est_
        expected = norm.pdf(X, 0, 1)

        # using only a reference computation (fails if quality gets worse)
        nptest.assert_allclose(actual, expected.ravel(), atol=0.0002519, rtol=0.29684159)

class TestDiffusionMapsKernelTest(unittest.TestCase):

    def test_is_symmetric01(self):
        # stochastic False

        # Note: in this case the alpha value is ignored
        k1 = DmapKernelFixed(is_stochastic=False, symmetrize_kernel=True)
        self.assertTrue(k1.is_symmetric)
        self.assertFalse(k1.is_symmetric_transform(is_pdist=True))  # No transformation is required

        k2 = DmapKernelFixed(is_stochastic=False, symmetrize_kernel=False)  # Even now the symmetric kernel is true
        self.assertTrue(k2.is_symmetric)
        self.assertFalse(k1.is_symmetric_transform(is_pdist=True))  # No transformation is required

    def test_is_symmetric02(self):
        # symmetric_kernel and alpha == 0
        k1 = DmapKernelFixed(is_stochastic=True, alpha=0, symmetrize_kernel=False)
        self.assertFalse(k1.is_symmetric)
        self.assertFalse(k1.is_symmetric_transform(is_pdist=True))

        k2 = DmapKernelFixed(is_stochastic=True, alpha=0, symmetrize_kernel=True)
        self.assertTrue(k2.is_symmetric)
        self.assertTrue(k2.is_symmetric_transform(is_pdist=True))

    def test_is_symmetric03(self):
        # symmetric_kernel and alpha > 0
        k1 = DmapKernelFixed(is_stochastic=True, alpha=1, symmetrize_kernel=False)
        self.assertFalse(k1.is_symmetric)
        self.assertFalse(k1.is_symmetric_transform(is_pdist=True))

        k2 = DmapKernelFixed(is_stochastic=True, alpha=1, symmetrize_kernel=True)
        self.assertTrue(k2.is_symmetric)
        self.assertTrue(k2.is_symmetric_transform(is_pdist=True))

    def test_is_symmetric04(self):
        # ways False when is_pdist==False
        k1 = DmapKernelFixed(is_stochastic=True, alpha=1, symmetrize_kernel=True)
        self.assertFalse(k1.is_symmetric_transform(is_pdist=False))

        k2 = DmapKernelFixed(is_stochastic=False, alpha=1, symmetrize_kernel=True)
        self.assertFalse(k2.is_symmetric_transform(is_pdist=False))


if __name__ == '__main__':

    verbose = os.getenv('VERBOSE')
    if verbose is not None:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    else:
        logging.basicConfig(level=logging.ERROR, format='%(message)s')

    # Comment in to run/debug specific tests

    #t = DiffusionMapsLegacyTest()
    #t.setUp()
    #t.test_kernel_matrix_simple_dense()
    #exit()

    # DiffusionMapsLegacyTest().test_sanity_dense_sparse()
    # exit()
    unittest.main()