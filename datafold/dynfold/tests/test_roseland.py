""" Unit test for the roseland module.

"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg
from scipy.stats import norm
from sklearn.datasets import make_swiss_roll

import datafold.pcfold as pfold
from datafold.dynfold import Roseland
from datafold.dynfold.tests.helper import *
from datafold.pcfold import GaussianKernel
from datafold.utils.general import random_subsample

try:
    import rdist
except ImportError:
    IMPORTED_RDIST = False
else:
    IMPORTED_RDIST = True


class RoselandTest(unittest.TestCase):
    def test_dense_sparse(self):

        data, _ = make_swiss_roll(1000, random_state=1)
        data_landmark, _ = random_subsample(data, 250)

        dense_case = Roseland(
            GaussianKernel(epsilon=1.25), Y=data_landmark, n_svdpairs=11
        ).fit(data, store_kernel_matrix=True)
        sparse_case = Roseland(
            GaussianKernel(epsilon=1.25),
            Y=data_landmark,
            n_svdpairs=11,
            dist_kwargs=dict(cut_off=1e100),
        ).fit(data, store_kernel_matrix=True)

        nptest.assert_allclose(
            dense_case.kernel_matrix_,
            sparse_case.kernel_matrix_.toarray(),
            rtol=1e-13,
            atol=1e-14,
        )
        nptest.assert_allclose(
            dense_case.svdvalues_, sparse_case.svdvalues_, rtol=1e-13, atol=1e-14
        )

        assert_equal_eigenvectors(dense_case.svdvectors_, sparse_case.svdvectors_)

    def test_time_exponent(self):
        data, _ = make_swiss_roll(2000, random_state=1)
        data_landmark, _ = random_subsample(data, 250)

        actual1 = Roseland(
            GaussianKernel(epsilon=1.5), n_svdpairs=5, Y=data_landmark, time_exponent=0
        ).fit_transform(data)

        actual2 = Roseland(
            GaussianKernel(epsilon=1.5),
            n_svdpairs=5,
            Y=data_landmark,
            time_exponent=1e-14,
        ).fit_transform(data)

        nptest.assert_allclose(np.abs(actual1), np.abs(actual2), rtol=1e-9, atol=1e-13)

    def test_set_target_coords1(self):
        X_swiss_all, _ = make_swiss_roll(n_samples=2000, noise=0, random_state=5)
        data_landmark, _ = random_subsample(X_swiss_all, 400)

        actual_rose = Roseland(
            GaussianKernel(epsilon=2.0), n_svdpairs=6, Y=data_landmark
        ).set_target_coords([1, 5])
        actual = actual_rose.fit_transform(X_swiss_all)

        actual_rose2 = Roseland(
            GaussianKernel(epsilon=2.0), n_svdpairs=6, Y=data_landmark
        )
        actual2 = (
            actual_rose2.fit(X_swiss_all)
            .set_target_coords([1, 5])
            .transform(X_swiss_all)
        )

        expected = Roseland(
            GaussianKernel(epsilon=2.0), n_svdpairs=6, Y=data_landmark
        ).fit_transform(X_swiss_all)

        nptest.assert_allclose(
            np.abs(actual), np.abs(expected[:, [1, 5]]), rtol=1e-10, atol=1e-14
        )
        nptest.assert_allclose(
            np.abs(actual2), np.abs(expected[:, [1, 5]]), rtol=1e-10, atol=1e-14
        )
        nptest.assert_allclose(
            np.abs(actual_rose.svdvectors_), np.abs(expected), rtol=1e-10, atol=1e-14
        )
        nptest.assert_allclose(
            np.abs(actual_rose2.svdvectors_), np.abs(expected), rtol=1e-10, atol=1e-14
        )

        self.assertEqual(actual_rose.n_features_out_, 2)
        self.assertEqual(actual_rose.feature_names_out_, None)

        self.assertEqual(actual_rose2.n_features_out_, 2)
        self.assertEqual(actual_rose2.feature_names_out_, None)

    def test_set_target_coords2(self):
        X_swiss_all, _ = make_swiss_roll(n_samples=500, noise=0, random_state=5)

        actual_rose = Roseland(GaussianKernel(epsilon=2.0), n_svdpairs=6).fit(
            X_swiss_all
        )
        actual_rose.set_target_coords(indices=[0, 1])

        with self.assertRaises(TypeError):
            actual_rose.set_target_coords(indices=[0.0, 1.5])

        with self.assertRaises(ValueError):
            actual_rose.set_target_coords(indices=[-1, 2])

        actual_rose.set_target_coords(indices=[0, 5])
        with self.assertRaises(ValueError):
            actual_rose.set_target_coords(indices=[0, 6])

    def test_nystrom_out_of_sample_swiss_roll(self, plot=False):

        X_swiss_all, color_all = make_swiss_roll(
            n_samples=4000, noise=0, random_state=5
        )
        data_landmark, _ = random_subsample(X_swiss_all, 1000)

        setting = {
            "kernel": GaussianKernel(epsilon=1.7),
            "n_svdpairs": 7,
            "Y": data_landmark,
        }

        rose_embed = Roseland(**setting).fit(X_swiss_all)

        if plot:
            from datafold.utils.plot import plot_pairwise_eigenvector

            plot_pairwise_eigenvector(
                eigenvectors=rose_embed.transform(X_swiss_all),
                n=1,
                fig_params=dict(figsize=[6, 6]),
                scatter_params=dict(cmap=plt.cm.Spectral, c=color_all),
            )

        rose_embed_eval_expected = rose_embed.svdvectors_[:, [1, 5]]
        rose_embed_eval_actual = rose_embed.set_target_coords(indices=[1, 5]).transform(
            X=X_swiss_all
        )

        # even though the target_coords were set, still all eigenvectors must be
        # accessible
        self.assertEqual(rose_embed.svdvectors_.shape[1], 7)

        nptest.assert_allclose(
            rose_embed_eval_actual, rose_embed_eval_expected, atol=1e-15
        )

        if plot:
            X_swiss_oos, color_oos = make_swiss_roll(
                n_samples=30000, noise=0, random_state=5
            )

            f, ax = plt.subplots(2, 3, figsize=(4, 4))
            marker = "."
            markersize = 0.2
            ax[0][0].scatter(
                rose_embed_eval_expected[:, 0],
                rose_embed_eval_expected[:, 1],
                s=markersize,
                marker=marker,
                c=color_all,
            )
            ax[0][0].set_title("expected Roseland singular vectors (Fit)")

            ax[0][1].scatter(
                rose_embed_eval_actual[:, 0],
                rose_embed_eval_actual[:, 1],
                s=markersize,
                marker=marker,
                c=color_all,
            )
            ax[0][1].set_title("actual Roseland singular vectors (fit -> transform)")

            absdiff = np.abs(rose_embed_eval_expected - rose_embed_eval_actual)
            abs_error_norm = np.linalg.norm(absdiff, axis=1)

            error_scatter = ax[0][2].scatter(
                rose_embed_eval_expected[:, 0],
                rose_embed_eval_expected[:, 1],
                s=markersize,
                marker=marker,
                c=abs_error_norm,
                cmap=plt.get_cmap("Reds"),
            )

            f.colorbar(error_scatter, ax=ax[0][2])
            ax[0][2].set_title("abs. difference")

            gh_embed_eval_oos = rose_embed.transform(X_swiss_oos)
            ax[1][0].scatter(
                gh_embed_eval_oos[:, 0],
                gh_embed_eval_oos[:, 1],
                s=markersize,
                marker=marker,
                c=color_oos,
            )

            ax[1][0].set_title(
                f"Roseland Nyström out-of-sample \n ({gh_embed_eval_oos.shape[0]} points) "
            )

            ax[1][2].text(
                0.01,
                0.5,
                f"both have same setting " f"chosen_svdvectors={[1, 5]}",
            )

            plt.show()

    def test_dist_kwargs(self):
        X_swiss_all, _ = make_swiss_roll(n_samples=4000, noise=0, random_state=5)
        data_landmark, _ = random_subsample(X_swiss_all, 1000)

        rose = Roseland(
            kernel=GaussianKernel(), Y=data_landmark, dist_kwargs=dict(cut_off=2)
        ).fit(X_swiss_all, store_kernel_matrix=True)

        self.assertEqual(rose.Y_fit_.dist_kwargs["cut_off"], 2)
        self.assertTrue(scipy.sparse.issparse(rose.kernel_matrix_))

    def test_speed(self):
        from time import time

        num_samples = 15000
        xmin, ymin = -2, -1
        width, height = 4, 2
        random_state = 1

        data = make_strip(xmin, ymin, width, height, num_samples)
        rng = np.random.default_rng(random_state)

        n_large_points = int(15000)
        L1, L2 = 4, 3  # width and height of the rectangle
        data = rng.uniform(
            low=(-L1 / 2, -L2 / 2), high=(L1 / 2, L2 / 2), size=(n_large_points, 2)
        )

        pcm = pfold.PCManifold(data)

        setting = {"kernel": GaussianKernel(), "n_svdpairs": 5, "gamma": 0.25}

        t0 = time()
        rose_embed = Roseland(**setting)

        t1 = time()
        rose_embed.fit(data, store_kernel_matrix=True)
        t2 = time()
        kernel_output = rose_embed.Y_fit_.compute_kernel_matrix(rose_embed.X_fit_)
        t22 = time()
        svdvects, svdvals, right_svdvects = scipy.sparse.linalg.svds(
            rose_embed.kernel_matrix_,
            k=rose_embed.n_svdpairs,
            which="LM",
        )
        t3 = time()

        print(
            f"kernel+svds: {t22-t2+t3-t2}, fit: {t2-t1}, kernel only: {t22-t2}, svds: {t3-t2}"
        )

        return 1


if __name__ == "__main__":

    t = RoselandTest()
    exit()

    unittest.main()
