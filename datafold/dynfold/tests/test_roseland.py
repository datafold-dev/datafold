import unittest
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pytest
import scipy.sparse.linalg
from sklearn.datasets import make_swiss_roll

from datafold.dynfold import Roseland
from datafold.dynfold.tests.test_helper import assert_equal_eigenvectors
from datafold.pcfold import GaussianKernel, TSCDataFrame
from datafold.utils.general import random_subsample
from datafold.utils.plot import plot_pairwise_eigenvector


class RoselandTest(unittest.TestCase):
    def test_dense_sparse(self):
        data, _ = make_swiss_roll(1000, random_state=1)
        data_landmark, _ = random_subsample(data, 250)

        dense_case = Roseland(
            GaussianKernel(epsilon=1.25), landmarks=data_landmark, n_svdtriplet=11
        ).fit(data, store_kernel_matrix=True)
        sparse_case = Roseland(
            GaussianKernel(epsilon=1.25, distance=dict(cut_off=1e100)),
            landmarks=data_landmark,
            n_svdtriplet=11,
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

        assert_equal_eigenvectors(dense_case.svdvec_left_, sparse_case.svdvec_left_)

    def test_time_exponent(self):
        data, _ = make_swiss_roll(2000, random_state=1)
        data_landmark, _ = random_subsample(data, 250)

        actual1 = Roseland(
            GaussianKernel(epsilon=1.5),
            n_svdtriplet=5,
            landmarks=data_landmark,
            time_exponent=0,
        ).fit_transform(data)

        actual2 = Roseland(
            GaussianKernel(epsilon=1.5),
            n_svdtriplet=5,
            landmarks=data_landmark,
            time_exponent=1e-14,
        ).fit_transform(data)

        nptest.assert_allclose(np.abs(actual1), np.abs(actual2), rtol=1e-9, atol=1e-13)

    def test_set_target_coords1(self):
        X_swiss_all, _ = make_swiss_roll(n_samples=2000, noise=0, random_state=5)
        data_landmark, _ = random_subsample(X_swiss_all, 400)

        actual_rose = Roseland(
            GaussianKernel(epsilon=2.0), n_svdtriplet=6, landmarks=data_landmark
        ).set_target_coords([1, 5])
        actual = actual_rose.fit_transform(X_swiss_all)

        actual_rose2 = Roseland(
            GaussianKernel(epsilon=2.0), n_svdtriplet=6, landmarks=data_landmark
        )
        actual2 = (
            actual_rose2.fit(X_swiss_all)
            .set_target_coords([1, 5])
            .transform(X_swiss_all)
        )

        expected = Roseland(
            GaussianKernel(epsilon=2.0), n_svdtriplet=6, landmarks=data_landmark
        ).fit_transform(X_swiss_all)

        nptest.assert_allclose(
            np.abs(actual), np.abs(expected[:, [1, 5]]), rtol=1e-10, atol=1e-14
        )
        nptest.assert_allclose(
            np.abs(actual2), np.abs(expected[:, [1, 5]]), rtol=1e-10, atol=1e-14
        )
        nptest.assert_allclose(
            np.abs(actual_rose.svdvec_left_), np.abs(expected), rtol=1e-10, atol=1e-14
        )
        nptest.assert_allclose(
            np.abs(actual_rose2.svdvec_left_), np.abs(expected), rtol=1e-10, atol=1e-14
        )

        self.assertEqual(actual_rose.n_features_out_, 2)
        self.assertEqual(actual_rose2.n_features_out_, 2)
        nptest.assert_equal(
            actual_rose.get_feature_names_out(), np.array(["rose1", "rose5"])
        )
        nptest.assert_equal(
            actual_rose2.get_feature_names_out(), np.array(["rose1", "rose5"])
        )

    def test_set_target_coords2(self):
        X_swiss_all, _ = make_swiss_roll(n_samples=500, noise=0, random_state=5)

        actual_rose = Roseland(GaussianKernel(epsilon=2.0), n_svdtriplet=6).fit(
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
        X_train, color_train = make_swiss_roll(n_samples=4000, noise=0, random_state=5)
        landmarks, _ = random_subsample(X_train, 1000, random_state=1)

        setting = {
            "kernel": GaussianKernel(epsilon=2.7),
            "n_svdtriplet": 7,
            "landmarks": landmarks,
            "alpha": 0,
        }

        rose_embed = Roseland(**setting).fit(X_train)

        indices = [1, 5]
        if plot:
            plot_pairwise_eigenvector(
                eigenvectors=rose_embed.svdvec_left_,
                n=1,
                fig_params=dict(figsize=[6, 6]),
                scatter_params=dict(cmap=plt.cm.Spectral, c=color_train),
            )

        rose_embed_eval_expected = rose_embed.svdvec_left_[:, indices]
        rose_embed_eval_actual = rose_embed.set_target_coords(
            indices=indices
        ).transform(X=X_train)

        # even though the target_coords were set, still all eigenvectors must be
        # accessible
        self.assertEqual(rose_embed.svdvec_left_.shape[1], 7)

        nptest.assert_allclose(
            rose_embed_eval_actual, rose_embed_eval_expected, atol=1e-15
        )

        if plot:
            X_oos, color_oos = make_swiss_roll(n_samples=30000, noise=0, random_state=3)

            f, ax = plt.subplots(2, 3, figsize=(4, 4))
            f.subplots_adjust(hspace=0.326)

            marker = "."
            markersize = 0.2
            ax[0][0].scatter(
                rose_embed_eval_expected[:, 0],
                rose_embed_eval_expected[:, 1],
                s=markersize,
                marker=marker,
                c=color_train,
            )
            ax[0][0].set_title("expected Roseland singular vectors (Fit)")

            ax[0][1].scatter(
                rose_embed_eval_actual[:, 0],
                rose_embed_eval_actual[:, 1],
                s=markersize,
                marker=marker,
                c=color_train,
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

            gh_embed_eval_oos = rose_embed.transform(X_oos)
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
                f"both have same setting chosen_svdvectors={indices}",
            )

            plt.show()

    def test_distance(self):
        X_swiss_all, _ = make_swiss_roll(n_samples=4000, noise=0, random_state=5)
        data_landmark, _ = random_subsample(X_swiss_all, 1000)

        rose = Roseland(
            kernel=GaussianKernel(distance=dict(cut_off=2)),
            landmarks=data_landmark,
        ).fit(X_swiss_all, store_kernel_matrix=True)

        self.assertEqual(rose.kernel.distance.cut_off, 4)
        self.assertTrue(scipy.sparse.issparse(rose.kernel_matrix_))

    @pytest.mark.skip("too expensive")
    def test_speed(self):
        random_state = 1
        rng = np.random.default_rng(random_state)

        n_large_points = int(15000)
        L1, L2 = 4, 3  # width and height of the rectangle
        data = rng.uniform(
            low=(-L1 / 2, -L2 / 2), high=(L1 / 2, L2 / 2), size=(n_large_points, 2)
        )

        setting = {"kernel": GaussianKernel(), "n_svdpairs": 5, "gamma": 0.25}

        rose_embed = Roseland(**setting)

        t1 = time()
        rose_embed.fit(data, store_kernel_matrix=True)
        t2 = time()
        _ = rose_embed.landmarks_.compute_kernel_matrix(data)
        t22 = time()
        _, _, _ = scipy.sparse.linalg.svds(
            rose_embed.kernel_matrix_,
            k=rose_embed.n_svdtriplet,
            which="LM",
        )
        t3 = time()

        print(
            f"kernel+svds: {t22-t2+t3-t2}, fit: {t2-t1}, kernel only: {t22-t2}, svds: {t3-t2}"
        )

        return 1

    def test_landmark_selection(self):
        landmark_random_state = 42
        X_swiss_all, _ = make_swiss_roll(n_samples=500, noise=0, random_state=5)

        data_landmark, _ = random_subsample(
            X_swiss_all, 100, random_state=landmark_random_state
        )

        given_int_case = Roseland(
            GaussianKernel(epsilon=2.0),
            n_svdtriplet=6,
            landmarks=100,
            random_state=landmark_random_state,
        ).fit(X_swiss_all)

        given_float_case = Roseland(
            GaussianKernel(epsilon=2.0),
            n_svdtriplet=6,
            landmarks=0.2,
            random_state=landmark_random_state,
        ).fit(X_swiss_all)

        given_array_case = Roseland(
            GaussianKernel(epsilon=2.0), n_svdtriplet=6, landmarks=data_landmark
        ).fit(X_swiss_all)

        assert_equal_eigenvectors(
            given_array_case.svdvec_left_, given_int_case.svdvec_left_
        )

        assert_equal_eigenvectors(
            given_array_case.svdvec_left_, given_float_case.svdvec_left_
        )

    @pytest.mark.filterwarnings(
        "ignore:X does not have valid feature names, but Roseland was "
        "fitted with feature names"
    )
    def test_types_tsc(self):
        # fit=TSCDataFrame
        _x = np.linspace(0, 2 * np.pi, 20)
        df = pd.DataFrame(
            np.column_stack([np.sin(_x), np.cos(_x)]), columns=["sin", "cos"]
        )

        tsc_data = TSCDataFrame.from_single_timeseries(df=df)

        rose = Roseland(kernel=GaussianKernel(epsilon=0.4), n_svdtriplet=4).fit(
            tsc_data, store_kernel_matrix=True
        )

        self.assertIsInstance(rose.svdvec_left_, TSCDataFrame)

        # insert TSCDataFrame -> output TSCDataFrame
        actual_tsc = rose.transform(tsc_data.iloc[:10, :])
        self.assertIsInstance(actual_tsc, TSCDataFrame)

        # insert np.ndarray -> output np.ndarray  -- raises warning from sklearn
        actual_nd = rose.transform(tsc_data.iloc[:10, :].to_numpy())
        self.assertIsInstance(actual_nd, np.ndarray)

        # check that computation it exactly the same
        nptest.assert_array_equal(actual_tsc.to_numpy(), actual_nd)

    @pytest.mark.filterwarnings(
        "ignore:X has feature names, but Roseland was fitted without feature names"
    )
    def test_types_pcm(self):
        # fit=TSCDataFrame
        _x = np.linspace(0, 2 * np.pi, 20)
        pcm_data = np.column_stack([np.sin(_x), np.cos(_x)])
        tsc_data = TSCDataFrame.from_single_timeseries(
            pd.DataFrame(pcm_data, columns=["sin", "cos"])
        )

        rose = Roseland(kernel=GaussianKernel(epsilon=0.4), n_svdtriplet=4).fit(
            pcm_data, store_kernel_matrix=True
        )

        self.assertIsInstance(rose.svdvec_left_, np.ndarray)

        # insert np.ndarray -> output np.ndarray
        actual_nd = rose.transform(pcm_data[:10, :])
        self.assertIsInstance(actual_nd, np.ndarray)

        # insert TSCDataFrame -> time information is returned, even when during fit no
        # time series data was returned
        actual_tsc = rose.transform(tsc_data.iloc[:10, :])
        self.assertIsInstance(actual_tsc, TSCDataFrame)

        nptest.assert_array_equal(actual_nd, actual_tsc)

        single_sample = tsc_data.iloc[[0], :]
        actual = rose.transform(single_sample)
        self.assertIsInstance(actual, TSCDataFrame)

    def test_sparse_time_series_collection(self):
        X1 = pd.DataFrame(make_swiss_roll(n_samples=250)[0])
        X2 = pd.DataFrame(make_swiss_roll(n_samples=250)[0])

        X = TSCDataFrame.from_frame_list([X1, X2])

        actual_rose = Roseland(
            kernel=GaussianKernel(epsilon=1.25, distance=dict(cut_off=10)),
            n_svdtriplet=6,
            random_state=42,
        )
        actual_result = actual_rose.fit_transform(X, store_kernel_matrix=True)

        expected_rose = Roseland(
            kernel=GaussianKernel(epsilon=1.25, distance=dict(cut_off=10)),
            n_svdtriplet=6,
            random_state=42,
        )

        expected_result = expected_rose.fit_transform(
            X.to_numpy(), store_kernel_matrix=True
        )

        self.assertIsInstance(actual_rose.kernel_matrix_, scipy.sparse.spmatrix)
        self.assertIsInstance(expected_rose.kernel_matrix_, scipy.sparse.spmatrix)
        self.assertIsInstance(actual_result, TSCDataFrame)
        self.assertIsInstance(expected_result, np.ndarray)

        nptest.assert_allclose(
            actual_result.to_numpy(), expected_result, rtol=1e-6, atol=1e-12
        )

    def test_invalid_kernel(self):
        from datafold.pcfold import ConeKernel

        with self.assertRaises(NotImplementedError):
            Roseland(kernel=ConeKernel()).fit(make_swiss_roll(100)[0])
