from datafold.dynfold.kernel import DmapKernelFixed
import unittest
import numpy as np


class TestDiffusionMapsKernelTest(unittest.TestCase):
    def test_is_symmetric01(self):
        # stochastic False

        # Note: in this case the alpha value is ignored
        k1 = DmapKernelFixed(is_stochastic=False, symmetrize_kernel=True)
        self.assertTrue(k1.is_symmetric)

        # No transformation to symmetrize the kernel is required
        self.assertFalse(k1.is_symmetric_transform(is_pdist=True))

        # Because the kernel is not stochastic, the kernel remains symmetric
        k2 = DmapKernelFixed(is_stochastic=False, symmetrize_kernel=False)
        self.assertTrue(k2.is_symmetric)

        # No transformation is required
        self.assertFalse(k1.is_symmetric_transform(is_pdist=True))

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
        # when is_pdist is False
        k1 = DmapKernelFixed(is_stochastic=True, alpha=1, symmetrize_kernel=True)
        self.assertFalse(k1.is_symmetric_transform(is_pdist=False))

        k2 = DmapKernelFixed(is_stochastic=False, alpha=1, symmetrize_kernel=True)
        self.assertFalse(k2.is_symmetric_transform(is_pdist=False))

    def test_missing_row_alpha_fit(self):
        data_X = np.random.rand(100, 5)
        data_Y = np.random.rand(5, 5)

        kernel = DmapKernelFixed(
            epsilon=1, is_stochastic=True, alpha=1, symmetrize_kernel=False
        )

        _, _, row_sums_alpha = kernel(X=data_X)

        with self.assertRaises(RuntimeError):
            kernel(X=data_X, Y=data_Y)

        # No error:
        kernel_kwargs = {"row_sums_alpha_fit": row_sums_alpha}
        kernel(X=data_X, Y=data_Y, kernel_kwargs=kernel_kwargs)


if __name__ == "__main__":
    TestDiffusionMapsKernelTest().test_missing_row_alpha_fit()
