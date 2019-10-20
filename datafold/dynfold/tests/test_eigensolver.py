#!/usr/bin/env python

import unittest

import numpy as np
import scipy.sparse

# fails if GPGPU is not set up (ARPACK and CUDA have to be installed)
import datafold.dynfold.gpu_eigensolver as gpu_eigensolver
from datafold.dynfold.kernel import KernelMethod


class GPUEigensolverTestCase(unittest.TestCase):
    def test_identity_matrix(self):
        np.random.seed(0)
        A = np.random.randn(100, 100)
        Q, R = np.linalg.qr(A)

        matrix = scipy.sparse.csr_matrix(R)
        ew_cpu, ev_cpu = KernelMethod.cpu_eigensolver(matrix, 2)
        ew_gpu, ev_gpu = gpu_eigensolver.eigensolver(matrix, 2)

        assert np.allclose(ew_cpu, ew_gpu), (ew_cpu, ew_gpu)

        assert np.allclose(matrix @ ev_cpu[0, :], ew_cpu[0] * ev_cpu[0, :])
        assert np.allclose(matrix @ ev_cpu[1, :], ew_cpu[1] * ev_cpu[1, :])

        assert np.allclose(matrix @ ev_gpu[0, :], ew_gpu[0] * ev_gpu[0, :])
        assert np.allclose(matrix @ ev_gpu[1, :], ew_gpu[1] * ev_gpu[1, :])


if __name__ == '__main__':
    unittest.main()