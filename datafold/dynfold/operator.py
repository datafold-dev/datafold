#!/usr/bin/env python3

import numpy as np
from datafold.dynfold.outofsample import GeometricHarmonicsInterpolator
from datafold.utils.maths import mat_dot_diagmat
import datafold.pcfold.pointcloud as pcfold


class KernelEigenfunctionInterpolator(GeometricHarmonicsInterpolator):
    # TODO: use (*args, **kwargs) and simply note that it is the same as in
    #  GeometricHarmonicsInterpolator?

    VALID_OPERATOR_NAMES = (
        "laplace_beltrami",
        "fokker_plank",
        "graph_laplacian",
        "rbf",
    )

    def __init__(
        self,
        epsilon: float = 1.0,
        num_eigenpairs: int = 11,
        cut_off: float = np.inf,
        is_stochastic: bool = False,
        alpha: float = 1,
        symmetrize_kernel=True,
        use_cuda=False,
        dist_backend="guess_optimal",
        dist_backend_kwargs=None,
    ):

        super(KernelEigenfunctionInterpolator, self).__init__(
            epsilon=epsilon,
            num_eigenpairs=num_eigenpairs,
            cut_off=cut_off,
            is_stochastic=is_stochastic,
            alpha=alpha,
            symmetrize_kernel=symmetrize_kernel,
            use_cuda=use_cuda,
            dist_backend=dist_backend,
            dist_backend_kwargs=dist_backend_kwargs,
        )

    @classmethod
    def from_name(cls, name, **kwargs):

        if name == "laplace_beltrami":
            eigfunc_interp = cls.laplace_beltrami(**kwargs)
        elif name == "fokker_planck":
            eigfunc_interp = cls.fokker_planck(**kwargs)
        elif name == "graph_laplacian":
            eigfunc_interp = cls.graph_laplacian(**kwargs)
        elif name == "rbf":
            eigfunc_interp = cls.rbf(**kwargs)
        else:
            raise ValueError(
                f"name='{name}' not known. Choose from {cls.VALID_OPERATOR_NAMES}"
            )

        if name not in cls.VALID_OPERATOR_NAMES:
            raise NotImplementedError(
                f"This is a bug. name={name} each name has to be "
                f"listed in VALID_OPERATOR_NAMES"
            )

        return eigfunc_interp

    @classmethod
    def laplace_beltrami(
        cls, epsilon=1.0, num_eigenpairs=10, **kwargs,
    ):
        return cls(
            epsilon=epsilon,
            num_eigenpairs=num_eigenpairs,
            is_stochastic=True,
            alpha=1.0,
            **kwargs,
        )

    @classmethod
    def fokker_planck(
        cls, epsilon=1.0, num_eigenpairs=10, **kwargs,
    ):
        return cls(
            epsilon=epsilon,
            num_eigenpairs=num_eigenpairs,
            is_stochastic=True,
            alpha=0.5,
            **kwargs,
        )

    @classmethod
    def graph_laplacian(
        cls, epsilon=1.0, num_eigenpairs=10, **kwargs,
    ):
        return cls(
            epsilon=epsilon,
            num_eigenpairs=num_eigenpairs,
            is_stochastic=True,
            alpha=0.0,
            **kwargs,
        )

    @classmethod
    def rbf(
        cls, epsilon=1.0, num_eigenpairs=10, **kwargs,
    ):
        return cls(
            epsilon=epsilon,
            num_eigenpairs=num_eigenpairs,
            is_stochastic=False,
            **kwargs,
        )

    def _precompute_aux(self) -> None:
        # NOTE: this in the special case of
        # target functions == own kernel eigenfunctions
        # this corresponds to the normal Nyström extension
        # \psi(x) = 1/Lambda @ kernel(x, .) @ \psi
        self._aux = mat_dot_diagmat(
            self.eigenvectors_.T, np.reciprocal(self.eigenvalues_)
        )

    def fit(
        self, X: np.ndarray, y=None, **fit_params
    ) -> "KernelEigenfunctionInterpolator":
        # y is only there to provide the same function from the base, but y is never
        # needed...
        if y is not None:
            raise ValueError(
                "Do not provide y for GeometricHarmonicsFunctionBasis. The target values "
                "are the eigenvectors computed."
            )

        X = self._check_X_y(X)

        self.X = pcfold.PCManifold(
            data=X,
            kernel=self.kernel_,
            cut_off=self.cut_off,
            dist_backend=self.dist_backend,
            **self.dist_backend_kwargs,
        )

        (
            self.kernel_matrix_,
            _basis_change_matrix,
            self._row_sums_alpha,
        ) = self.X.compute_kernel_matrix()

        self.eigenvalues_, self.eigenvectors_ = self.solve_eigenproblem(
            self.kernel_matrix_, _basis_change_matrix, self.use_cuda
        )

        if self.kernel_.is_symmetric_transform(is_pdist=True):
            self.kernel_matrix_ = self._unsymmetric_kernel_matrix(
                kernel_matrix=self.kernel_matrix_,
                basis_change_matrix=_basis_change_matrix,
            )

        self.y = self.eigenvectors_.T  # TODO: see #44

        self._precompute_aux()
        return self

    def score(self, X, y, sample_weight=None, multioutput="uniform_average") -> float:

        return super(KernelEigenfunctionInterpolator, self).score(
            X=X, y=y, sample_weight=sample_weight, multioutput=multioutput
        )
