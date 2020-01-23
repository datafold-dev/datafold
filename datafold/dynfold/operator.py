#!/usr/bin/env python3

import numpy as np
from datafold.dynfold.diffusion_maps import DiffusionMaps
from datafold.pcfold.timeseries.transform import TSCTransformMixIn
from datafold.utils.maths import mat_dot_diagmat
import datafold.pcfold.pointcloud as pcfold


# TODO: do not inherit from GeometricHamronics, replace with DiffusionMaps?
# TODO: make it a TSCTransformer
# TODO: inherit from KernelMethod (or DiffusionMaps?)


class TSCEigfuncInterpolator(TSCTransformMixIn):
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

        super(TSCEigfuncInterpolator, self).__init__(
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
