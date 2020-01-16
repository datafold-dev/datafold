"""Module for plotting diffusion maps and associated diagnostics.

"""

__all__ = [
    "plot_diffusion_maps",
    "plot_spectrum",
    "plot_results",
    "plot_l_vs_epsilon",
    "plot_l_vs_epsilon_heuristic",
    "plot_eps_vs_error",
    "plot_number_eigenvectors_vs_error",
]

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from datafold.dynfold.diffusion_maps import DiffusionMaps


def _get_rows_and_columns(num_plots: int) -> Tuple[int, int]:
    # Get optimal number of rows and columns to display figures.

    # num_plots - Number of subplots
    #
    # rows - Optimal number of rows.
    # cols - Optimal number of columns.

    if num_plots <= 10:
        layouts = {
            1: (1, 1),
            2: (1, 2),
            3: (1, 3),
            4: (2, 2),
            5: (2, 3),
            6: (2, 3),
            7: (2, 4),
            8: (2, 4),
            9: (3, 9),
            10: (2, 5),
        }
        rows, cols = layouts[num_plots]
    else:
        rows = int(np.ceil(np.sqrt(num_plots)))
        cols = rows

    return rows, cols


def plot_spectrum(eigenvalues: np.ndarray) -> None:
    """Plot spectrum.

    Figure shows the modulus of the spectrum
    of the kernel in the diffusion map calculation.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of the kernel matrix.

    """

    plt.figure(1)
    plt.step(np.arange(1, eigenvalues.shape[0]), np.abs(eigenvalues[1:]))
    plt.xticks(range(1, eigenvalues.shape[0]))
    plt.xlabel("Eigenvalue index")
    plt.ylabel("| Eigenvalue |")
    plt.title("Eigenvalues")


def plot_data_colored_by_diffusion_maps(
    data: np.ndarray, eigenvectors: np.ndarray
) -> None:
    """Plot colored data.

    The plotted Figure shows the original (2D) data colored by the value of each diffusion
    map.

    Parameters
    ----------
    data : np.ndarray
        Original (or downsampled) data set.
    eigenvectors : np.ndarray
        Eigenvectors of the kernel matrix. The zeroth axis indexes each vector.

    """

    x = data[:, 0]
    y = data[:, 1]
    num_eigenvectors = max(eigenvectors.shape[0] - 1, 10)
    plt.figure(2)
    rows, cols = _get_rows_and_columns(num_eigenvectors)

    for k in range(1, eigenvectors.shape[0]):
        plt.subplot(rows, cols, k)
        plt.scatter(x, y, c=eigenvectors[k, :], cmap="RdBu_r", rasterized=True)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.axis("off")
        plt.title("$\\psi_{{{}}}$".format(k))


def plot_number_eigenvectors_vs_error(
    points: np.ndarray,
    values: np.ndarray,
    epsilon: float,
    k: int = 4,
    number_of_eigenpairs: np.ndarray = np.arange(2, 11),
) -> None:
    """Plot number of eigenvectors vs. k-fold interpolation error.

    Parameters
    ----------
    points : np.ndarray
        Training points.
    values : np.ndarray
        Training targets values.
    epsilon : float
        Bandwidth of kernel for interpolation.
    k: int = 4
        Number of folds in cross-validation.
    number_of_eigenpairs: numpy.ndarray = np.arange(1, 11)
        Upper bound for the number of eigenpairs to evaluate.
    diffusion_maps_options: Optional[Dict] = None
        Options to pass to the diffusion maps used for interpolation. The parameter
        'num_eigenpairs' is changed according to values in 'number_of_eigenpairs'.
    """

    cv_errors = np.zeros(number_of_eigenpairs.shape[0])
    for n in number_of_eigenpairs:
        # TODO: broken, should use the GridSearchCV from sklearn
        # cv_error = k_fold_error(points, values, 0.7, epsilon, num_eigenpairs=n, k=k)
        # cv_errors[n - 1] = cv_error
        pass

    plt.title(f"Interpolation error ({k}-fold-cross-validated)", size=15)
    plt.xlabel("number of eigenpairs", size=12)
    plt.ylabel(f"{k}-fold-cv error", size=12)
    plt.plot(number_of_eigenpairs, cv_errors)
    plt.show()


def plot_eps_vs_error(
    points: np.array,
    values: np.array,
    epsilons: np.ndarray,
    num_eigenpairs: int,
    k: int = 4,
) -> None:
    """Plot epsilon vs. error.

    This function loglog-plots the k-cross-validated interpolation error for a range of
    epsilons.

    Parameters
    ----------
    points: np.ndarray
        Training points.
    values: np.ndarray
        Training targets.
    epsilons: np.ndarray
        Epsilons for which to plot the error. Array must be one dimensional.
    num_eigenpairs: int
        Number of eigenpairs to use in gometric harmonics.
    k: int
        Number of folds in cross-validation
    """

    cv_errors = np.zeros(len(epsilons))
    for i, eps in enumerate(epsilons):
        cv_errors[i] = k_fold_error(points, values, 0.7, eps, num_eigenpairs, k)

    plt.rc("text", usetex=True)
    plt.title(fr"$\varepsilon$ vs. {k}-fold-cross-validated error", size=15)
    plt.xlabel(r"$\varepsilon$", size=12)
    plt.ylabel("Error", size=12)
    plt.loglog(epsilons, cv_errors)
    plt.show()


def _plot_eigenvectors(
    eigenvector1: np.ndarray,
    eigenvector2: np.ndarray,
    colors: np.ndarray,
    ax: plt.Axes,
    title: str,
) -> None:
    # Plots data parametrized by two diffusion map coordinates.

    # eigenvector1 - First eigenvector of kernel matrix.
    # eigenvector2 - Second eigenvector of kernel matrix.
    # title: str     Title of plot. Usually contains something like
    #                r"$\Psi_1$ vs. $\Psi_{{{}}}$".format(i).

    ax.scatter(
        eigenvector1, eigenvector2, 0.1, c=colors, marker=".", cmap=plt.cm.Spectral
    )
    ax.axis("off")
    ax.set_title(title)


def plot_eigenvectors_n_vs_all(
    eigenvectors: np.ndarray, n: int, colors: Optional[np.ndarray] = None
) -> None:
    """Comparison of n-th eigenvector on x-axis and remaining eigenvectors on y-axis.

    Parameters
    ----------
    eigenvectors : np.ndarray
        Eigenvectors of the kernel matrix. The zeroth axis indexes each vector.
    n: int
        Eigenvector to plot on x-axis.
    colors:
        Colors for visualization of points.
    """

    fig = plt.figure(figsize=[6, 12])

    plot_eigenvectors = np.min([eigenvectors.shape[0], 10])
    is_even = np.mod(plot_eigenvectors, 2) == 0

    for i in range(2, plot_eigenvectors):
        # TODO: this plots also the trivial case i == n
        _plot_eigenvectors(
            eigenvectors[n, :],
            eigenvectors[i, :],
            colors,
            fig.add_subplot(plot_eigenvectors // 2 - is_even, 2, i - 1),
            r"$\Psi_{{{}}}$ vs. $\Psi_{{{}}}$".format(n, i),
        )
    # plt.tight_layout()


def plot_results(
    data: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    colors: np.ndarray,
    spectrum: bool = True,
    colored_data: bool = True,
    diffusion_maps_2d: bool = True,
) -> None:
    """Plot overview of the diffusion map.

    Plots up to three figures:

    1. spectrum of the kernel in the diffusion map
    2. original data (only first two coordinates) colored by the value of each diffusion
       map coordinate
    3. data transformed by the first two (non-trivial) diffusion maps

    Parameters
    ----------
    data : np.ndarray
        Original (or downsampled) data set.
    eigenvalues : np.ndarray
        Eigenvalues of the kernel matrix.
    eigenvectors : np.ndarray
        Eigenvectors of the kernel matrix. The zeroth axis indexes each
        vector.
    colors: np.ndarray
        Coloring in 1 vs. all plot.
    spectrum: bool=True
        Option for the first figure.
    colored_data: bool=True
        Option for the second figure.
    diffusion_maps_2d: bool=True
        Option for the third figure.
    """

    # plt.rc('text', usetex=True)

    if spectrum:
        plot_spectrum(eigenvalues)

    if colored_data:
        plot_data_colored_by_diffusion_maps(data, eigenvectors)

    if diffusion_maps_2d:
        plot_eigenvectors_n_vs_all(eigenvectors, 1, colors)

    # plt.tight_layout()
    plt.show()


def plot_diffusion_maps(
    data: np.ndarray, dmaps: DiffusionMaps, colors: np.ndarray = "b"
) -> None:
    """Plot diffusion maps.

    High-level interface to plot_results.

    Parameters
    ----------
    data : np.ndarray
        Original (or downsampled) data set.
    dmaps: DiffusionMaps
        DiffusionMaps of the data.
    colors: np.ndarray='b'
        Color map for the diffusion map coordinates.
    """

    plot_results(data, dmaps.eigenvalues, dmaps.eigenvectors, colors)


def _plot_secant(points: np.ndarray, i: int, ax: plt.Axes) -> None:
    # Plot secant line at index.
    # Plot the approximate tangent/secant line of the curve given by the points array at
    # the given point.

    # points:     Function values.
    # i: int      Plot tangent of curve at this index into points array.
    # ax: Axes    The line is added to the axes.

    p_0 = points[i - 1, :]
    p_2 = points[i + 1, :]
    x = np.array([p_0[0], p_2[0]])
    y = np.array([p_0[1], p_2[1]])
    ax.plot(x, y, label=r"secant at $\varepsilon$ = {}".format(x[0]))


def plot_l_vs_epsilon(
    data: np.ndarray,
    epsilons_coarse: List[float] = None,
    epsilons_fine: List[float] = None,
    epsilon_magnitude: int = None,
    epsilon: float = None,
    lut_km_coarse: "LookupKernelMatsEpsilon" = None,
    lut_km_fine: "LookupKernelMatsEpsilon" = None,
) -> None:
    """Plot L(epsilon).

    Plot L(epsilon) (Master Thesis of Bah (2008)) for different epsilons. Extended by
    the optional insertion of a secant line in the neighborhood of an epsilon.

    Parameters
    ----------
    data : np.ndarray
        Original (or downsampled) data set.
    epsilons_coarse: List[float] = None
        Coarse range of epsilons for using float keys into kernel matrices.
    epsilons_fine: List[float] = None
        Fine range of epsilons for producing smooth-looking plots.
    epsilon_magnitude: int = None
        Plot secant line at this magnitude. If not provided, no secant line will be
        plotted.
    epsilon: float = None
        Plot vertical line at this epsilon. If not provided, no vertical line will be
        plotted.
    lut_km_coarse: LookupKernelMatsEpsilon = None
        Option to provide precomputed kernel matrices to avoid re-computation.
        Coarse-scaled for float indexing.
    lut_km_fine: LookupKernelMatsEpsilon = None
        Option to provide precomputed kernel matrices to avoid re-computation.
        Fine-scaled for smooth-looking plots.
    """

    plt.rc("text", usetex=True)

    fig, ax = plt.subplots(figsize=[8, 6])

    if epsilons_coarse:
        if lut_km_coarse:
            lut_km_coarse = LookupKernelMatsEpsilon(data, epsilons_coarse)
            ls_coarse = _ls_of_kernel_matrices(lut_km_coarse)
            points = np.array([np.array(epsilons_coarse), ls_coarse]).transpose()
            if epsilon_magnitude is not None:
                index_epsilon = epsilons_coarse.index(
                    round(10 ** epsilon_magnitude, -epsilon_magnitude)
                )
                _plot_secant(points, index_epsilon, ax)

    if epsilons_fine:
        if not lut_km_fine:
            lut_km_fine = LookupKernelMatsEpsilon(data, epsilons_fine)
        ls_fine = _ls_of_kernel_matrices(lut_km_fine)
        ax.plot(epsilons_fine, ls_fine, label=r"L versus $\varepsilon$")
        if epsilon:
            ax.vlines(
                [epsilon],
                ymin=ls_fine.min(),
                ymax=ls_fine.max(),
                linestyles="dashed",
                color="r",
                label=r"$\varepsilon$ = {}".format(epsilon),
            )

    ax.set_xlabel(r"$\varepsilon$", size=15)
    ax.set_ylabel(r"L($\varepsilon$)", size=15)
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.legend(loc=4, prop={"size": 13})
    plt.show()


def log_midpoint_rule(data: np.ndarray) -> float:
    """Compute the first key at which the data half its range plus min.

    Heuristic to find a good epsilon. The key of the epsilon, at which the value of
    data  is half the range of data plus its minimum value.

    Parameters
    ----------
    data : np.ndarray
        Must be one dimensional.
    """
    # TODO: heuristics to find a suitable epsilon should probably be at DMAP module

    dat_min = data.min()
    dat_max = data.max()
    data_range = dat_max - dat_min
    log_midpoint = (np.log(dat_min) + np.log(data_range)) / 2.0
    return float(log_midpoint)


def plot_l_vs_epsilon_heuristic(
    data: np.ndarray, epsilon_min_magnitude: int, epsilon_max_magnitude: int
) -> None:
    """
    Plot L(epsilon) (Master Thesis of Bah (2008)) for different epsilons. Extended by the
    insertion of a vertical line at an epsilon which approximates the epsilon, at which
    L(epsilon) is at the log of the midpoint of its value range. This is a heuristic
    for an optimal epsilon.

    Parameters
    ----------
    data : np.ndarray
        Data set.
    epsilon_min_magnitude: int
        Least magnitude of epsilon to analyze.
    epsilon_max_magnitude: int
        Highest magnitude of epsilon to analyze.
    """

    if epsilon_max_magnitude <= epsilon_min_magnitude:
        raise ValueError(
            "epsilon_max_magnitude must be larger than epsilon_min_magnitude"
        )

    epsilons_fine = (
        10 ** np.arange(epsilon_min_magnitude, epsilon_max_magnitude, 0.25)
    ).tolist()
    lut_km_fine = LookupKernelMatsEpsilon(data, epsilons_fine)

    ls = _ls_of_kernel_matrices(lut_km_fine)
    log_midpoint = log_midpoint_rule(ls)
    index_epsilon = _closest_to(np.log(ls), log_midpoint)[0]
    epsilon_1 = epsilons_fine[index_epsilon]

    epsilon = epsilon_1
    plot_l_vs_epsilon(
        data, epsilons_fine=epsilons_fine, lut_km_fine=lut_km_fine, epsilon=epsilon
    )


def _ls_of_kernel_matrices(kernel_matrices: "LookupKernelMatsEpsilon") -> np.ndarray:
    # Compute L values of kernel matrices.
    # This function computes the L values for kernel matrices provided by a lookup table f
    # or kernel matrices.

    # Kernel matrices to compute L values from as a lut. Axis 0 must index each kernel
    # matrix.
    return np.array([kernel_matrices(eps) for eps in kernel_matrices.epsilons]).sum(
        axis=(1, 2)
    )


def _closest_to(a: List[float], i: float):
    # Return closest value to `i` in `a` and the corresponding index. Useful when
    # looking up values when the key is safe but the key-range is rounded differently.
    return min(enumerate(a), key=lambda x: abs(x[1] - i))


class LookupDmapsEpsilon(object):
    """Look up for diffusion maps using epsilon as key and diffusion maps as
    value and stores them.

    Attributes
    ----------
    epsilons : List[float]
        array of epsilons. This corresponds to the range of the keys
    """

    def __init__(
        self, data: np.ndarray, epsilons: List[float], normalize_kernel: bool = True
    ) -> None:
        """Compute diffusion maps for each epsilon.

        This function computes diffusion maps for different epsilons.

        Parameters
        ----------
        data : np.ndarray
            Data set to analyze. Its 0-th axis must index each data point.
        epsilons: List[float]
            Values of epsilon to store the dmaps for each of them.
        normalize_kernel: bool=True
            Determines wheter the kernel is normalized.
        """

        self._epsilons = (
            epsilons  # For each epsilon the corresponding diffusion maps object.
        )

        self._dmaps: List[DiffusionMaps] = []
        for eps in epsilons:
            try:
                dmap = DiffusionMaps(
                    epsilon=eps, num_eigenpairs=11, is_stochastic=normalize_kernel
                ).fit(data)
                self._dmaps.append(dmap)
            except Exception as e:
                print(eps)
                raise e
        self.eps_min = min(epsilons)
        self.eps_max = max(epsilons)

    def __call__(self, epsilon: float) -> DiffusionMaps:
        """Provide Diffusion Maps corresponding to epsilon.
        This function returns diffusion maps for an epsilon provided by the caller.

        Parameters
        ----------
        epsilon : np.ndarray
            Bandwidth of the kernel of the diffusion maps to return.
        """

        eps_index = self._epsilons.index(epsilon)
        return self._dmaps[eps_index]


class LookupKernelMatsEpsilon(object):
    """Look up for diffusion distances using epsilon as key and diffusion distances as
    value and stores them.

    Attributes
    ----------
    epsilons : np.ndarray
        array of epsilons. This corresponds to the range of the keys.
    """

    def __init__(self, data: np.ndarray, epsilons: List[float]) -> None:
        """Compute diffusion maps for each epsilon.

        This function computes diffusion maps for different epsilons.

        Parameters
        ----------
        data : np.ndarray
            Data set to analyze. Its 0-th axis must index each data point.
        """

        self.epsilons = epsilons

        # For each epsilon the corresponding kernel matrix.
        self._kernel_mats = [
            KernelBase.apply_kernel(data, epsilon) for epsilon in self.epsilons
        ]

    def __call__(self, epsilon: float) -> np.ndarray:
        """Provide kernel matricees of diffusion maps corresponding to epsilon.

        This function returns the kernel matrix of diffusion maps for an epsilon
        provided by the caller.

        Parameters
        ----------
        epsilon: np.ndarray
            Bandwidth of the kernel of the diffusion maps to return.
        """

        eps_index = self.epsilons.index(epsilon)
        return self._kernel_mats[eps_index]
