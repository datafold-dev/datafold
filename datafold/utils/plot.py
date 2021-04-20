#!/usr/bin/env python3

import warnings
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from datafold._decorators import warn_experimental_function


def plot_eigenvalues(
    eigenvalues: np.ndarray,
    *,
    plot_unit_circle: bool = False,
    semilogy: bool = False,
    ax=None,
    subplot_kwargs: Optional[Dict[str, object]] = None,
    plot_kwargs: Optional[Dict[str, object]] = None,
):
    """Plots eigenvalue distribution.

    Parameters
    ----------
    eigenvalues
        Complex or real eigenvalues.

    plot_unit_circle
        If True, include unit circle on complex plane.

    semilogy
        Enable logarithmic y-axis. Parameter is ignored for complex eigenvalues.

    ax
        Plot in existing matplotlib axes object. ``subplot_kwargs`` are ignored then.

    subplot_kwargs
        Keyword arguments passed to ``plt.subplot``.

    plot_kwargs
        Keyword arguments passed to ``ax.plot(**plot_kwargs)``

    Returns
    -------
    """

    if ax is None:
        _, ax = plt.subplots(**({} if subplot_kwargs is None else subplot_kwargs))

    plot_kwargs = plot_kwargs or {}
    plot_kwargs.setdefault("marker", "+")
    plot_kwargs.setdefault("linewidth", 0)

    if eigenvalues.dtype == complex:

        ax.plot(np.real(eigenvalues), np.imag(eigenvalues), **plot_kwargs)

        if plot_unit_circle:
            circle_values = np.linspace(0, 2 * np.pi, 3000)
            ax.plot(np.cos(circle_values), np.sin(circle_values), "-", color="gray")
            ax.set_aspect("equal")

        with plt.rc_context(rc={"text.usetex": True}):
            ax.set_xlabel("$\\Re(\\lambda)$")
            ax.set_ylabel("$\\Im(\\lambda)$")

    elif eigenvalues.dtype == float:

        if plot_unit_circle:
            warnings.warn(
                "eigenvalues are real-valued, 'plot_unit_circle=True' is ignored"
            )

        eigenvalues = np.sort(eigenvalues.copy())[::-1]

        _ylabel_text = "eigenvalue $\\lambda$"
        if semilogy:
            ax.semilogy(np.arange(len(eigenvalues)), eigenvalues, **plot_kwargs)
            _ylabel_text = _ylabel_text + " (log scale)"
        else:
            ax.plot(np.arange(len(eigenvalues)), eigenvalues, **plot_kwargs)

        with plt.rc_context(rc={"text.usetex": True}):
            ax.set_ylabel(_ylabel_text)

        ax.set_xlabel("index eigenvalue")

    return ax


def plot_eigenvalues_time(
    time_values: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    system_type="flowmap",
    delta_time: Optional[float] = None,
    ax=None,
    subplots_kwargs=None,
    plot_kwargs=None,
):
    r"""Plot eigenvalues over time.

    The eigenvalues :math:`\lambda_k` (y-axis) are plot with respect to the
    ``system_type``:

    * "flowmap" - requires to set ``delta_time``
        .. math::
            \vert \lambda_k^{t / \Delta t} \vert

    * "differential"
        .. math::
            \exp( \lambda_k \cdot t )

    For linear dynamical system, the plot is informative to show the
    eigenvalues contribution over the time horizon of ``time_values``. Eigenpairs with
    :math:`\lambda < 1` have a decaying contribution over time, eigenpairs with
    :math:`\lambda > 1` lead to exponential growth and prohibit long time term
    predictions.

    Parameters
    ----------

    time_values
        The time values on the x-axis.

    eigenvalues
        Eigenvalues to plot on the y-axis.

    system_type
        There are two modes to describe a linear dynamical system (see also
        :py:class:`.LinearDynamicalSystem`)
        * "flowmap" - discrete system
        * "differential" - continuous system

    delta_time
        Reference system time of type "flowmap".

    ax
        Matplotlib ``Axes`` object to plot in.

    subplots_kwargs
        Keyword arguments passed to ``matplotlib.pyplot.subpplot``. Ignored if
        ``ax`` is not ``None``.

    plot_kwargs
        Keyword arguments passed to ``ax.plot(**kwargs)``.

    Returns
    -------
    matplotlib axes object

    """

    n_timesteps = len(time_values)

    if system_type == "flowmap":
        if delta_time is None:
            raise ValueError(
                "For 'system_type=flowmap', the parameter 'delta_time' must be provided."
            )

        values_matrix = np.abs(
            np.power(
                np.outer(eigenvalues, np.ones(n_timesteps)),
                time_values[np.newaxis, :] / delta_time,
            )
        )
    elif system_type == "differential":
        values_matrix = np.abs(np.exp(np.outer(eigenvalues, time_values)))
    else:
        raise ValueError(
            f"system_type={system_type} not known. Choose between "
            f"[flowmap, differential]."
        )

    if ax is None:
        f, ax = plt.subplots(**({} if subplots_kwargs is None else subplots_kwargs))

    plot_kwargs = plot_kwargs or {}
    plot_kwargs.setdefault("linestyle", "-")
    plot_kwargs.setdefault("color", "black")

    ax.plot(time_values, values_matrix.T, **plot_kwargs)

    with plt.rc_context(rc={"text.usetex": True}):
        ax.set_xlabel("time ($t$)")

        if system_type == "flowmap":
            ax.set_ylabel("$\\vert \\lambda^{t / \\Delta t} \\vert$")
        else:  # continuous
            ax.set_ylabel("$\\vert \\exp(\\lambda \cdot t) \\vert$")

    return ax


def plot_pairwise_eigenvector(
    eigenvectors: np.ndarray,
    n: int,
    idx_start=0,
    scatter_params: Optional[Dict] = None,
    fig_params: Optional[Dict] = None,
) -> None:
    """Plot scatter plot of n-th eigenvector on x-axis and remaining eigenvectors on
    y-axis.

    Parameters
    ----------
    eigenvectors
        Eigenvectors of the kernel matrix of shape `(n_samples, n_eigenvectors)`. The
        eigenvectors are assumed to be sorted by the index.

    n
        eigenvector index (in columns) to plot on x-axis

    idx_start
        is the eigenvector index of the first columns (useful when trivial constant
        eigenvectors are removed before, then set `idx_start=1`).

    colors
        visualize the points

    scatter_params
        keyword arguments handled to  `matplotlib.pyplot.scatter()`

    fig_params
        keyword arguments handled to `matplotlib.pyplot.figure()`
    """

    eigenvectors = np.asarray(eigenvectors)

    # -1 because the trivial case "n versus n" is skipped
    n_eigenvectors = eigenvectors.shape[1] - 1

    fig_params = {} if fig_params is None else fig_params

    ncols = fig_params.pop("ncols", 2)
    nrows = fig_params.pop("nrows", int(np.ceil(n_eigenvectors / 2)))
    sharex = fig_params.pop("sharex", True)
    sharey = fig_params.pop("sharey", True)

    f, ax = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, **fig_params
    )

    correct_one = 0

    for i, idx_eigvec in enumerate(range(n_eigenvectors + 1)):

        if i == n:
            correct_one = 1
            continue
        else:
            i = i - correct_one

        current_row = i // ncols
        current_col = i - current_row * ncols

        if nrows == 1:
            _ax = ax[current_col]
        elif ncols == 1:
            _ax = ax[current_row]
        else:
            _ax = ax[current_row, current_col]

        _ax.scatter(
            eigenvectors[:, n],
            eigenvectors[:, idx_eigvec],
            **{} if scatter_params is None else scatter_params,
        )

        _ax.set_title(
            r"$\Psi_{{{}}}$ vs. $\Psi_{{{}}}$".format(
                n + idx_start, idx_eigvec + idx_start
            )
        )


@warn_experimental_function
def plot_scales(pcm, scale_range=(1e-5, 1e3), n_scale_tests=20) -> None:
    """Plots for varying scales.

    .. warning::

    Parameters
    ----------

    pcm
        point cloud manifold

    scale_range
        lower and upper limit of scale

    n_scale_tests
        number of points
    """

    np.random.seed(1)

    scales = np.exp(
        np.linspace(np.log(scale_range[0]), np.log(scale_range[1]), n_scale_tests)
    )
    scale_sum = np.zeros_like(scales)

    distance_matrix = pcm.compute_distance_matrix()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    save_eps = pcm.kernel.epsilon

    for i, scale in enumerate(scales):

        pcm.kernel.epsilon = scale
        kernel_matrix_scale = pcm.kernel.eval(distance_matrix=distance_matrix)
        kernel_sum = kernel_matrix_scale.sum()

        scale_sum[i] = kernel_sum / (kernel_matrix_scale.shape[0] ** 2)

    # ax.loglog(scales, scale_sum, 'k-', label='points')
    pcm.kernel.epsilon = save_eps

    gradient = np.exp(
        np.gradient(np.log(scale_sum), np.log(scales)[1] - np.log(scales)[0])
    )
    ax.semilogx(scales, gradient, "k-", label="points")

    igmax = np.argmax(gradient)

    eps = scales[igmax]
    dimension = gradient[igmax] - 1 / 2

    ax.semilogx(
        [scales[igmax], scales[igmax]],
        [np.min(gradient), np.max(gradient)],
        "r-",
        label=r"max at $\epsilon=%.5f$" % (eps),
    )
    ax.semilogx(
        [np.min(scales), np.max(scales)],
        [gradient[igmax], gradient[igmax]],
        "b-",
        label=r"dimension $\approx %.1f$" % (dimension),
    )

    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\mathbb{E}_\epsilon$")
    # ax.loglog(scales, 1*scales, 'r--', label='dim=1')
    # ax.loglog(scales, 2*scales, 'g--', label='dim=2')
    ax.legend()
    fig.tight_layout()


if __name__ == "__main__":
    plot_pairwise_eigenvector(eigenvectors=np.random.rand(500, 10), n=0, idx_start=1)
    plt.show()
