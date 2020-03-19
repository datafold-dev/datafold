#!/usr/bin/env python3

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_eigenvalues(
    eigenvalues, plot_unit_circle=False, semilogy=False, subplot_kwargs=None
):

    f, ax = plt.subplots(**({} if subplot_kwargs is None else subplot_kwargs))

    if eigenvalues.dtype == np.complexfloating:
        ax.plot(np.real(eigenvalues), np.imag(eigenvalues), "+")

        if plot_unit_circle:
            circle_values = np.linspace(0, 2 * np.pi, 3000)
            ax.plot(np.cos(circle_values), np.sin(circle_values), "b-")
            ax.set_aspect("equal")

        with plt.rc_context(rc={"text.usetex": True}):
            ax.set_xlabel("$\\Re(\\lambda)$")
            ax.set_ylabel("$\\Im(\\lambda)$")

    elif eigenvalues.dtype == np.floating:

        if plot_unit_circle:
            warnings.warn(
                "eigenvalues are real-valued, 'plot_unit_circle=True' is ignored"
            )

        eigenvalues = np.sort(eigenvalues.copy())[::-1]

        _ylabel_text = "eigenvalue $\\lambda$"
        if semilogy:
            ax.semilogy(np.arange(len(eigenvalues)), eigenvalues, "+")
            _ylabel_text = _ylabel_text + " (log scale)"
        else:
            ax.plot(np.arange(len(eigenvalues)), eigenvalues, "+")

        with plt.rc_context(rc={"text.usetex": True}):
            ax.set_ylabel(_ylabel_text)

        ax.set_xlabel("index eigenvalue")

    return ax


def plot_eigenvalues_time(eigenvalues, n_timesteps, subplot_kwargs=None):
    vander_matrix = np.vander(np.abs(eigenvalues), n_timesteps, increasing=True)

    # TODO: for the handling of None-kwargs there could be an datafold.utils for more
    #  readability
    f, ax = plt.subplots(**({} if subplot_kwargs is None else subplot_kwargs))
    ax.plot(np.arange(n_timesteps), vander_matrix.T, "-")

    with plt.rc_context(rc={"text.usetex": True}):
        ax.set_xlabel("time step ($t$) [dimensionless]")
        ax.set_ylabel("abs. eigenvalue $\\vert \\lambda^{t} \\vert$")

    return ax


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

    eigenvectors = np.asarray(eigenvectors)

    fig = plt.figure(figsize=[6, 12])

    plot_eigenvectors = np.min([eigenvectors.shape[0], 10])
    is_even = np.mod(plot_eigenvectors, 2) == 0

    for i in range(2, plot_eigenvectors):
        # TODO: this plots also the trivial case i == n
        ax = fig.add_subplot(plot_eigenvectors // 2 - is_even, 2, i - 1)

        ax.scatter(
            eigenvectors[n, :],
            eigenvectors[i, :],
            0.1,
            c=colors,
            marker=".",
            cmap=plt.cm.Spectral,
        )
        ax.axis("off")
        ax.set_title(r"$\Psi_{{{}}}$ vs. $\Psi_{{{}}}$".format(n, i))


def plot_eigenfunction_along_axis(X_axis_values, evec_evals):
    idx_sort = np.argsort(X_axis_values)[::-1]

    plt.figure()
    plt.plot(X_axis_values[idx_sort], evec_evals[idx_sort], "-*")


@NotImplementedError
def plot_hist_distance_sample(X, subsample, metric="euclidean"):
    """The plot can be used to make decision on where to set a cut_off. """
    # TODO
    pass


if __name__ == "__main__":
    plot_eigenvalues_time(np.random.rand(10), 10)
    plt.show()
