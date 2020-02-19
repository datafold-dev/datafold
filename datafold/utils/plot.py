#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def plot_eigenvalues(eigenvalues, plot_unit_circle=False, semilogy=False):

    plt.figure()

    if eigenvalues.dtype == np.complexfloating:
        plt.plot(np.real(eigenvalues), np.imag(eigenvalues), "+")

        if plot_unit_circle:
            circle_values = np.linspace(0, 2 * np.pi, 3000)
            plt.plot(np.cos(circle_values), np.sin(circle_values), "b-")
            plt.axis("equal")

    elif eigenvalues.dtype == np.floating:
        eigenvalues = np.sort(eigenvalues.copy())[::-1]

        if semilogy:
            plt.semilogy(np.arange(len(eigenvalues)), eigenvalues, "+")
        else:
            plt.plot(np.arange(len(eigenvalues)), eigenvalues, "+")


def plot_eigenvalues_time(eigenvalues, n_timesteps):
    vander_matrix = np.vander(np.abs(eigenvalues), n_timesteps, increasing=True)

    plt.figure()
    plt.plot(np.arange(n_timesteps), vander_matrix.T, "-")


def plot_eigenfunction_along_axis(X_axis_values, evec_evals):
    idx_sort = np.argsort(X_axis_values)[::-1]

    plt.figure()
    plt.plot(X_axis_values[idx_sort], evec_evals[idx_sort], "-*")


def plot_hist_distance_sample(X, subsample, metric="euclidean"):
    """The plot can be used to make decision on where to set a cut_off. """
    # TODO
    pass


if __name__ == "__main__":
    plot_eigenvalues_time(np.random.rand(10), 10)
    plt.show()
