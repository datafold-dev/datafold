"""Module for interactive plotting diffusion maps and associated diagnostics.

"""

from typing import Callable, List

import numpy as np

from datafold.dynfold.diffusion_maps import DiffusionMaps
from datafold.dynfold.plot import (
    LookupDmapsEpsilon,
    LookupKernelMatsEpsilon,
    plot_diffusion_maps,
    plot_l_vs_epsilon,
    plot_results,
)

import ipywidgets as widgets
from IPython.display import display


def _checkboxes(
    descriptions: List[str], defaults: List[bool]
) -> List[widgets.Checkbox]:
    """Generates checkboxes.

    Generates interactive checkboxes.

    Parameters
    ----------
    descriptions: List[str]
        names of the checkboxes
    defaults: List[bool]=[False,False,False]
        Default interactive options

    """

    checkbox_list = []
    for description, val in zip(descriptions, defaults):
        checkbox_list.append(
            widgets.Checkbox(value=val, description=description, disabled=False)
        )
    return checkbox_list


def _plot_diffusion_maps(
    data: np.array,
    epsilon_magnitude: int,
    lut_dmaps: Callable[[float], DiffusionMaps] = None,
    colors: np.array = None,
) -> None:
    """Plot diffusion maps.

    Private convenience function for (data,epsilon_magnitude) signature instead of
    (data,dmap) signature.


    Parameters
    ----------
    data : np.array
        Original (or downsampled) data set.
    epsilon_magnitude: float
        Order of magnitude of epsilon parameter of diffusion maps
    lut_dmaps: Callable[[float],DiffusionMaps]=None
        Lookup Table with arg epsilon for diffusion map
    colors: np.array=None
        Optional argument to allow the user to provide coloring.

    """

    epsilon = round(10.0 ** epsilon_magnitude, -epsilon_magnitude)

    if lut_dmaps:
        dmaps = lut_dmaps(epsilon)
    else:
        dmaps = DiffusionMaps.select_new_kernel(data, epsilon).fit(num_eigenpairs=11)
    plot_diffusion_maps(data, dmaps, colors)


def plot_diffusion_maps_interactive_epsilon(
    data: np.array,
    epsilon_min_magnitude: int,
    epsilon_max_magnitude: int,
    colors: np.array = None,
) -> None:
    """Plot diffusion maps with variable epsilon.

    Provides user with a slider for epsilon, when working in jupyter notebooks. In the
    beginning, the diffusion maps for all provided epsilons is computed.


    Parameters
    ----------
    data : np.array
        Original (or downsampled) data set.
    epsilon_min_magnitude: int
        Least magnitude of epsilon to analyze.
    epsilon_max_magnitude: int
        Highest magnitude of epsilon to analyze.
    colors: np.array=None
        Optional argument to allow the user to provide coloring.

    """

    epsilons = [
        round(10.0 ** p, -p)
        for p in range(epsilon_min_magnitude, epsilon_max_magnitude + 1, 1)
    ]
    lut_dmaps = LookupDmapsEpsilon(data, epsilons)
    epsilon_magnitude = widgets.IntSlider(
        min=epsilon_min_magnitude,
        max=epsilon_max_magnitude,
        step=1,
        continuous_update=False,
    )

    out = widgets.interactive_output(
        _plot_diffusion_maps,
        {
            "data": widgets.fixed(data),
            "epsilon_magnitude": epsilon_magnitude,
            "colors": widgets.fixed(colors),
            "lut_dmaps": widgets.fixed(lut_dmaps),
        },
    )
    display(epsilon_magnitude, out)


def plot_diffusion_maps_interactively(
    data: np.array,
    dmaps: DiffusionMaps,
    colors: np.array = None,
    defaults: List[bool] = None,
) -> None:
    """Plot diffusion maps interactively.

    Provides user with interactive plot options for jupyter notebooks.


    Parameters
    ----------
    data : np.array
        Original (or downsampled) data set.
    dmaps: DiffusionMaps
        DiffusionMaps of the data.
    colors: np.array=None
        Optional argument to allow the user to provide coloring.
    defaults: List[bool]=[False,False,False]
        Default interactive options

    """

    if defaults is None:
        defaults = [False, False, False]

    spectrum, colored_data, diffusion_maps_2d = _checkboxes(
        ["spectrum", "data - eigenvector", "diffusion_maps_2D"], defaults
    )

    ui = widgets.HBox([spectrum, colored_data, diffusion_maps_2d])
    out = widgets.interactive_output(
        plot_results,
        {
            "data": widgets.fixed(data),
            "eigenvalues": widgets.fixed(dmaps.eigenvalues),
            "eigenvectors": widgets.fixed(dmaps.eigenvectors),
            "colors": widgets.fixed(colors),
            "spectrum": spectrum,
            "colored_data": colored_data,
            "diffusion_maps_2d": diffusion_maps_2d,
        },
    )
    display(ui, out)


def plot_l_vs_epsilon_interactively(
    data: np.array, epsilon_min_magnitude: int, epsilon_max_magnitude: int
) -> None:
    """Plot L(epsilon) with interactive functionality.

    Plot L(epsilon) (Master Thesis of Bah (2008)) for different epsilons.
    Extended by the insertion of a secant line in the neighborhood of an
    epsilon, which can be varied by an interactive tool for jupyter notebooks.

    Parameters
    ----------
    data : np.array
        Original (or downsampled) data set.
    epsilon_min_magnitude: int
        Least magnitude of epsilon to analyze.
    epsilon_max_magnitude: int
        Highest magnitude of epsilon to analyze.

    """

    if epsilon_max_magnitude <= epsilon_min_magnitude:
        raise ValueError(
            "epsilon_max_magnitude must be larger than epsilon_min_magnitude"
        )

    # DMaps need to be computed for more epsilon values than provided by the user to
    # ensure that the secant line can be computed.
    epsilons_coarse = [
        round(10.0 ** p, -p)
        for p in range(epsilon_min_magnitude - 1, epsilon_max_magnitude + 2, 1)
    ]
    epsilons_fine = (
        10 ** np.arange(epsilon_min_magnitude - 1, epsilon_max_magnitude + 1.25, 0.25)
    ).tolist()

    lut_km_coarse = LookupKernelMatsEpsilon(data, epsilons_coarse)
    lut_km_fine = LookupKernelMatsEpsilon(data, epsilons_fine)

    epsilon_magnitude = widgets.IntSlider(
        min=epsilon_min_magnitude,
        max=epsilon_max_magnitude,
        step=1,
        continuous_update=False,
    )

    out = widgets.interactive_output(
        plot_l_vs_epsilon,
        {
            "data": widgets.fixed(data),
            "epsilons_coarse": widgets.fixed(epsilons_coarse),
            "epsilons_fine": widgets.fixed(epsilons_fine),
            "epsilon_magnitude": epsilon_magnitude,
            "lut_km_coarse": widgets.fixed(lut_km_coarse),
            "lut_km_fine": widgets.fixed(lut_km_fine),
            "epsilon": widgets.fixed(None),
        },
    )
    display(epsilon_magnitude, out)
