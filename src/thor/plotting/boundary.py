import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def plot_segmented_polygons(polygons, fill_values, show_boundaries=False, **kwargs):
    """
    Plot segmented polygons colored by supplied fill values.

    Parameters
    ----------
    polygons: :py:class:`list` of :py:class:`list` of :py:class:`tuple`
        List of polygons, where each polygon is represented as a list of vertices (tuples).
    fill_values: :py:class:`list` of :py:class:`float`
        List of fill values for each polygon.
    show_boundaries: :py:class:`bool`, optional
        Whether to show cell boundaries. Defaults to :py:obj:`False`.
    **kwargs
        Additional keyword arguments for customizing colormap normalization (vmin, vmax).

    Returns:
        :py:obj:`None`
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colormap = plt.cm.get_cmap('viridis')
    vmin = kwargs.get('vmin', min(fill_values))
    vmax = kwargs.get('vmax', max(fill_values))
    norm = Normalize(vmin=vmin, vmax=vmax)

    for poly, size in zip(polygons, fill_values):
        color = colormap(norm(size))
        coords = np.array(poly)
        ax.fill(coords[:, 0], coords[:, 1], closed=True, facecolor=color)

    # Create a mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # This line is crucial to create a mappable

    # Create the colorbar using the mappable
    cbar = plt.colorbar(sm, ax=ax, format="%d")
    cbar.set_label("Cell Size")

    if show_boundaries:
        plot_cell_boundaries(polygons, ax=ax)


def plot_cell_boundaries(polygons, ax=None, show=False, **kwargs):
    """
    Plot cell boundaries for the given polygons.

    Parameters
    ----------
    polygons : :py:class:`list` of :py:class:`list` of :py:class:`tuple`
        List of polygons, where each polygon is represented as a list of vertices (tuples).
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes on which to plot the cell boundaries. If :py:obj:`None`, a new figure and axes will be created. Defaults to :py:obj:`None`.
    show : :py:class:`bool`, optional
        Whether to show the plot. Defaults to :py:obj:`False`. `ax` will be returned if `show` is :py:obj:`False`.
    **kwargs
        Additional keyword arguments for customizing the plot taken by `matplotlib.pyplot.plot`.
    
    Returns
    -------
    `ax` if show is :py:obj:`False`, else :py:obj:`None`
    """

    if 'linewidth' not in kwargs:
        kwargs.update({'linewidth': 0.5})
    if 'color' not in kwargs:
        kwargs.update({'color': 'black'})

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    for poly in polygons:
        coords = np.array(poly)
        ax.plot(coords[:, 0], coords[:, 1], **kwargs)

    if show:
        plt.show()
    else:
        return ax
