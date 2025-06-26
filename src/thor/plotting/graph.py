import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_cell_graph(
    adata,
    conn_key='snn',
    figsize=(8, 8),
    dpi=150,
    xlim=None,
    ylim=None,
    s=1,
    lw=1,
    offset_x=0,
    offset_y=0,
    return_fig=False,
    **kwds
):
    """
    Plot a cell graph using NetworkX and spatial coordinates.

    Parameters
    ----------
    adata : AnnData object
        An AnnData object containing the cell graph and spatial coordinates information.
    conn_key : str, optional
        The key for accessing the cell graph connectivity data in `adata.obsp`. Default is 'snn'.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (8, 8).
    dpi : int, optional
        Dots per inch for the figure. Default is 150.
    xlim : tuple, optional
        Limits for the x-axis of the plot. Default is None, which uses the default axis limits.
    ylim : tuple, optional
        Limits for the y-axis of the plot. Default is None, which uses the default axis limits.
    s : float, optional
        Scaling factor for the node size in the plot. Default is 1.
    lw : float, optional
        Scaling factor for the edge width in the plot. Default is 1.
    return_fig : bool, optional
        Whether to return the figure instead of displaying it. Default is False.
    **kwds : dict, optional
        Additional keyword arguments for customizing the network plot. Some useful options include:
        - 'node_color': List of hex color codes or matplotlib colors for node coloring.

    Returns
    -------
    None or matplotlib.figure.Figure
        If `return_fig` is False (default), the function displays the plot and returns None.
        If `return_fig` is True, the function returns the matplotlib Figure object without displaying the plot.

    Notes
    -----
    This function plots a cell graph using NetworkX based on the provided connectivity data and spatial coordinates
    stored in the `adata` object. The graph nodes are positioned in the plot according to the spatial coordinates,
    and the node sizes and edge widths are scaled based on the node degrees and edge weights, respectively.

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> # Assuming `adata` is an AnnData object containing the necessary connectivity and spatial information.
    >>> plot_cell_graph(adata)

    >>> # Customizing the plot with optional arguments
    >>> plot_cell_graph(adata, conn_key='snn_connectivities', figsize=(10, 10), s=2, lw=2, node_color='#FF0000')

    """


    conn_key = conn_key if conn_key in adata.obsp else f'{conn_key}_connectivities'
    try:
        # NetworkX 2
        G = nx.from_numpy_matrix(adata.obsp[conn_key])
    except AttributeError:
        G = nx.Graph(adata.obsp[conn_key].toarray())
    coor = adata.obsm['spatial']  # np.array([1, -1])

    # apply offset
    coor = coor - np.array([offset_x, offset_y])

    if xlim is not None:
        xlim = (xlim[0]-offset_x, xlim[1]-offset_x)
    if ylim is not None:
        ylim = (ylim[0]-offset_y, ylim[1]-offset_y)

    node_degree = np.array(list(dict(G.degree).values()))
    edge_weight = np.array([G.edges[e]['weight'] for e in G.edges])
    kwds.update(
        {
            'node_size': node_degree * 0.01 * s,
            'width': edge_weight * 0.1 * lw
        }
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    nx.draw(G, coor, ax=ax, **kwds)
    ax.invert_yaxis()

    if 'cmap' in kwds:
        cmap = kwds['cmap']
        assert 'node_color' in kwds
        color_vals = kwds['node_color']
        vmin = kwds.get('vmin', min(color_vals))
        vmax = kwds.get('vmax', max(color_vals))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
        sm._A = []
        cbar_loc = [1.01, 0.1, 0.05, 0.15]
        axins = ax.inset_axes(cbar_loc)
        plt.colorbar(sm, cax=axins, orientation="vertical")
        

    plt.xlim(xlim)
    plt.ylim(ylim)
    if return_fig:
        return fig
    plt.show()
