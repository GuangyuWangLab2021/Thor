import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def get_spot_deconvolution(ad, cell_group_key='clusters'):
    hetero_spot = {}

    for spot, cells in ad.uns['all_cells_in_spot'].items():
        cells = [c for c in cells if c in ad.obs_names]
        cell_types_in_spot = ad.obs[cell_group_key][cells]

        if len(cell_types_in_spot) > 0:
            hetero_spot[spot] = cell_types_in_spot.value_counts(normalize=True)

    ad.obsm['deconvolution'] = pd.DataFrame(hetero_spot).T



# Adapted from stlearn
def deconvolution_plot(
    adata,
    library_id: str = None,
    use_label: str = "louvain",
    cluster: [int, str] = None,
    celltype: str = None,
    celltype_threshold: float = 0,
    data_alpha: float = 1.0,
    threshold: float = 0.0,
    palette: dict = None,  # The colors to use for each label...
    tissue_alpha: float = 1.0,
    title: str = None,
    spot_size=10,
    show_axis: bool = False,
    show_legend: bool = True,
    show_donut: bool = True,
    cropped: bool = True,
    margin: int = 100,
    name: str = None,
    dpi: int = 150,
    output: str = 'pie_plot.pdf',
    copy: bool = False,
    figsize: tuple = (6.4, 4.8),
    show=True,
):
    """\
    Clustering plot for sptial transcriptomics data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_id
        Library id stored in AnnData.
    use_label
        Use label result of cluster method.
    list_cluster
        Choose set of clusters that will display in the plot.
    data_alpha
        Opacity of the spot.
    tissue_alpha
        Opacity of the tissue.
    spot_size
        Size of the spot.
    show_axis
        Show axis or not.
    show_legend
        Show legend or not.
    show_donut
        Whether to show the donut plot or not.
    show_trajectory
        Show the spatial trajectory or not. It requires stlearn.spatial.trajectory.pseudotimespace.
    show_subcluster
        Show subcluster or not. It requires stlearn.spatial.trajectory.global_level.
    name
        Name of the output figure file.
    dpi
        DPI of the output figure.
    output
        Save the figure as file or not.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Nothing
    """

    # plt.rcParams['figure.dpi'] = dpi

    imagecol = adata.obsm['spatial'][:, 0]
    imagerow = adata.obsm['spatial'][:, 1]

    fig, ax = plt.subplots(figsize=figsize)

    label = adata.obsm["deconvolution"].T

    tmp = label.sum(axis=1)

    label_filter = label.loc[tmp[tmp >= np.quantile(tmp, threshold)].index]

    base = adata.obsm['spatial']

    if celltype is not None:
        base = base.loc[adata.obs_names[adata.obsm["deconvolution"][celltype] >
                                        celltype_threshold]]

    label_filter_ = label_filter.copy()
    colors = label_filter.index.map(palette)

    for i, xy in enumerate(base):
        _ = ax.pie(
            label_filter_.T.iloc[i].values,
            colors=colors,
            center=(xy[0], xy[1]),
            radius=spot_size,
            frame=True,
        )
    ax.autoscale()

    if show_donut:
        ax_pie = fig.add_axes([0.5, -0.4, 0.03, 0.5])

        def my_autopct(pct):
            return ("%1.0f%%" % pct) if pct >= 4 else ""

        ax_pie.pie(
            label_filter_.sum(axis=1),
            colors=colors,
            radius=10,
            # frame=True,
            autopct=my_autopct,
            pctdistance=1.1,
            startangle=90,
            wedgeprops=dict(width=(3), edgecolor="w", antialiased=True),
            textprops={"fontsize": 5},
        )

    ax.axis("off")

    if cropped:
        ax.set_xlim(imagecol.min() - margin, imagecol.max() + margin)

        ax.set_ylim(imagerow.min() - margin, imagerow.max() + margin)

        ax.set_ylim(ax.get_ylim()[::-1])

        # plt.gca().invert_yaxis()

    if name is None:
        name = use_label

    if output is not None:
        fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()


def plot_colorbar(
    palette_dict,
    dpi=300,
    output='colorbar.pdf'
):


    # Continuous legend using colorbar
    my_order = [
        'HPC', 'HPC_CA1', 'HPC_CA2/3', 'HPC_DG', 'THAL_venmedial/lateral',
        'THAL_medHabenula', 'THAL_latHabenula', 'WM_ventral', 'WM_dorsal',
        'PIA_dorsal', 'chpl', 'Others'
    ][::-1]

    color_vals = list(range(0, 12, 1))

    my_colors = [palette_dict[c] for c in my_order]
    my_cmap = ListedColormap(my_colors)
    my_norm = mpl.colors.Normalize(0, 12)

    fig = plt.figure()
    ax_cb = fig.add_axes([0.9, 0.25, 0.03, 0.5], axisbelow=False)
    cb = mpl.colorbar.ColorbarBase(
        ax_cb, cmap=my_cmap, norm=my_norm, ticks=color_vals
    )

    cb.ax.tick_params(size=0)
    loc = np.array(color_vals) + 0.5
    cb.set_ticks(loc)
    cb.set_ticklabels(my_order)
    cb.outline.set_visible(False)

    plt.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0)
