import logging

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import zscore
from thor.utils import update_kwargs, update_kwargs_exclusive

from ._utils import get_cells_ROI, paint_regions, sample_n_paint_regions

logger = logging.getLogger(__name__)


def single(
    var_name,
    img_mask,
    ax=None,
    vor=None,
    full_res_im=None,
    ROI_tuple=None,
    img_alpha=1,
    figsize=(5, 5),
    dpi=500,
    show_cbar=True,
    cbar_loc=[1.04, 0.02, 0.05, 0.15],
    return_fig=False,
    show=False,
    **imshow_kwds
):
    """Color the cells or nuclei with one variable, gene name, or any other observable.

    Parameters
    ----------
    var_name : :py:class:`str`
        Name of the variable to color the cells or nuclei. For example, a gene name, cell type, etc.
    img_mask : :class:`numpy.ndarray`
        Image mask filled with the variable values. Should have the same size as `full_res_im`.
    ax : :class:`matplotlib.axes.Axes`, optional
        The Axes object to plot the gene mask. If :py:obj:`None`, a new figure and Axes will be created.
    vor : :class:`scipy.spatial.Voronoi`, optional
        The Voronoi diagram object.
    full_res_im : :class:`numpy.ndarray`, optional
        The full-size image where the gene expression mask should be plotted.
    ROI_tuple : :py:class:`tuple`, optional
        A tuple (left, bottom, width, height) representing the region of interest (ROI)
        where the gene expression mask should be displayed. If :py:obj:`None`, the entire `img_mask` will be plotted.
    img_alpha : :py:class:`float`, optional
        Alpha value (transparency) of the `full_res_im` if provided. Default is 1 (fully opaque).
    figsize : :py:class:`tuple`, optional
        Figure size in inches (width, height). Default is (5, 5).
    dpi : :py:class:`int`, optional
        Dots per inch for the figure. Default is 500.
    show_cbar : :py:class:`bool`, optional
        Whether to show the colorbar. Default is :py:obj:`True`.
    cbar_loc : :py:class:`list`, optional
        The location of the colorbar as [left, bottom, width, height]. Default is [1.04, 0.02, 0.05, 0.15].
    return_fig : :py:class:`bool`, optional
        If :py:obj:`True`, return the Figure object. Default is :py:obj:`False`.
    show : :py:class:`bool`, optional
        If :py:obj:`True`, display the plot. If :py:obj:`False`, return the Axes object without displaying the plot. Default is :py:obj:`False`.
    **imshow_kwds : :py:class:`dict`, optional
        Additional keyword arguments for customizing the imshow function, such as 'cmap', 'vmin', 'vmax', etc.

    Returns
    -------
    :class:`matplotlib.axes.Axes` or :class:`matplotlib.figure.Figure`, optional
        If `return_fig` is :py:obj:`False` and `show` is :py:obj:`False`, the function returns the Axes object.
        If `return_fig` is :py:obj:`True`, the function returns the Figure object.
        If `show` is :py:obj:`True`, the function displays the plot and returns :py:obj:`None`.

    Notes
    -----
    This function plots the gene expression mask within a specified region of interest (ROI) or the entire `img_mask`
    if `ROI_tuple` is :py:obj:`None`. The function can overlay the gene expression mask on a full-size image (`full_res_im`) and/or
    a Voronoi diagram (`vor`) if provided.

    The function allows customization of the colormap using the `use_global_vmax` and `use_global_vmin` parameters,
    which automatically set the maximum and minimum color values based on the global maximum and minimum values of the
    `img_mask`, respectively.

    Examples
    --------
    >>> import numpy as np

    >>> # Assuming `img_mask` is a numpy array representing the gene expression mask.
    >>> # Also, `vor` is a Voronoi diagram object, and `full_res_im` is a numpy array containing the full-size image.
    >>> pl.single("Gene_X", img_mask, vor=vor, full_res_im=full_res_im, ROI_tuple=(10, 20, 30, 40))

    >>> # Customizing the plot with optional arguments
    >>> pl.single("Gene_Y", img_mask, show_cbar=False, cmap='viridis', use_global_vmax=False, figsize=(8, 6))
    """

    if ROI_tuple is not None:
        assert len(
            ROI_tuple
        ) == 4, "ROI_tuple should be a tuple of (left, bottom, width, height)!"

    #assert use_global_vmax, "use_global_vmax=False is not supported yet!"
    #assert use_global_vmin, "use_global_vmin=False is not supported yet!"
    #val_max = np.max(var_values)
    #val_min = np.min(var_values)
    #vmin, vmax = imshow_kwds.get('vmin', val_min), imshow_kwds.get('vmax', val_max)
    #vmin, vmax = update_vmin_vmax((vmin, vmax), (val_min, val_max))
    #imshow_kwds.update({'vmin': vmin, 'vmax': vmax})

    imshow_kwds.update({'extent': None})
    # cmap = imshow_kwds.get('cmap', 'viridis')
    # cmap_new, sm = get_colormap(cmap, var_values, cell_labels, vmin, vmax)
    # imshow_kwds.update({'cmap': cmap_new})

    if ax is None:
        fig, ax = create_axes(fig_size=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    if ROI_tuple is not None:
        ax_img = plot_within_roi(
            ax, img_mask, full_res_im, vor, ROI_tuple, img_alpha, **imshow_kwds
        )
    else:
        raise NotImplementedError(
            "Visualizing the whole image is not supported! Use BioGIS instead."
        )
        #ax_img = plot_entire_mask(ax, img_mask, full_res_im, vor, img_alpha, **imshow_kwds)

    set_xlim_ylim(ax, img_mask, ROI_tuple)

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    if var_name is not None and var_name != '':
        ax.set_title(var_name)

    if show_cbar:
        add_colorbar(ax, ax_img, cbar_loc, var_name)

    plt.tight_layout()

    if return_fig:
        return fig

    if show:
        plt.show()
    else:
        return ax


def multiple(
    vars_list,
    img_masks_list,
    ax=None,
    vor=None,
    palette='tab20',
    full_res_im=None,
    ROI_tuple=None,
    img_alpha=1,
    figsize=(5, 5),
    dpi=150,
    show_legend=True,
    legend_loc='upper left',
    show_cbar=False,
    return_fig=False,
    show=False,
    **imshow_kwds
):
    """Color the cells or nuclei with multiple variables.
    
    Parameters
    ----------
    vars_list : :py:class:`list`
        List of variable names to color the cells or nuclei.
    img_masks_list : :py:class:`list` of :class:`numpy.ndarray`
        List of image masks filled with the variable values. Each mask should have the same size as `full_res_im`.
    ax : :class:`matplotlib.axes.Axes`, optional
        The Axes object to plot the gene masks. If :py:obj:`None`, a new figure and Axes will be created.
    vor : :class:`scipy.spatial.Voronoi`, optional
        The Voronoi diagram object.
    palette : :py:class:`str` or :py:class:`list` or :py:class:`tuple` or :class:`numpy.ndarray`, optional
        The color palette to use for coloring variables. Can be a valid matplotlib colormap name, or a list/tuple/array of colors.
    full_res_im : :class:`numpy.ndarray`, optional
        The full-size image where the gene expression masks should be plotted.
    ROI_tuple : :py:class:`tuple`, optional
        A tuple (left, bottom, width, height) representing the region of interest (ROI) where the gene expression masks should be displayed.
    img_alpha : :py:class:`float`, optional
        Alpha value (transparency) of the `full_res_im` if provided. Default is 1 (fully opaque).
    figsize : :py:class:`tuple`, optional
        Figure size in inches (width, height). Default is (5, 5).
    dpi : :py:class:`int`, optional
        Dots per inch for the figure. Default is 150.
    show_legend : :py:class:`bool`, optional
        Whether to show the legend indicating variables. Default is :py:obj:`True`.
    legend_loc : :py:class:`str`, optional
        Location for the legend. Default is 'upper left'.
    show_cbar : :py:class:`bool`, optional
        Whether to show the colorbar. Default is :py:obj:`False`.
    return_fig : :py:class:`bool`, optional
        If :py:obj:`True`, return the Figure object. Default is :py:obj:`False`.
    show : :py:class:`bool`, optional
        If :py:obj:`True`, display the plot. If :py:obj:`False`, return the Axes object without displaying the plot. Default is :py:obj:`False`.
    **imshow_kwds : :py:class:`dict`, optional
        Additional keyword arguments for customizing the imshow function, such as 'cmap', 'vmin', 'vmax', etc.

    Returns
    -------
    :class:`matplotlib.axes.Axes` or :class:`matplotlib.figure.Figure`, optional
        If `return_fig` is :py:obj:`False` and `show` is :py:obj:`False`, the function returns the Axes object.
        If `return_fig` is :py:obj:`True`, the function returns the Figure object.
        If `show` is :py:obj:`True`, the function displays the plot and returns :py:obj:`None`.

    Notes
    -----
    This function plots multiple gene expression masks within a specified region of interest (ROI) or the entire `img_masks_list`
    if `ROI_tuple` is :py:obj:`None`. Each variable's mask is color-coded using the provided palette.

    The function supports showing either a legend indicating the variables (`show_legend=True`) or a single colorbar
    (`show_cbar=True`) for the entire plot. Both options cannot be selected simultaneously.

    Examples
    --------
    >>> import numpy as np

    >>> # Assuming `img_masks_list` is a list of numpy arrays representing gene expression masks.
    >>> # Also, `vor` is a Voronoi diagram object, and `full_res_im` is a numpy array containing the full-size image.
    >>> pl.multiple(["Gene_X", "Gene_Y"], img_masks_list, vor=vor, full_res_im=full_res_im, ROI_tuple=(10, 20, 30, 40))

    >>> # Customizing the plot with optional arguments
    >>> pl.multiple(["Gene_A", "Gene_B"], img_masks_list, show_legend=False, palette=['red', 'blue'], use_global_vmax=False, figsize=(8, 6))
    """

    vars_list = list(vars_list)
    assert len(vars_list) == len(
        img_masks_list
    ), 'Inequal length of the vars_list and img_masks_list!'
    # `extent` would be confusing with `xlim`, `ylim`
    imshow_kwds.update({'extent': None})

    vPara_iter = dict()
    for vPara in ['vmax', 'vmin', 'cmap']:
        if vPara in imshow_kwds:
            if isinstance(imshow_kwds[vPara], (list, tuple, np.ndarray)):
                vPara_iter[vPara] = imshow_kwds[vPara]
            else:
                vPara_iter[vPara] = [imshow_kwds[vPara]] * len(vars_list)

    assert show_cbar != show_legend, "Can only show the colorbar or the legend, not both or neither!"

    palette = get_palette(palette, vars_list)

    plot_genes = []
    vars_list_use = []
    if ROI_tuple is not None:
        l, b, w, h = ROI_tuple
        for i, gene in enumerate(vars_list):
            gmask_plot = img_masks_list[i][b:b + h + 1, l:l + w + 1]

            # skip non-expressed genes in the region
            if gmask_plot.all() is np.ma.masked:
                continue
            plot_genes.append(gene)
            vars_list_use.append(vars_list[i])
    else:
        for i, gene in enumerate(vars_list):
            gmask_plot = img_masks_list[i]

            if gmask_plot.all() is np.ma.masked:
                continue
            plot_genes.append(gene)
            vars_list_use.append(vars_list[i])

    for i, gene in enumerate(vars_list_use):

        if show_cbar:
            cbar_loc = np.array([1.04, 0.02, 0.05, 0.15]
                               ) + np.array([0, 0.2, 0, 0]) * i
            if 'cmap' in vPara_iter:
                color_gene_cmap = vPara_iter['cmap'][i]
            else:
                color_gene_cmap = mcolors.LinearSegmentedColormap.from_list(
                    'custom_monotonic_cmap', ['white'] + [palette[gene]]
                )

        if show_legend:
            cbar_loc = None
            color_gene_cmap = mcolors.ListedColormap([palette[gene]])

        imshow_kwds.update({'cmap': color_gene_cmap})
        for vPara in ['vmax', 'vmin']:
            try:
                imshow_kwds.update({vPara: vPara_iter[vPara][i]})
            except:
                pass

        if i == vars_list_use.index(plot_genes[0]):
            # handle background image and the cell Voronoi outlines.
            fig = single(
                gene,
                img_masks_list[i],
                vor=vor,
                full_res_im=full_res_im,
                ROI_tuple=ROI_tuple,
                img_alpha=img_alpha,
                show_cbar=show_cbar,
                cbar_loc=cbar_loc,
                show=False,
                return_fig=True,
                ax=ax,
                figsize=figsize,
                dpi=dpi,
                **imshow_kwds
            )
        else:
            fig = single(
                gene,
                img_masks_list[i],
                vor=None,
                full_res_im=None,
                ROI_tuple=ROI_tuple,
                show_cbar=show_cbar,
                cbar_loc=cbar_loc,
                show=False,
                return_fig=True,
                ax=fig.axes[0],
                **imshow_kwds
            )

    ax = fig.axes[0]
    if show_legend:
        legend_elements = [
            Patch(facecolor=palette[gene], label=gene) for gene in plot_genes
        ]

        ax.legend(
            handles=legend_elements,
            loc=legend_loc,
            bbox_to_anchor=[1, 0.75],
            title='molecules',
            fancybox=True
        )

    ax.set_title('')

    if return_fig:
        return fig
    if show:
        plt.show()
    else:
        return ax


def clusters(
    cluster_labels_list, image_size=None, cells_pixels=None, **multiple_kwds
):
    """Color the cells or nuclei with cluster labels.

    Parameters
    ----------
    cluster_labels_list : list
        List of cluster labels for each cell or nucleus.
    image_size : tuple, optional
        Tuple (height, width) representing the size of the image.
    cells_pixels : list of numpy arrays, optional
        List of numpy arrays representing the pixels of each cell or nucleus.
    **multiple_kwds : dict, optional
        Additional keyword arguments for the `multiple` function.
    
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot or matplotlib.figure.Figure, optional
        If `return_fig` is False and `show` is False, the function returns the Axes object.
        If `return_fig` is True, the function returns the Figure object.
        If `show` is True, the function displays the plot and returns None.
    """

    ct_dummies = pd.get_dummies(cluster_labels_list)
    cmask_var_cluster = dict()
    for ct in ct_dummies:
        var_color = ct_dummies[ct].values
        cmask_var_cluster[ct] = paint_regions(
            image_size, cells_pixels, var_color
        )
        cmask_var_cluster[ct] = np.ma.masked_equal(cmask_var_cluster[ct], 0)
    return multiple(
        ct_dummies.columns, [cmask_var_cluster[ct] for ct in ct_dummies],
        **multiple_kwds
    )


def single_molecule(
    var,
    var_expression,
    cells_pixels,
    image_size=None,
    global_norm=True,
    **single_kwds
):
    """Color the cells or nuclei with a variable, gene or any observable (gene expression vector as input).

    Parameters
    ----------
    var : str
        Variable name to color the cells or nuclei.
    var_expression : numpy 1d array
        Array of shape (n_cells, ) containing the expression values of the variables.
    cells_pixels : list of numpy arrays
        List of numpy arrays representing the pixels of each cell or nucleus.
    image_size : tuple, optional
        Tuple (height, width) representing the size of the image.
    global_norm : bool, optional
        If True, normalize the expression values of each variable across all cells. Default is True.
    **single_kwds : dict, optional
        Additional keyword arguments for the `single` function.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot or matplotlib.figure.Figure, optional
        If `return_fig` is False and `show` is False, the function returns the Axes object.
        If `return_fig` is True, the function returns the Figure object.
        If `show` is True, the function displays the plot and returns None.

    """
    im = single_kwds.get('full_res_im', None)
    assert (image_size
            is not None) or (im is not None), "image size is not provided!"
    if (image_size is not None) and (im is not None):
        assert image_size == im.shape[:2], "image size is not consistent!"
    if image_size is None:
        image_size = im.shape[:2]

    ROI_tuple = single_kwds['ROI_tuple']
    assert ROI_tuple is not None, "ROI_tuple is not provided!"

    nmask = paint_regions(
        image_size, cells_pixels, cell_colors_list=var_expression
    )

    vmin = single_kwds.get('vmin', None)
    vmax = single_kwds.get('vmax', None)

    if global_norm:
        var_color = var_expression
    else:
        center_pos = np.array(
            [
                np.stack(cells_pixels[i]).mean(axis=1)
                for i in range(len(cells_pixels))
            ]
        )
        cells_on_patch = get_cells_ROI(center_pos, ROI_tuple)
        var_color = var_expression[cells_on_patch]

    vmin, vmax = update_vmin_vmax((vmin, vmax), var_color)
    single_kwds.update({'vmin': vmin, 'vmax': vmax})

    return single(var, nmask, **single_kwds)


def multi_molecules(
    vars_list,
    vars_expression_array,
    cells_pixels,
    image_size=None,
    global_norm=True,
    **multiple_kwds
):
    """Color the cells or nuclei with multiple variables, gene or any observable (gene expression array as input).

    Parameters
    ----------
    vars_list : list
        List of variable names to color the cells or nuclei.
    vars_expression_array : numpy array
        Array of shape (n_cells, n_variables) containing the expression values of the variables.
    image_size : tuple
        Tuple (height, width) representing the size of the image.
    cells_pixels : list of numpy arrays
        List of numpy arrays representing the pixels of each cell or nucleus.
    global_norm : bool, optional
        If True, normalize the expression values of each variable across all cells. Default is True.
    **multiple_kwds : dict, optional
        Additional keyword arguments for the `multiple` function.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot or matplotlib.figure.Figure, optional
        If `return_fig` is False and `show` is False, the function returns the Axes object.
        If `return_fig` is True, the function returns the Figure object.
        If `show` is True, the function displays the plot and returns None.

    """
    im = multiple_kwds.get('full_res_im', None)
    assert (image_size
            is not None) or (im is not None), "image size is not provided!"
    if (image_size is not None) and (im is not None):
        assert image_size == im.shape[:2], "image size is not consistent!"

    ROI_tuple = multiple_kwds['ROI_tuple']
    assert ROI_tuple is not None, "ROI_tuple is not provided!"
    #center_pos = np.array([(np.mean(cells_pixels[i]), np.mean(cells_pixels[i])) for i in range(len(cells_pixels))])
    if not global_norm:
        center_pos = np.array(
            [
                np.array(cells_pixels[i]).mean(axis=1)
                for i in range(len(cells_pixels))
            ]
        )
        cells_on_patch = get_cells_ROI(center_pos, ROI_tuple)

    mask_var_list = {}
    for i, var_name in enumerate(vars_list):
        var_color = vars_expression_array[:, i]
        nmask = paint_regions(
            image_size, cells_pixels, cell_colors_list=var_color
        )
        mask_var_list[var_name] = nmask

    vmin_list = multiple_kwds.get('vmin', [None] * len(vars_list))
    vmax_list = multiple_kwds.get('vmax', [None] * len(vars_list))

    for i in range(len(vars_list)):
        if global_norm:
            var_color = vars_expression_array[:, i]
        else:
            var_color = vars_expression_array[cells_on_patch, i]
        vmin_list[i], vmax_list[i] = update_vmin_vmax(
            (vmin_list[i], vmax_list[i]), var_color
        )
    multiple_kwds.update({'vmin': vmin_list, 'vmax': vmax_list})

    return multiple(
        vars_list, [mask_var_list[gene] for gene in vars_list], **multiple_kwds
    )


def multi_molecules_sample(
    vars_list,
    vars_expression_array,
    image_size,
    cells_pixels,
    global_norm=False,
    sample_more=1,
    random_seed=None,
    **multiple_kwds
):
    """Color the cells or nuclei with multiple variables, gene or any observable.
    Here the expression values are represented by sampled pixels drawn from the cells or nuclei.

    Parameters
    ----------
    vars_list : list
        List of variable names to color the cells or nuclei.
    vars_expression_array : numpy array
        Array of shape (n_cells, n_variables) containing the expression values of the variables.
    image_size : tuple
        Tuple (height, width) representing the size of the image.
    cells_pixels : list of numpy arrays
        List of numpy arrays representing the pixels of each cell or nucleus.
    global_norm : bool, optional
        If True, normalize the expression values of each variable across all cells. Default is False.
    sample_more : int, optional
        Number of additional samples to draw from each cell or nucleus. Default is 1.
    random_seed : int, optional
        Random seed for reproducibility. Default is None.
    **multiple_kwds : dict, optional
        Additional keyword arguments for the `multiple` function.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot or matplotlib.figure.Figure, optional
        If `return_fig` is False and `show` is False, the function returns the Axes object.
        If `return_fig` is True, the function returns the Figure object.
        If `show` is True, the function displays the plot and returns None.

    """

    mask_var_sample = {}

    ROI_tuple = multiple_kwds['ROI_tuple']
    assert ROI_tuple is not None
    center_pos = np.array(
        [
            (np.mean(cells_pixels[i]), np.mean(cells_pixels[i]))
            for i in range(len(cells_pixels))
        ]
    )
    cells_on_patch = get_cells_ROI(center_pos, ROI_tuple)

    for i, var_name in enumerate(vars_list):
        var_color = vars_expression_array[:, i]
        var_color = zscore(np.array(var_color))

        mask = sample_n_paint_regions(
            image_size,
            cells_pixels,
            cells_on_patch,
            var_color,
            global_norm=global_norm,
            sample_more=sample_more,
            random_seed=random_seed
        )
        mask_var_sample[var_name] = mask

    return multiple(
        vars_list, [mask_var_sample[gene] for gene in vars_list],
        **multiple_kwds
    )


# Helper functions
def create_axes(fig_size, dpi):
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    fig_size = fig.get_size_inches()
    fig_size = fig_size / fig_size.max()
    width, height = fig_size * 0.9
    ax = plt.axes([0, 0.05, width, height])
    return fig, ax


def plot_within_roi(
    ax, img_mask, full_res_im, vor, ROI_tuple, img_alpha, **imshow_kwds
):
    l, b, w, h = ROI_tuple

    if full_res_im is not None:
        if len(full_res_im.shape) == 2:
            #im_type is 'gray'
            ax.imshow(
                full_res_im[b:b + h + 1, l:l + w + 1],
                alpha=img_alpha,
                zorder=0,
                cmap='gray'
            )
        if len(full_res_im.shape) == 3:
            #im_type is 'color':
            ax.imshow(
                full_res_im[b:b + h + 1, l:l + w + 1],
                alpha=img_alpha,
                zorder=0
            )

    if isinstance(vor, Voronoi):
        vor.vertices = vor.vertices - np.array((l, b))
        voronoi_plot_2d(
            vor, show_points=False, show_vertices=False, ax=ax, zorder=999
        )
        vor.vertices = vor.vertices + np.array((l, b))

    ax_img = ax.imshow(img_mask[b:b + h + 1, l:l + w + 1], **imshow_kwds)
    return ax_img


def plot_entire_mask(
    ax, img_mask, full_res_im, vor, vmax, img_alpha, **imshow_kwds
):
    """ Note: It is not efficient to plot the whole image. Please use Mjolnir instead
    """

    if full_res_im is not None:
        if len(full_res_im.shape) == 2:
            ax.imshow(full_res_im, alpha=img_alpha, zorder=0, cmap='gray')
        if len(full_res_im.shape) == 3:
            ax.imshow(full_res_im, alpha=img_alpha, zorder=0)

    if isinstance(vor, Voronoi):
        voronoi_plot_2d(
            vor, show_points=False, show_vertices=False, ax=ax, zorder=999
        )

    ax_img = ax.imshow(img_mask, **imshow_kwds)
    return ax_img


def set_xlim_ylim(ax, img_mask, ROI_tuple):
    if ROI_tuple is not None:
        l, b, w, h = ROI_tuple
        ax.set_xlim(0, w + 1)
        ax.set_ylim(0, h + 1)
    else:
        h, w = img_mask.shape
        ax.set_xlim(0, w + 1)
        ax.set_ylim(0, h + 1)


def add_scalebar(ax, units_per_pixel, unit, location, **kwargs):
    """
    Example
    -------
    microPerPixel = thor.utils.convert_pixel_to_micron_visium(ad, res='fullres', spotDiameterinMicron=65)
    fig, ax = plt.subplots()
    ax.imshow(rim)
    add_scalebar(ax, microPerPixel, 'um', 'lower left')

    """
    scalebar_kwargs = {
            'location': location,
            'units': unit,
            'fixed_value': 50,
            'fixed_units': unit,
            'box_alpha': 0.6,
            'color': 'black',
            'rotation': 'horizontal'
            }

    scalebar_kwargs = update_kwargs_exclusive(ScaleBar, scalebar_kwargs)
    scalebar_kwargs.update(kwargs)

    scalebar = ScaleBar(units_per_pixel, **scalebar_kwargs)
    ax.add_artist(scalebar)


def get_palette(palette, vars_list):
    if isinstance(palette, str):
        try:
            palette = mpl.colormaps[palette].colors
        except IndexError:
            logger.error(
                "Please specify a valid color palette name from matplotlib.colormaps"
            )

    if isinstance(palette, (list, tuple, np.ndarray)):
        palette = dict(zip(vars_list, palette[:len(vars_list)]))

    assert isinstance(palette, dict)
    return palette


def add_colorbar(ax, ax_img, cbar_loc, label):
    axins = ax.inset_axes(cbar_loc)
    cbar = plt.colorbar(ax_img, cax=axins, orientation="vertical")
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(label, rotation=90)


def update_vmin_vmax(v_set, v_data):
    """Parse vmin, vmax.
    v_set : tuple
        (vmin, vmax) set by users, can be None, int, float, or string. If string, should be like 'p1', representing 1 percentile.
    v_data : vector
        the color data to be plotted
    
    Returns
    -------
    vmin, vmax : float
        the parsed vmin, vmax  
    """
    vmin_set, vmax_set = v_set
    vmin_data, vmax_data = v_data.min(), v_data.max()

    if vmax_set is None:
        vmax = vmax_data
    elif isinstance(vmax_set, (int, float)):
        vmax = min(vmax_set, vmax_data)
    elif isinstance(vmax_set, str):
        vmax = np.quantile(v_data, float(vmax_set[1:]) * 0.01)
    else:
        raise ValueError("vmax should be None, int, float, or string!")

    if vmin_set is None:
        vmin = vmin_data
    elif isinstance(vmin_set, (int, float)):
        vmin = max(vmin_set, vmin_data)
    elif isinstance(vmin_set, str):
        vmin = np.quantile(v_data, float(vmin_set[1:]) * 0.01)
    else:
        raise ValueError("vmin should be None, int, float, or string!")

    assert vmin < vmax, "vmin should be smaller than vmax!"
    return vmin, vmax
