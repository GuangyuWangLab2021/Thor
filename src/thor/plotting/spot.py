import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def parse_vmin_vmax(vmin, vmax, arr):
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()
    if isinstance(vmin, str):
        vmin = quantile_to_number(vmin, arr)
    if isinstance(vmax, str):
        vmax = quantile_to_number(vmax, arr)
    return vmin, vmax

def set_xlim_ylim(ax, ROI_tuple):
        l, b, w, h = ROI_tuple
        ax.set_xlim(0, w + 1)
        ax.set_ylim(0, h + 1)

def on_patch_rect(xy, ROI_tuple):
    l, b, w, h = ROI_tuple
    x_min, x_max, y_min, y_max = l, l+w, b, b+h
    x = xy[:, 0]
    y = xy[:, 1]

    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

def plot_within_roi(ax, full_res_im, ROI_tuple, img_alpha):
    l, b, w, h = ROI_tuple

    if len(full_res_im.shape) == 2:
        ax.imshow(full_res_im[b:b + h + 1, l:l + w + 1], alpha=img_alpha, zorder=0, cmap='gray')
    if len(full_res_im.shape) == 3:
        ax.imshow(full_res_im[b:b + h + 1, l:l + w + 1], alpha=img_alpha, zorder=0)
    return ax

def create_colorbar(ax, colormap, norm):
    # Create a mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # This line is crucial to create a mappable

    # Create the colorbar using the mappable
    cbar = plt.colorbar(sm, ax=ax)
#    cbar.set_label("Expression")    

def quantile_to_number(input_str, arr):
    """ Convert quantile to number
    Args:
        input_str (str): input string, which has to be 'p' followed by a number
        arr (np.ndarray): 1D expression array of a gene
    Returns:
        float: quantile value

    By Pengzhi Zhang
    """

    assert isinstance(input_str, str), "input_str has to be 'p' followed by a number between 0 and 100"

    q = float(input_str[1:])
    q = 0.01 * q
    number = np.quantile(arr, q)
    return number

def plot_spot(im, centers, diameter, colors, ROI_tuple=None, cmap='viridis', alpha=1, local=False, vmin=None, vmax=None, img_alpha=1,
              show_colorbar=True, save_path=None):
    """
    Plot spot expression on the full resolution image (in a region).

    Parameters
    ----------
    im : np.ndarray
        Full resolution image.
    centers : np.ndarray
        Spot coordinates in pixels in the original image.
    diameter : float
        Diameter of the spots in pixels in the original image.
    colors : np.ndarray
        Values used to color the spots. It is set to be the gene expression values.
    ROI_tuple : tuple, optional, default: None
        A tuple of (left, bottom, width, height) of the region of interest (ROI) to plot. If None, the whole image will be plotted.
    cmap : str, optional, default: 'viridis'
        Colormap to use for the spots.
    alpha : float, optional, default: 1
        Transparency of the spots.
    local : bool, optional, default: False
        Whether to use the local color scale for the spots. If True, the color scale will be based on the spots within the ROI.
    vmin : float or str, optional, default: None
        Minimum value of the color scale. If None, the minimum value of the colors will be used. If it is a string, it has to be 'p' followed by a number between 0 and 100, which will be converted to the quantile of the colors.
    vmax : float or str, optional, default: None
        Maximum value of the color scale. If None, the maximum value of the colors will be used. If it is a string, it has to be 'p' followed by a number between 0 and 100, which will be converted to the quantile of the colors.
    img_alpha : float, optional, default: 1
        Transparency of the full resolution image.
    show_colorbar : bool, optional, default: True
        Whether to show the colorbar.
    save_path : str, optional, default: None
        If not None, the plot will be saved to the path. Otherwise return the matplotlib figure object.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> from thor.utils import get_spot_diameter_in_pixels
    >>> gene_vec = ad[:, 'FOS'].X.toarray()[:, 0]
    >>> coors = ad.obsm['spatial']
    >>> diameter = get_spot_diameter_in_pixels(ad)
    >>> roi_tuple = (0, 0, 1000, 1000)
    >>> fig = plot_spot(fullres_im_arr, coors, diameter, gene_vec, ROI_tuple=roi_tuple, img_alpha=0.5, local=False, show_colorbar=False)
    """
    width, height = im.shape[0], im.shape[1]
    fig, ax = plt.subplots(figsize=(8, 8))

    n_cells = len(centers)
    radius = diameter * 0.5

    colormap = plt.cm.get_cmap(cmap)
    if ROI_tuple is None:
        ROI_tuple = (0, 0, width, height) 
    l, b, w, h = ROI_tuple
    origin_ROI = np.array([l, b])
    on_patch = on_patch_rect(centers, ROI_tuple)

    if local:
        colors_range = colors[on_patch]
    else:
        colors_range = colors

    vmin, vmax = parse_vmin_vmax(vmin, vmax, colors_range)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        
    for i in range(n_cells):
        if not on_patch[i]:
            continue
        center = centers[i] - origin_ROI
        color = colormap(norm(colors[i]))
        ax.add_patch(plt.Circle(center, radius=radius, facecolor=color, lw=3, fill=True, alpha=alpha))

    set_xlim_ylim(ax, ROI_tuple)

    plot_within_roi(ax, im, ROI_tuple, img_alpha)
    ax.invert_yaxis()

    ax.set_xticks([])
    ax.set_yticks([])

    if show_colorbar:
        create_colorbar(ax, colormap, norm)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.)
        plt.close(fig)
    else:
        return fig
