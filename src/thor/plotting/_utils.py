# utility functions used for plotting
import logging
import os

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.sparse as sp

from scipy.spatial import Voronoi
from scipy.sparse import load_npz
from skimage import draw
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors

from thor.utils import on_patch_rect
from thor.utils import get_adata_layer_array

logger = logging.getLogger(__name__)


def map_nuclei_pixels(cm, ad_cell_pos):
    """
    Identify nuclei regions and match them with cell positions.

    Parameters
    ----------
    cm : numpy array, shape (n_pixels_row, n_pixels_col)
        The numpy array representing identified "nuclei" from an H&E image.
    ad_cell_pos : numpy array, shape (n_cells, 2)
        The pixel locations of all cells.

    Returns
    -------
    nuclei_region_pixels : list
        A list of tuples, each containing the pixel coordinates of a "nuclei" region
        that matches with the given cell positions in `ad_cell_pos`.
        The format of the tuples is (row_coordinates, column_coordinates).

    Notes
    -----
    This function takes a numpy array `cm`, representing identified "nuclei" from an H&E image,
    and a numpy array `ad_cell_pos`, containing the pixel locations of all cells.

    The function computes the centroid coordinates of the "nuclei" regions using `regionprops` function,
    then uses the `NearestNeighbors` algorithm to find the nearest nuclei region for each cell position
    in `ad_cell_pos` based on Euclidean distance. If the distance between a cell and a nuclei region is within
    a tolerance (`tol`), the nuclei region is considered to match with the cell, and its pixel coordinates
    are stored in the `nuclei_region_pixels` list.

    The returned `nuclei_region_pixels` list contains tuples, where each tuple represents the row and column
    coordinates of a "nuclei" region that matches with a cell in `ad_cell_pos`.

    Examples
    --------
    >>> import numpy as np

    >>> # Assuming `cm` is a numpy array representing identified nuclei from an H&E image.
    >>> # Also, `ad_cell_pos` is a numpy array containing cell pixel locations.
    >>> nuclei_pixels = map_nuclei_pixels(cm, ad_cell_pos)

    >>> # `nuclei_pixels` will be a list of tuples containing the matched nuclei pixel coordinates.
    >>> # Example: [(array([0, 1, 2]), array([3, 4, 5])), (array([10, 11, 12]), array([13, 14, 15])), ...]
    """

    tol = 1e-6
    regions = regionprops(cm)
    xy = np.array([r.centroid[::-1] for r in regions])
    nbrs = NearestNeighbors(n_neighbors=1).fit(xy)
    distance, indices = nbrs.kneighbors(ad_cell_pos)

    nuclei_region_pixels = np.zeros((len(ad_cell_pos), 2), dtype=object)
    for i, (idx, dist) in enumerate(zip(indices[:, 0], distance[:, 0])):
        if dist <= tol:
            nuclei_region_pixels[i] = (regions[idx].coords[:, 0], regions[idx].coords[:, 1])
        else:
            nuclei_region_pixels[i] = ([], [])
    nuclei_region_pixels = nuclei_region_pixels.tolist()
    return nuclei_region_pixels


def get_cells_voronoi(pos, ROI_tuple):
    """
    Calculate the Voronoi diagram for given cell positions and clip it within a region of interest.

    Parameters
    ----------
    pos : numpy array, shape (n_cells, 2)
        The pixel locations of all nuclei.
    ROI_tuple : tuple
        A tuple (xmin, ymin, width, height) representing the region of interest (ROI) where the Voronoi
        diagram should be computed and clipped.

    Returns
    -------
    cells_pixels : list
        A list of tuples, each containing the pixel coordinates of the Voronoi region associated with a cell.
        The format of the tuples is (row_coordinates, column_coordinates).
    vor : scipy.spatial.Voronoi object
        The Voronoi diagram object for all provided cells in the ROI.

    Notes
    -----
    This function calculates the Voronoi diagram for the given cell positions (`pos`) using `scipy.spatial.Voronoi`.
    The Voronoi diagram represents the division of space into regions, where each region contains one cell as its
    center point. The function then clips the Voronoi vertices within the region of interest specified by `ROI_tuple`.

    If a Voronoi region has no finite vertices or contains an infinite vertex (-1), the cell is considered as a
    single point, and its centroid coordinates are stored in the `cells_pixels` list.

    For cells with finite Voronoi regions, the function retrieves the vertices of the region, calculates the polygon
    (Voronoi cell) associated with each cell, and extracts the pixel coordinates within the cell's polygon. The pixel
    coordinates are stored in the `cells_pixels` list.

    The returned `cells_pixels` list contains tuples, where each tuple represents the row and column coordinates of the
    Voronoi region associated with a cell.

    Examples
    --------
    >>> import numpy as np

    >>> # Assuming `pos` is a numpy array containing cell pixel locations.
    >>> # Also, `ROI_tuple` specifies the region of interest (xmin, ymin, width, height).
    >>> cells_pixels, voronoi_obj = get_cells_voronoi(pos, ROI_tuple)

    >>> # `cells_pixels` will be a list of tuples containing the pixel coordinates of Voronoi regions.
    >>> # Example: [(array([0, 1, 2]), array([3, 4, 5])), (array([10, 11, 12]), array([13, 14, 15])), ...]
    """
    # Extract the region of interest (ROI) boundaries
    xmin, ymin = ROI_tuple[0], ROI_tuple[1]
    xmax, ymax = ROI_tuple[0] + ROI_tuple[2], ROI_tuple[1] + ROI_tuple[3]

    # Compute the Voronoi diagram for the cell positions
    vor = Voronoi(pos)

    # Clip the Voronoi vertices within the region of interest (ROI)
    vor.vertices = np.clip(vor.vertices, (xmin, ymin), (xmax, ymax))

    cells_pixels = []

    # Loop through each cell to calculate its Voronoi region
    for p in range(len(pos)):
        region_index_p = vor.point_region[p]
        region = vor.regions[region_index_p]

        # If Voronoi region has no finite vertices or contains an infinite vertex (-1),
        # consider the cell as a single point with its centroid coordinates.
        if len(region) == 0 or -1 in region:
            centr = tuple((
                np.array([pos[p][0].astype(int)]),
                np.array([pos[p][1].astype(int)])
            ))
            cells_pixels.append(centr)
            continue

        # Extract the Voronoi vertices for the cell's region and calculate the polygon (Voronoi cell)
        x, y = np.array([vor.vertices[i] for i in region]).T

        # Extract the pixel coordinates within the cell's polygon using skimage.draw.polygon function
        rr, cc = draw.polygon(y, x)

        # Uncomment the following lines if you want to remove pixels outside of the image bounds
        # in_box = np.where((ymin <= rr) & (rr <= ymax) & (xmin <= cc) & (cc <= xmax))
        # rr, cc = rr[in_box], cc[in_box]

        cells_pixels.append((rr, cc))

    return cells_pixels, vor


def get_cyto_mask(cell_mask, nuc_mask):
    return np.ma.masked_less_equal(cell_mask.data - nuc_mask.data, 0)


def get_mask_gt_threshold(masked, expr_thres, unify=True):
    masked_less_or_equal = np.ma.masked_less_equal(masked, expr_thres)

    if unify:
        # normalize it as we use single color
        masked_less_or_equal[masked_less_or_equal == False] = 1

    return masked_less_or_equal


def get_mask_ge_threshold(masked, expr_thres, unify=True):
    masked_less = np.ma.masked_less(masked, expr_thres)

    if unify:
        # normalize it as we use single color
        masked_less[masked_less == False] = 1

    return masked_less


def paint_regions(image_shape, matched_regions, cell_colors_list):
    """ Paint regions with values to color

    Parameters
    ----------
    image_shape : tuple
        shape of the image
    matched_regions : list
        list of regions to be painted. Each region is a 2d numpy array (npixels, 2)
    cell_colors_list : list
        list of values to be painted

    Returns
    -------
    filled : numpy array
        the painted image array.
    """
    # matched regions are supposed to be x, y
    # matching to image array, it should be x -> cc, y -> rr
    filled = np.ma.masked_all(image_shape)
    for i, r in enumerate(matched_regions):
        cc, rr = r
        filled[cc, rr] = cell_colors_list[i]

    return filled


def sample_pixels(
    pixels_pool,
    expression,
    expression_max,
    expression_min,
    expression_median=None,
    sample_more=2
):
    cell_size = len(pixels_pool[0])

    if cell_size == 0:
        return ([], [])

    def get_ratio_uniform(expression):
        if expression_max > expression_min:
            return min(
                (expression - expression_min) /
                (expression_max - expression_min), 1
            )
        elif expression_max == 0:
            return 0
        else:
            # expression_max == expression_min
            return 1

    def get_ratio_sigmoid(a=5):
        ratio = get_ratio_uniform(expression)
        ratio_med = get_ratio_uniform(expression_median)
        P = 1 + np.exp(a * (ratio_med - ratio))
        P = 1 / P

        return P

    try:
        n_sample = cell_size * min(
            1, sample_more * get_ratio_uniform(expression)
        )
        sampled = np.random.choice(
            range(cell_size), size=int(n_sample), replace=False
        )
    except:
        print(cell_size, get_ratio_uniform(expression))

    pixels_sampled = (pixels_pool[0][sampled], pixels_pool[1][sampled])
    return pixels_sampled


def get_cells_ROI(xy, ROI_tuple):
    x_min = ROI_tuple[0]
    x_max = ROI_tuple[0] + ROI_tuple[2]
    y_min = ROI_tuple[1]
    y_max = ROI_tuple[1] + ROI_tuple[3]
    xy_range = x_min, x_max, y_min, y_max

    return on_patch_rect(xy, xy_range)


def sample_n_paint_regions(
    image_shape,
    matched_regions,
    cells_on_patch,
    cell_colors_list,
    global_norm=False,
    sample_more=2,
    cmax=None,
    cmin=None,
    random_seed=None
):
    # be mindful that matched regions are supposed to be x, y
    # matching to image array, it should be x -> cc, y -> rr
    from statistics import median

    np.random.seed(random_seed)
    filled = np.ma.masked_all(image_shape)
    cell_colors_list = np.array(cell_colors_list)
    if global_norm:
        gex_max = cmax if cmax is not None else max(cell_colors_list)
        gex_min = cmin if cmin is not None else min(cell_colors_list)
        gex_med = median(cell_colors_list)
    else:
        gex_max = cmax if cmax is not None else max(
            cell_colors_list[cells_on_patch]
        )
        gex_min = cmin if cmin is not None else min(
            cell_colors_list[cells_on_patch]
        )
        gex_med = median(cell_colors_list[cells_on_patch])

    #print(f"normalized by {gex_min} and {gex_max}")
    for i, r in enumerate(matched_regions):
        if cells_on_patch[i]:
            gex = cell_colors_list[i]
            cc, rr = sample_pixels(
                r, gex, gex_max, gex_min, gex_med, sample_more=sample_more
            )
            filled[cc, rr] = gex

    return filled


def get_color(index, palette='tab20'):
    import matplotlib.colors as mc
    tabcolors = mc.mpl.colormaps.get_cmap(palette)
    return mc.rgb2hex(tabcolors.colors[index])


def get_nuclei_pixels(adata, segmentation_mask_path, save_path='nuc.npy'):
    """
    Extract nuclei regions.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix with cell positions stored in `adata.obsm['spatial']`.
    segmentation_mask_path : str
        The path to the segmentation mask file.
    save_path : str, optional (default: 'nuc.npy')
        The path to save the extracted nuclei regions.

    Returns
    -------
    nuclei_region_pixels : list
        A list of tuples, each containing the pixel coordinates of a "nuclei" region
        that matches with the given cell positions in `ad_cell_pos`.
        The format of the tuples is (row_coordinates, column_coordinates).
    """

    cm = load_npz(segmentation_mask_path).toarray()

    if 'seg_label' in adata.obs:
        # use seg_label to get nuclei pixels directly from the cell mask, no need to map
        nuclei_pixels_list = get_nuclei_pixels_from_label(cm, adata.obs['seg_label'].values)
    else:
        cell_pos = adata.obsm['spatial']
        # Note c_index starts from 0. so 0 <--> '1' in cell mask label (where '0' means null detection).
        nuclei_pixels_list = map_nuclei_pixels(cm, cell_pos)

    if save_path:
        #np.save(os.path.join(save_path, 'cell_labels.npy'), c_index, allow_pickle=True)
        np.save(save_path, np.array(nuclei_pixels_list, dtype=object), allow_pickle=True)

    return nuclei_pixels_list


def get_nuclei_pixels_from_label(cm, ad_cell_label):
    """Get nuclei pixels from label.

    Args:
        cm (np.ndarray): Cell mask.
        ad_cell_label (np.ndarray): Adherent cell label.

    Returns:
        list: Nuclei region pixels.

    Note:
        Label should be from 1 to n_cells (with skips). However if a rescaled `cm` is used, there is a chance that the ad_cell_label is not
        present in `cm`.
    """

    ad_cell_label = ad_cell_label.astype(int)
    # Label should be from 1 to n_cells
    assert np.all(ad_cell_label > 0), "Label should be from 1 to n_cells"

    regions = regionprops(cm)
    seg_label_indices = {r.label:i for i, r in enumerate(regions)}


    nuclei_pixels_list = []
    for l in ad_cell_label:
        if l not in seg_label_indices:
            nuclei_pixels_list.append(([], []))
        else:
            nuclei_pixels_list.append((regions[seg_label_indices[l]].coords[:, 0], regions[seg_label_indices[l]].coords[:, 1]))

    return nuclei_pixels_list


def process_color(adata, color=None):
    """
    Returns:
      values: 1D np.ndarray of length n_obs, where values[i] is the color for the i-th row
      mapping: dict[int,str] or None  (only for categorical → code→category)
    """
    if color is None:
        raise ValueError("`color` must be specified")

    # 1) gene expression layer
    if color in adata.var_names:
        vals = get_adata_layer_array(adata[:, color])[:, 0]
        return vals.astype(float), None

    # 2) obs column
    if color in adata.obs:
        col = adata.obs[color]
        if pd.api.types.is_numeric_dtype(col):
            return col.values.astype(float), None
        # categorical → codes + reverse mapping
        cats = pd.Categorical(col)
        codes = cats.codes.astype(int)
        revmap = dict(enumerate(cats.categories))
        return codes, revmap

    raise KeyError(f"'{color}' not found in adata.var_names or adata.obs")


def get_painted_mask(cm, adata, color, seg_label_key='seg_label'):
    """
    Paint `cm` (csr_matrix) by looking up each cell-ID in a dict.

    Returns:
      painted: same type as cm, but with .data or array values replaced by floats
      revmap: dict[int,str] or None  (for categorical legends)

    Examples:
        filled_mask_1 = get_painted_mask(mask, ad, gene, 'seg_label')
    """
    # 1) extract per-row values + optional categorical mapping
    seg_ids = adata.obs[seg_label_key].values  # shape = (n_cells,)
    values, revmap = process_color(adata, color)

    if len(values) != len(seg_ids):
        raise ValueError("Mismatch: seg_label array vs. color values length")

    # 2) build dict: ID → value
    id2val = {int(cid): float(val) for cid, val in zip(seg_ids, values)}

    # 3) remap:
    assert sp.issparse(cm), "Please input the cell mask as a sparse matrix"
    background_mask = (cm.A == 0)

    painted_sparse = cm.copy().astype(float)
    painted_sparse.data = np.fromiter(
        ( id2val.get(int(x)) for x in painted_sparse.data ),
        dtype=float,
        count=painted_sparse.data.size
    )

    # convert to dense for masking:
    painted = painted_sparse.toarray()

    # 4) mask out background (zeros)
    painted_masked = ma.MaskedArray(painted, background_mask)
    return painted_masked, revmap



legend_props = {
    'bbox_to_anchor': (1.04, 1),
    'borderaxespad': 0,
    'handleheight': 0.5,
    'handlelength': 0.5,
    'prop': {
        'size': 1
    },
}


