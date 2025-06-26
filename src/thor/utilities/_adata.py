import logging
import numpy as np
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def get_adata_layer(adata, layer_key=None):
    cnts = adata.X if layer_key is None else adata.layers[layer_key]
    return cnts


def get_adata_layer_array(adata, layer_key=None):
    cnts = get_adata_layer(adata, layer_key=layer_key)

    if issparse(cnts):
        cnts = cnts.toarray()

    return cnts


def get_library_id(adata):
    assert 'spatial' in adata.uns, "spatial not present in adata.uns"
    library_ids = adata.uns['spatial'].keys()
    try:
        library_id = list(library_ids)[0]
        return library_id
    except IndexError:
        logger.error('No library_id found in adata')


def get_scalefactors(adata, library_id=None):
    if library_id is None:
        library_id = get_library_id(adata)
    try:
        scalef = adata.uns['spatial'][library_id]['scalefactors']
        return scalef
    except IndexError:
        logger.error('scalefactors not found in adata')


def get_spot_diameter_in_pixels(adata, library_id=None):
    scalef = get_scalefactors(adata, library_id=library_id) 
    try:
        spot_diameter = scalef['spot_diameter_fullres']
        return spot_diameter    
    except TypeError:
        pass
    except KeyError:
        logger.error('spot_diameter_fullres not found in adata')


# maybe move to analysis module later
def convert_pixel_to_micron_visium(adata, res='fullres', spotDiameterinMicron=65):
    """Convert pixel to micron for Visium data.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    res: str    
        Resolution of the image. Valid options are 'fullres', 'hires', or 'lowres'.
    spotDiameterinMicron: int or float (read only)
        Spot diameter in micron. default: 65 for Visium data.
    
    Returns
    -------
    microPerPixel: float
        Micron per pixel in given resolution of the image.
    """

    spotDiameterinPixelFullRes = get_spot_diameter_in_pixels(adata)

    if res in ['lowres', 'hires']:
        scalef = get_scalefactors(adata)[f'tissue_{res}_scalef']
    else:
        scalef = 1

    microPerPixel = spotDiameterinMicron / spotDiameterinPixelFullRes / scalef

    return microPerPixel


def get_gene_symbols(adata, gene_names_list=None, gene_symbols=None):
    """Get gene symbols from adata. Suppose `gene_names_list` are the same as `adata.var_names`.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    gene_names: list
        List of gene names.
    
    Returns
    -------
    genes: list
        List of gene symbols.
    """

    assert gene_symbols is not None
    mapping = adata.var[gene_symbols].to_dict()
    
    genes = []
    for gene in gene_names_list:
        genes.append(mapping[gene])

    return genes


def split_obs_coords(obs_coords, n_rows, n_cols):
    """ split the obs_coords to n_rows * n_cols subsets according to the x and y coordinates.
    The returned booleans can be used to split the adata object to subsets. 
    Parameters:
    -----------
    obs_coords: numpy array
        Nuclei/cells centroids.

    Returns:
    --------
    subsets: Dictionary
        keys are names of the subsets, values are boolean arrays of the obs_coords.

    """

    intersec_col = np.linspace(
        obs_coords.min(axis=0)[0],
        obs_coords.max(axis=0)[0], n_cols + 1
    )
    intersec_row = np.linspace(
        obs_coords.min(axis=0)[1],
        obs_coords.max(axis=0)[1], n_rows + 1
    )

    logger.info(f"evenly split the data to ({n_rows}, {n_cols}).")

    subsets = {}
    for i in range(n_cols):
        x_min = intersec_col[i]
        x_max = intersec_col[i + 1]
        for j in range(n_rows):
            name_ = f"_{i}_{j}"

            y_min = intersec_row[j]
            y_max = intersec_row[j + 1]

            select = np.logical_and(
                np.logical_and(obs_coords[:, 1] < y_max, obs_coords[:, 1] > y_min),
                np.logical_and(obs_coords[:, 0] < x_max, obs_coords[:, 0] > x_min)
            )
            subsets[name_] = select
    return subsets


