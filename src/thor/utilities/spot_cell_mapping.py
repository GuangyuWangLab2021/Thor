import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy, variation
from scipy.sparse import csr_matrix
from collections import Counter
from statistics import median

from ._math import row_normalize_sparse
from ._adata import get_spot_diameter_in_pixels

logger = logging.getLogger(__name__)


def adata_spot_to_cell(adata_spot, node_feat, obs_features=None, margin=2):
    """ Assigning ST of segmented cells by the nearest spots (1 or 0 for each cell). If the nearest spot is within the margin, the cell is assigned to the spot.
    
    Parameters
    ----------
    adata_spot: :class:`anndata.AnnData`
        The AnnData object containing the spot data.
    node_feat: :class:`pandas.DataFrame`
        The DataFrame containing the cell features.
    obs_features: :py:class:`list`
        The sublist of cell features (node_feat) to be kept in the resulting cell adata. Default is to keep all features.
    margin: :py:class:`float`
        The distance threshold to find the closest spot specified in the unit of spot radius. Default: 2.

    Returns
    -------
    adata_cell: :class:`anndata.AnnData`
        The AnnData object containing the cell data.
    spot_to_cell: :py:class:`list`
        The list of spot indices that are mapped to cells.
    """

    logger.info(
        "The first two columns in the node_feat DataFrame need to be consistent with the spatial coordinates from obsm['spatial']."
    )

    spot_pos = pd.DataFrame(
        adata_spot.obsm['spatial'],
        columns=['x', 'y'],
        index=adata_spot.obs.index
    )

    nbrs = NearestNeighbors(n_neighbors=1).fit(spot_pos[['x', 'y']].values)
    distances, indices = nbrs.kneighbors(node_feat.iloc[:, :2].values)

    spot_diameter = get_spot_diameter_in_pixels(adata_spot)

    if spot_diameter is not None:
        logger.info(f"Mapping cells to the closest spots within {margin} x the spot radius")
        spot_radius = spot_diameter * 0.5

        selected = (distances < margin * spot_radius)
        selected = selected[:, 0]

        # those are spot indices 
        indices = indices[selected]
        node_feat = node_feat[selected]

    nearest_spot = spot_pos.iloc[indices.T[0], :]
    spot_to_cell = nearest_spot.index.values

    adata_cell = adata_spot[spot_to_cell]
    adata_cell.obs['spot_barcodes'] = adata_cell.obs.index.tolist()
    adata_cell.obsm['spatial'] = node_feat.iloc[:, :2].values


    if obs_features is None:
        obs_features = node_feat.columns
    else:
        obs_features = get_list_in_reference(obs_features, node_feat.columns)

    adata_cell.uns['cell_image_props'] = np.array(obs_features, dtype=object)

    df = pd.concat(
        [
            adata_cell.obs.set_index(node_feat.index),
            node_feat[obs_features]
        ],
        axis=1
    )

    # using adata_index
    df['seg_label'] = df.index
    adata_cell.obs = df.set_index(adata_cell.obs.index)

    # delete var keys carried over from the spot data
    unkept_var_keys = ['spatially_variable', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'vae_genes']
    for var_key in unkept_var_keys:
        if var_key in adata_cell.var:
            del adata_cell.var[var_key]
    if 'hvg' in adata_cell.uns:
        del adata_cell.uns['hvg']

    return adata_cell, spot_to_cell


def generate_cell_adata(cell_features_path, spot_adata_path, obs_features=None, mapping_margin=10):
    """
    """
    node_feat = pd.read_csv(cell_features_path, index_col=0)
    adata = sc.read_h5ad(spot_adata_path)
    try:
        adata.obsm['spatial'] = np.asarray(adata.obsm['spatial'], dtype='float')
    except:
        pass

    # Here we need to double check the cells segmented from the image
    # and the spatial coordinate match (no rotation, flipping, scaling etc)
    logger.info("Please check alignment of cells and spots")

    adata_cell, spot_to_cell = adata_spot_to_cell(adata, node_feat, obs_features=obs_features, margin=mapping_margin)
    superimpose_spot_adata(adata.obsm['spatial'], adata_cell.obsm['spatial'])

    # obs index is now changed from spot barcode to spot barcode + '-num'
    adata_cell.obs_names_make_unique()

    return adata_cell


def superimpose_spot_adata(spot_pos, cell_pos):
    fig = plt.figure(dpi=200, figsize=(5, 5))
    plt.scatter(spot_pos[:, 0], spot_pos[:, 1], s=25*0.4, facecolors='none', edgecolors='gray', zorder=1)
    plt.scatter(cell_pos[:, 0], cell_pos[:, 1], s=1*0.4, alpha=0.5, zorder=2)
    fig.gca().invert_yaxis()

    plt.show()


def get_list_in_reference(lst, lst_ref):
    """Get the intersection of two lists. Preserving the order of the first list.

    Parameters
    ----------
    lst: :py:class:`list`
        The list to be filtered.
    lst_ref: :py:class:`list`
        The reference list.

    Returns
    -------
    lst_in: :py:class:`list`
        The intersection of the two lists.
    """

    lst_in = []
    lst_out = []
    for i in lst:
        if i in lst_ref:
            lst_in.append(i)
        else:
            lst_out.append(i)
    if len(lst_out) > 0:
        logger.warning(f'{lst_out} excluded')
    if len(lst_in) == 0:
        logger.error(f'None included. Setting to {lst_ref}')
        return lst_ref
    return lst_in
        

def get_labels_entropy(labels):
    """Calculate the entropy of the labels.
    Parameters
    ----------
    labels: :py:class:`list` of :py:class:`str`
        The labels of the cells.
    
    Returns
    -------
    :py:class:`float`
        The entropy of the labels.
    """
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts)


def get_spot_heterogeneity_entropy(df, cell_group_key='clusters', spot_identifier='spot_barcodes'):
    """Calculate the entropy of the cell groups in each spot.
    
    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        cell-level DataFrame
    cell_group_key: :py:class:`str`
        The column name of the cell group. Default: 'clusters'
    spot_identifier: :py:class:`str`
        The column name of the spot identifier. Default: 'spot_barcodes'
    
    Returns
    -------
    :class:`pandas.Series`
        The heterogeneity of the cell groups in each spot casted to all cells with the same index of the input Pandas DataFrame.
    """
    entropy_series = df.groupby([spot_identifier]
                                   )[cell_group_key].agg(get_labels_entropy)
    entropy_series.rename('spot_heterogeneity_entropy', inplace=True)

    df_merged = pd.merge(df, entropy_series, left_on=spot_identifier, right_index=True, how='left')
    hetero_series = df_merged['spot_heterogeneity_entropy']
    return hetero_series


def get_spot_heterogeneity_cv(df, cell_features_list, spot_identifier):
    """Calculate the entropy of the cell groups in each spot.
    
    Parameters
    ----------
    df: pd.DataFrame
        cell-level dataframe
    cell_features_list: list of str
        The list of cell features to calculate the coefficient of variation. 
    spot_identifier: str
        The column name of the spot identifier. 
    
    Returns
    -------
    hetero_series: pd.Series
        The heterogeneity of the cell groups in each spot casted to all cells with the same index of the input Pandas DataFrame.
    """

    # Note variation() is to compute the coefficient of variation (CV) = std / mean in scipy.stats
    # In case that the mean values of features are near zero, doing minmax scaling here.

    
    #df_norm = pd.DataFrame(MinMaxScaler().fit_transform(df[cell_features_list]), columns=cell_features_list, index=df.index)
    #df_norm[spot_identifier] = df[spot_identifier]
    df_norm = df.copy()
    cv_mean_series = df_norm.groupby([spot_identifier]
                                   )[cell_features_list].agg(variation).mean(axis=1)

    cv_mean_series.rename('spot_heterogeneity_cv', inplace=True)

    df_merged = pd.merge(df_norm, cv_mean_series, left_on=spot_identifier, right_index=True, how='left')
    
    hetero_series = df_merged['spot_heterogeneity_cv']
    return hetero_series


# This function can be replaced by pd.DataFrame.groupby().agg() for performance and readability
def estimate_spot_from_cells(x_cell, cellxspot, mapping_method='mean'):
    """
    cellxspot is sparse matrix
    """

    assert mapping_method in ['mean', 'sum']

    # When you transpose a csr_matrix, it becomes a dense array...
    spotxcell = csr_matrix(cellxspot.T)

    if mapping_method == 'mean':
        spotxcell = row_normalize_sparse(spotxcell)

    x_spot = spotxcell * x_cell
    return x_spot


def distribute_to_cells_from_spot(x_spot, cellxspot):
    """
    cellxspot is sparse matrix
    """
    x_cell = cellxspot * x_spot
    return x_cell


def downsample(ad, cells_per_spot=1):
    """ Downsample the cells per spot to a given number.

    Parameters
    ----------
    ad: AnnData
        The AnnData object containing the cell data. Expecting the spot_barcodes column in obs.
    cells_per_spot: int
        The number of cells to be kept per spot. Default: 1.

    Returns
    -------
    ad_sub: AnnData
        The AnnData object containing the downsampled cell data. The original adata is unchanged.

    """
    spot_barcodes = ad.obs.spot_barcodes.drop_duplicates()
    spot_barcodes = list(spot_barcodes)
    selected_barcodes = spot_barcodes

    for i in range(1, cells_per_spot):
        new_selection = list(map(lambda x: str(x)+f"-{i}", spot_barcodes))
        selected_barcodes += new_selection
    ad_sub = ad[ad.obs.index.isin(selected_barcodes), :].copy()
    return ad_sub
