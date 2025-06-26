from typing import Dict, Optional
import logging

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from sklearn.neighbors import NearestNeighbors

from thor.utils import get_adata_layer_array, get_library_id, update_kwargs_exclusive, get_scalefactors, on_patch_rect

logger = logging.getLogger(__name__)

def precompute_nearest_pairs_distances(
    adata: anndata.AnnData,
    cutoff: float,
    spatial_key: Optional[str] = "spatial"
) -> None:
    """
    Precompute pairwise distances between cells within a given cutoff distance using the NearestNeighbors class from scikit-learn.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cutoff
        The cutoff distance for the nearest neighbor search.
    spatial_key
        The key of the spatial coordinates in adata.obsm.

    Returns
    -------
    None

    Examples
    --------
    >>> import scanpy as sc
    >>> import thor
    >>> adata = sc.datasets.visium_sge()
    >>> distance = 500 # distance cutoff in microns
    >>> microns_per_pixel = 0.25
    >>> distance_pixel = distance / microns_per_pixel
    >>> thor.analy.precompute_nearest_pairs_distances(adata, cutoff=distance_pixel)
    """
    X = adata.obsm[spatial_key]
    neigh = NearestNeighbors(radius=cutoff)
    neigh.fit(X)
    A = neigh.radius_neighbors_graph(X, mode="distance")
    adata.obsp["spatial_distance"] = A


def get_pathway(pathway_name: str, pathways_df: pd.DataFrame, name_col_index: int = 2) -> pd.DataFrame:
    """
    Get a subset of the pathways database containing the specified pathway.

    Parameters
    ----------
    pathway_name : str
        The name of the pathway to retrieve.
    pathways_df : pandas.DataFrame
        The database of pathways to search.
    name_col_index : int, optional
        The index of the column containing the pathway names in the database.
        Default is 2.

    Returns
    -------
    pandas.DataFrame
        A subset of the pathways database containing the specified pathway.
    """
    pathway_name = pathway_name.upper()
    sub_ligrec = pathways_df[pathways_df.iloc[:, name_col_index].str.upper().isin([pathway_name])]
    return sub_ligrec


def split_pathways(pathways_df: pd.DataFrame, name_col_index: int = 2) -> Dict[str, pd.DataFrame]:
    """
    Split the pathways database into a dictionary of pathways.

    Parameters
    ----------
    pathways_df : pandas.DataFrame
        The DataFrame containing the pathways column.
    name_col_index : int, optional
        The index of the column containing the pathway names. Default is 2.

    Returns
    -------
    dict
        A dictionary of pathways, where the keys are the pathway names and the values are the corresponding DataFrames.
    """
    pathways = pathways_df.iloc[:, name_col_index].unique()

    pathways_dict = {
        p: get_pathway(p, pathways_df, name_col_index=name_col_index)
        for p in pathways
    }

    return pathways_dict


def add_image_row_col(adata, spatial_key="spatial"):
    adata.obs.loc[:, "imagerow"] = adata.obsm[spatial_key][:, 0]
    adata.obs.loc[:, "imagecol"] = adata.obsm[spatial_key][:, 1]


def prepare_adata(adata, layer=None, img_key="hires", gene_symbols_key="feature_name"):

    ad_sym = adata.copy()
    try:
        del ad_sym.raw
    except:
        pass

    if gene_symbols_key in ad_sym.var.columns:
        ad_sym.var_names = ad_sym.var.loc[:, gene_symbols_key]

    # use specified layer
    ad_sym.X = get_adata_layer_array(adata, layer_key=layer)
    try:
        del ad_sym.layers
    except:
        pass

    # set image to use
    library_id = get_library_id(adata)
    ad_sym.uns["spatial"][library_id]["use_quality"] = img_key
    add_image_row_col(ad_sym)

    # filter cells with no expression
    ad_sym = ad_sym[ad_sym.X.sum(axis=1) > 0]

    # Remove incompatible genes
    def remove_incompatible_genes(adata, incompatible_gene_characters_list=["_"]):
        return adata[:, ~adata.var.index.str.contains("|".join(incompatible_gene_characters_list))]

    ad_sym = remove_incompatible_genes(ad_sym)
    return ad_sym


def run_commot(adata, region=None, gene_symbols_key="feature_name", **kwargs):
    """
    Run the cell-cell communication analysis using the modified `COMMOT <https://doi.org/10.1038/s41592-022-01728-4>`_ method. 
    This is a wrapper function the :py:func:`commot.tl.spatial_communication`. 
    The function first prepares the data by filtering and processing the input data, and precompute the cell-cell distances matrix
    more efficiently, supporting sparse matrix.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    region : :py:class:`list`, optional
        The region to analyze in the format [left, right, lower, upper]. left < right and lower < upper.
    gene_symbols_key : :py:class:`str`, optional
        The key for gene symbols in adata.var. Default is "feature_name".
    kwargs: :py:class:`dict`, optional
        Additional keyword arguments for :py:func:`commot.tl.spatial_communication`.

    Returns
    -------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.

    See Also
    --------
    :py:func:`commot.tl.spatial_communication`
    """
    import commot as ct


    if region is not None:
        region = np.array(region) 
        xy = adata.obsm["spatial"]

        ad_selected = adata[on_patch_rect(xy, region)].copy()
    else:
        ad_selected = adata.copy()


    ad_selected = prepare_adata(ad_selected, gene_symbols_key=gene_symbols_key)
    commot_kwargs = update_kwargs_exclusive(ct.tl.spatial_communication, kwargs)

    precompute_nearest_pairs_distances(ad_selected, commot_kwargs["dis_thr"])
    
    ct.tl.spatial_communication(
        ad_selected,
        **commot_kwargs
    )
        
    return ad_selected


def plot_commot(adata, region=None, **kwargs):
    """
    Plot the cell-cell communication analysis results on a spatial plot.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    region : list, optional
        The region to plot in the format [left, right, lower, upper]. left < right and lower < upper.
    kwargs
        Additional keyword arguments for the `cc.tl.communication_direction` and `cc.pl.plot_cell_communication` functions.
    """

    import commot as ct
    from matplotlib import pyplot as plt


    communication_direction_kwargs = update_kwargs_exclusive(ct.tl.communication_direction, kwargs)
    ct.tl.communication_direction(adata, **communication_direction_kwargs)

    plot_kwargs = update_kwargs_exclusive(ct.pl.plot_cell_communication, kwargs)
    ax = ct.pl.plot_cell_communication(adata, **plot_kwargs)

    if region is not None:
        scalef = get_scalefactors(adata)['tissue_hires_scalef']
        lower = region[2] * scalef
        higher = region[3] * scalef
        left = region[0] * scalef
        right = region[1] * scalef

        ax.set_xlim(left, right)
        ax.set_ylim(lower, higher)

    ax.invert_yaxis()
    plt.show()


def commot_to_dynamo(ad_commot, pathway, database, lr="sender", basis="spatial"):
    """Convert the COMMOT output to dynamo format.

    Parameters
    ----------
    ad_commot : :class:`~anndata.AnnData`
        Annotated data matrix. The COMMOT results are stored in `obsm`.
    pathway : str
        The pathway name.
    database : str
        The database name.
    lr : str, optional
        The ligand-receptor direction. Default is "sender".
    basis : str, optional
        The low-dimensional embedding used for COMMOT analysis. Default is "spatial" for spatial transcriptomics data.
    
    Returns
    -------
    ad_sig : :class:`~anndata.AnnData`
        Annotated data matrix. The ligand-receptor directions are stored in `obsm`.

    Notes
    -----
    Headsup! The function is not tested on multiple datasets yet.
    """
    
    ligrec_obsm_key = f"commot-{database}-sum-{lr}"

    # skip summary
    molecules = ad_commot.obsm[ligrec_obsm_key].columns.map(lambda x: len(x.split("-"))>2)
    ad_sig = sc.AnnData(ad_commot.obsm[ligrec_obsm_key].loc[:, molecules].iloc[:, :-1])

    # chage key names in dynamo convention
    ad_sig.obsm[f"X_{basis}"] = ad_commot.obsm[basis]
    ad_sig.obsm[f"velocity_{basis}"] = ad_commot.obsm[f"commot_{lr}_vf-{database}-{pathway}"]
    ad_sig.obs = ad_commot.obs

    return ad_sig
