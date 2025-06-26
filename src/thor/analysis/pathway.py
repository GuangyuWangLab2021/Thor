import logging

import numpy as np
import pandas as pd
import anndata

logger = logging.getLogger()

from typing import Optional
from thor.utils import spatial_smooth


def get_pathway_score(
        adata: anndata.AnnData, 
        layer: Optional[str] = None, 
        net_df: Optional[pd.DataFrame] = None, 
        smooth_radius: Optional[float] = 200) -> anndata.AnnData:
    """
    Calculate pathway score for each cell using over-representation analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    layer : str, optional
        Layer to use for the calculation
    net_df : pd.DataFrame, optional
        Dataframe with the network information. It should contain the following columns:
        - geneset: name of the geneset
        - genesymbol: name of the gene
        - weight: weight of the gene in the geneset (optional)
    smooth_radius : float, optional
        Radius for the smoothing. Default is 200.

    Returns
    -------
    AnnData
        Annotated data matrix with the pathway as the var_names and the pathway score as the X

    Notes
    -----
    This function calculates the pathway score for each cell using over-representation analysis. It uses the `dc.run_ora` function to perform the analysis and stores the results in the `adata.obsm` attribute. The pathway score is stored in the `adata.X` attribute and the pathway names are stored in the `adata.var_names` attribute.

    Examples
    --------
    >>> import pandas as pd
    >>> import scanpy as sc
    >>> import anndata as ad
    >>> from thor.analysis import get_pathway_score
    >>> adata = sc.datasets.pbmc3k_processed()
    >>> net_df = pd.read_csv('path/to/network.csv')
    >>> adata = get_pathway_score(adata, layer='counts', net_df=net_df)
    """
    import decoupler as dc

    if not isinstance(adata, anndata.AnnData):
        raise TypeError("adata must be an instance of anndata.AnnData")
    if net_df is not None and not isinstance(net_df, pd.DataFrame):
        raise TypeError("net_df must be an instance of pandas.DataFrame")
    if layer is not None and not isinstance(layer, str):
        raise TypeError("layer must be a string or None")

    if layer is not None:
        adata.X = adata.layers[layer]

    dc.run_ora(
        mat=adata,
        net=net_df,
        source='geneset',
        target='genesymbol',
        verbose=True,
        use_raw=False
    )

    # Store in a different key
    adata.obsm['msigdb_ora_estimate'] = adata.obsm['ora_estimate'].copy()
    adata.obsm['msigdb_ora_pvals'] = adata.obsm['ora_pvals'].copy()

    acts = dc.get_acts(adata, obsm_key='msigdb_ora_estimate')

    # We need to remove inf and set them to the maximum value observed
    acts_v = acts.X.ravel()
    max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
    acts.X[~np.isfinite(acts.X)] = max_e

    if smooth_radius is not None:
        x_smooth = spatial_smooth(acts.obsm['spatial'], acts.X, radius=smooth_radius)
        acts.layers['smoothed'] = x_smooth

    return acts


def get_celltype_specific_pathways(msigdb: pd.DataFrame, adata: anndata.AnnData, kw_list: list, smooth_radius: Optional[float] = 200) -> anndata.AnnData:
    """
    Get cell-type specific pathways from a given gene set database.

    Parameters
    ----------
    msigdb : pandas.DataFrame
        A pandas dataframe containing gene set collections and their corresponding gene symbols.
    adata : AnnData object
        An annotated data matrix containing gene expression data.
    kw_list : list
        A list of keywords to search for in the gene set collections.
    smooth_radius : float, optional
        Radius for the smoothing. Default is 200 in the spatial unit.

    Returns
    -------
    acts_bp_sel : AnnData object
        An annotated data matrix containing the pathway scores for each cell type.
    """
    bp = msigdb[msigdb['collection'] == 'go_biological_process']
    bp_sel = bp[bp['geneset'].apply(lambda p: np.any(list(map(lambda x: x in p, kw_list))))]

    bp_sel = bp_sel[~bp_sel.duplicated(['geneset', 'genesymbol'])]
    bp_sel.loc[:, 'geneset'] = [name.split('GOBP_')[1] for name in bp_sel['geneset']]

    acts_bp_sel = get_pathway_score(adata, layer=None, net_df=bp_sel, smooth_radius=smooth_radius)

    return acts_bp_sel


def get_collection_pathways(msigdb: pd.DataFrame, adata: anndata.AnnData, coll: str = 'hallmark', smooth_radius: Optional[float] = 200) -> anndata.AnnData:
    """Get pathway scores for a specific gene set collection.

    Parameters
    ----------
    msigdb : pandas.DataFrame
        A pandas dataframe containing gene set collections.
    adata : anndata.AnnData
        An AnnData object containing gene expression data.
    coll : str, optional
        The name of the gene set collection to use. Default is 'hallmark'.

    Returns
    -------
    anndata.AnnData
        An AnnData object containing pathway scores for the specified gene set collection.
    """
    bp_sel = msigdb[msigdb['collection'] == coll]
    bp_sel = bp_sel[~bp_sel.duplicated(['geneset', 'genesymbol'])]
    bp_sel.loc[:, 'geneset'] = ['_'.join(name.split('_')[1:]) for name in bp_sel['geneset']]
    acts_bp_sel = get_pathway_score(adata, layer=None, net_df=bp_sel, smooth_radius=smooth_radius)

    return acts_bp_sel


def get_tf_activity(adata: anndata.AnnData, layer: Optional[str] = None, net_df: Optional[pd.DataFrame] = None, smooth_radius: Optional[float] = 200) -> anndata.AnnData:
    """
    Infer TF activity using the CollecTRI database. This function calculates the pathway score for each cell using Univariate Linear Model. 

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    layer : str, optional
        Layer to use for the calculation
    net_df : pd.DataFrame, optional
        Dataframe with the network information. It should contain the following columns:
        - source: name of the TF
        - target: name of the regulated gene
        - weight: weight of the gene 
    smooth_radius : float, optional
        Radius for the smoothing. Default is 200.

    Returns
    -------
    AnnData
        Annotated data matrix with the TF as the var_names and the score as the X

    """
    import decoupler as dc

    if not isinstance(adata, anndata.AnnData):
        raise TypeError("adata must be an instance of anndata.AnnData")
    if net_df is not None and not isinstance(net_df, pd.DataFrame):
        raise TypeError("net_df must be an instance of pandas.DataFrame")
    if layer is not None and not isinstance(layer, str):
        raise TypeError("layer must be a string or None")

    if layer is not None:
        adata.X = adata.layers[layer]

    dc.run_ulm(
        mat=adata,
        net=net_df,
        source='source',
        target='target',
        weight='weight',
        verbose=True,
        use_raw=False
    )

    # Store in a different key
    adata.obsm['msigdb_ulm_estimate'] = adata.obsm['ulm_estimate'].copy()
    adata.obsm['msigdb_ulm_pvals'] = adata.obsm['ulm_pvals'].copy()

    acts = dc.get_acts(adata, obsm_key='msigdb_ulm_estimate')

    # We need to remove inf and set them to the maximum value observed
    acts_v = acts.X.ravel()
    max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
    acts.X[~np.isfinite(acts.X)] = max_e

    if smooth_radius is not None:
        x_smooth = spatial_smooth(acts.obsm['spatial'], acts.X, radius=smooth_radius)
        acts.layers['smoothed'] = x_smooth

    return acts

