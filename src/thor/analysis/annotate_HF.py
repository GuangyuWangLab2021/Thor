import scanpy as sc
from sklearn.cluster import KMeans
import pandas as pd
import anndata

def get_fraction(x: pd.Series, thres: float = 0.) -> object:
    """
    Get the most frequent value in a pandas Series or DataFrame column.

    Parameters
    ----------
    x : pandas.Series or pandas.DataFrame column
        The column to compute the most frequent value from.
    thres : float, optional
        The threshold for the fraction of the most frequent value in the column.
        If the fraction is less than or equal to `thres`, return None.
        Default is 0.

    Returns
    -------
    most_prob : object or None
        The most frequent value in the column.
        If the fraction of the most frequent value is less than or equal to `thres`, return None.
    """
    frac = pd.value_counts(x).to_dict()
    most_prob = sorted(frac.items(), key=lambda x: x[1], reverse=True)[0][0]

    if frac[most_prob] / sum(frac.values()) > thres:
        return most_prob
    else:
        return None


def annotate(ad: anndata.AnnData, name: str, anno_key: str) -> None:
    """
    Annotate the given Anndata object with the fraction of cells in each cluster that have a certain annotation.

    Parameters
    ----------
    ad : anndata.AnnData
        An Anndata object to be annotated.
    name : str
        The name of the cluster column in the obs dataframe of the Anndata object.
    anno_key : str
        The name prefix of the annotation.

    Returns
    -------
    None
    """
    cluster_anno_dict = ad.obs.groupby(f'cluster_{name}'
                                      )[anno_key].agg(get_fraction).to_dict()
    ad.obs[f'{anno_key}_{name}'] = ad.obs[f'cluster_{name}'].map(
        cluster_anno_dict
    )


def annotate_by_expression(
    ad1: anndata.AnnData,
    name: str = 'cell_type_updated',
    ref_annotate_key: str = 'cell_type',
    layer: str = 'snn_20_sample1',
    resolution: int = 1000,
    random_state: int = 0
) -> None:
    """
    Annotate cells by clustering based on gene expression data.

    Parameters
    ----------
    ad1 : anndata.AnnData
        Annotated data matrix.
    name : str, optional
        Name of the new annotation column, by default 'cell_type_updated'.
    ref_annotate_key : str, optional
        Name of the reference annotation column, by default 'cell_type'.
    layer : str, optional
        Name of the layer to use for clustering, by default 'snn_20_sample1'.
    resolution : int, optional
        Number of clusters to generate, by default 1000.
    random_state : int, optional
        Random seed for reproducibility, by default 0.

    Returns
    -------
    None
    """
    xfeat = sc.tl.pca(ad1.layers[layer])
    kmeans = KMeans(n_clusters=resolution, random_state=random_state,
                    n_init=1).fit(xfeat)
    ad1.obs[f'cluster_{name}'] = kmeans.labels_.astype(str)
    annotate(ad1, name, ref_annotate_key)
