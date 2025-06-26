import logging
import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from shapely.geometry import Polygon
import scanpy as sc
from typing import Tuple, Optional, List
from anndata import AnnData

from thor.pl import deg
from thor.utils import (
    convert_pixel_to_micron_visium, get_adata_layer, get_region, row_normalize
)
from ._utils import read_polygon_ROI

logger = logging.getLogger(__name__)


def get_expression_fc(arr_base: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """
    Calculates the log2 fold change of gene expression between two arrays.

    Parameters
    ----------
    arr_base : np.ndarray
        The baseline array of gene expression values.
    arr : np.ndarray
        The array of gene expression values to compare to the baseline.

    Returns
    -------
    np.ndarray
        The 1d array of log2 fold change values for each gene.

    Notes
    -----
    The input arrays should have the following dimensions:
    - rows correspond to cells
    - columns correspond to genes

    This function applies cell normalization to ensure that all cells have equal total expression.
    """
    tiny = 1E-6
    # applying cell normalization (all cells have equal total expression)
    arr = row_normalize(arr)
    arr = arr.mean(axis=0, keepdims=True)

    arr_base = row_normalize(arr_base)
    arr_base = arr_base.mean(axis=0, keepdims=True)

    return np.log2(arr + tiny) - np.log2(arr_base + tiny)


def analyze_gene_expression_gradient(
    adata,
    img_key: str = "fullres",
    layer_key: str = None,
    range_from_edge: Tuple[int, int] = [-150, 150],
    baseline_from_edge: Tuple[int, int] = [-150, -100],
    bin_size: int = 30,
    n_top_genes: int = 10,
    min_mean_gene_expression: float = 0.1,
    tmpout_path: str = 'geg.json',
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Analyze gene expression against a baseline in a selected region of interest (ROI).

    Parameters
    ----------
    adata : anndata.Anndata
        The input data matrix.
    img_key : str, optional
        The key for the image where the json ROI is drawn. Default is "fullres". Valid options are "lowres" (unlikely), "hires" (unlikely), and "fullres".
    layer_key : str, optional
        The key for the layer data in `adata.layers`.
    range_from_edge : Tuple[int, int], optional
        The range of the ROI from the edge of the image.
    baseline_from_edge : Tuple[int, int], optional
        The range of the baseline from the edge of the image.
    bin_size : int, optional
        The size of the bins for computing the differential gene expression.
    n_top_genes : int, optional
        The number of top genes to plot.
    min_mean_gene_expression : float, optional
        The minimum mean gene expression to filter genes.
    tmpout_path : str, optional
        The path to the temporary output file.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, np.ndarray]
        A tuple containing the differential gene expression dataframe, the ROI polygon, and the baseline polygon.
   
    """
    roi_shape = read_polygon_ROI(tmpout_path, adata, img_key=img_key)

    # Filter genes whose average expression < 0.1
    adata_roi = get_region(adata, roi_shape)
    gene_filter = adata_roi.layers[layer_key].mean(
        axis=0
    ) > min_mean_gene_expression

    de_df, roi_polygon, baseline_polygon = compute_dge_against_baseline(
        adata[:, gene_filter],
        roi_shape,
        layer_key=layer_key,
        range_from_edge=range_from_edge,
        baseline_from_edge=baseline_from_edge,
        bin_size=bin_size
    )

    de_df_to_smooth = de_df.copy()
    fit_win = min(20, de_df_to_smooth.shape[0])
    smoothed_arr = savgol_filter(
        de_df_to_smooth, fit_win, 5, mode='nearest', axis=0
    )
    smoothed_de_df = pd.DataFrame(smoothed_arr, columns=de_df_to_smooth.columns)
    smoothed_de_df.set_index(de_df.index, inplace=True)

    genes = get_top_genes(smoothed_de_df, n=n_top_genes)
    deg(
        data=smoothed_de_df,
        baseline_from_edge=baseline_from_edge,
        genes=genes,
        cmaps=["Oranges", "Blues"],
        lw=5,
        annotate=True,
        text_offset_x=0.08,
        text_offset_y=0,
        text_size=8,
    )
    return de_df, roi_polygon, baseline_polygon


def compute_dge_between_regions(
    ad_r1: AnnData,
    ad_r2: AnnData,
    test_method: str = 't-test',
    pval_cutoff: float = 0.01,
    log2fc_min: float = 1,
    log2fc_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute differential gene expression (DGE) between two regions.

    Parameters
    ----------
    ad_r1 : AnnData
        Annotated data matrix for region 1.
    ad_r2 : AnnData
        Annotated data matrix for region 2.
    test_method : str, optional
        Statistical test method to use for DGE analysis. Valid options are 'logreg', 't-test', 'wilcoxon', 't-test_overestim_var'. Default is 't-test'.
    pval_cutoff : float, optional
        P-value cutoff for significance, by default 0.01.
    log2fc_min : float, optional
        Minimum log2 fold change for significance, by default 1.
    log2fc_max : float, optional
        Maximum log2 fold change for significance, by default None.

    Returns
    -------
    np.ndarray
        A numpy array containing the DGE results.

    """
    adatas = {'r1': ad_r1, 'r2': ad_r2}
    ad = sc.concat(adatas, axis=0, uns_merge=None, label='region')
    sc.tl.rank_genes_groups(
        ad, groupby='region', method=test_method, key_added=test_method
    )
    tt = sc.get.rank_genes_groups_df(
        ad, group=None, key=test_method, pval_cutoff=pval_cutoff, log2fc_min=log2fc_min, log2fc_max=log2fc_max
    )
    return tt


def compute_dge_against_baseline(
    adata: AnnData,
    roi_shape: List[Tuple[float, float]],
    layer_key: Optional[str] = None,
    range_from_edge: List[float] = [-150, 150],
    baseline_from_edge: List[float] = [-150, -100],
    bin_size: float = 30
) -> Tuple[pd.DataFrame, Polygon, Polygon]:
    """
    Compute differential gene expression against a baseline region of interest.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    roi_shape : List[Tuple[float, float]]
        List of (x, y) coordinates defining the shape of the region of interest in pixels (full resolution).
    layer_key : str, optional
        Key for the layer of the adata object to use for expression data. If None, use X.
    range_from_edge : List[float], optional
        Distance range from the edge of the region of interest in micrometers (microns). Default is [-150, 150].
    baseline_from_edge : List[float], optional
        Distance range from the edge of the region of interest to define the baseline region in micrometers (microns).
        Default is [-150, -100].
    bin_size : float, optional
        Bin size in micrometers (microns) to use for computing differential expression. Default is 30.

    Returns
    -------
    Tuple[pd.DataFrame, Polygon, Polygon]
        Tuple containing the differential gene expression dataframe, the polygon defining the region of interest,
        and the polygon defining the baseline region.
    """
    poly = Polygon(roi_shape)

    pixel_to_micron = convert_pixel_to_micron_visium(adata, res="fullres")
    micron_to_pixel = 1 / pixel_to_micron
    baseline_polygon = poly.buffer(
        baseline_from_edge[1] * micron_to_pixel
    ) - poly.buffer(baseline_from_edge[0] * micron_to_pixel)
    baseline_ad = get_region(adata, baseline_polygon)

    roi_polygon = poly.buffer(
        range_from_edge[1] * micron_to_pixel
    ) - poly.buffer(range_from_edge[0] * micron_to_pixel)
    roi_ad = get_region(adata, roi_polygon)

    baseline_expr = get_adata_layer(baseline_ad, layer_key)
    logger.info(
        f"{baseline_ad.shape[0]} ({baseline_ad.shape[0]/roi_ad.shape[0]*100}%) cells in baseline compared to the whole selected region of interest."
    )

    #direction = np.sign(range_from_edge)
    coverage = range_from_edge[1] - range_from_edge[0]
    n_bins = math.ceil(coverage / bin_size)

    log2fc = {}
    bin_size_in_pixel = bin_size * micron_to_pixel

    for n in range(n_bins):
        dist = n * bin_size + range_from_edge[0]
        dist_in_pixel = dist * micron_to_pixel
        logger.info(f"Distance to edge: {dist} um")
        selected = poly.buffer(
            dist_in_pixel + 0.5 * bin_size_in_pixel
        ) - poly.buffer(dist_in_pixel - 0.5 * bin_size_in_pixel)
        selected_ad = get_region(adata, selected)
        if selected_ad.shape[0] < 1:
            logger.warning(f"No cells at distance {dist} to edge.")
            continue
        selected_expr = get_adata_layer(selected_ad, layer_key)
        cellsxgenes_expr = get_expression_fc(baseline_expr, selected_expr)
        log2fc[dist] = np.mean(cellsxgenes_expr, axis=0)
    de_df = pd.DataFrame(log2fc, index=list(adata.var.index)).T.reset_index()
    de_df.rename(columns={"index": "distance"}, inplace=True)
    de_df.set_index("distance", inplace=True)
    #de_df = de_df.rolling(5, min_periods=1).mean()
    return de_df, roi_polygon, baseline_polygon


def get_top_genes(de_df: pd.DataFrame, n: int = 10) -> Tuple[List[str], List[str]]:
    """
    Get the top up-regulated and down-regulated genes from a differential expression DataFrame.

    Parameters
    ----------
    de_df : pd.DataFrame
        A pandas DataFrame containing differential expression data.
    n : int, optional
        The number of genes to return for each category, by default 10.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists of gene names, the first for up-regulated genes and the second for down-regulated genes.
    """
    ddf = de_df.iloc[-1, 1:]
    sorted_ddf = ddf.sort_values()

    genes_down = sorted_ddf[:n].index
    genes_down = [g for g in genes_down if ddf[g] < 0][::-1]

    genes_up = sorted_ddf[-1 * n:].index
    genes_up = [g for g in genes_up if ddf[g] > 0][::-1]

    genes = [genes_up, genes_down]
    return genes
