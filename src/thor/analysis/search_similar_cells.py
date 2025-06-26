import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

def get_feature(adata, obs_features_list):
    ad_hist = ad.AnnData(adata.obs[obs_features_list])
    ad_hist.obsm['spatial'] = adata.obsm['spatial'] 

    # important - feature values are not scaled, so need to scale them before PCA
    sc.pp.scale(ad_hist)

    sc.tl.pca(ad_hist, n_comps=3)

    adata.obs['hist_PC1'] = ad_hist.obsm['X_pca'][:,0]
    adata.obs['hist_PC2'] = ad_hist.obsm['X_pca'][:,1]
    adata.obs['hist_PC3'] = ad_hist.obsm['X_pca'][:,2]

    sc.tl.pca(adata, n_comps=3)

    adata.obs['expr_PC1'] = adata.obsm['X_pca'][:,0]*0.5
    adata.obs['expr_PC2'] = adata.obsm['X_pca'][:,1]*0.5
    adata.obs['expr_PC3'] = adata.obsm['X_pca'][:,2]*0.5

    return adata

def pearson_corr(np_arr_2d, indices_list):
    """ Calculate the pearson correlation between a list of rows and all rows in a 2d numpy array (each row is a cell)

    Parameters
    ----------
    np_arr_2d: 2d numpy array
        The 2d numpy array to calculate the pearson correlation on
    indices_list: list
        The list of indices of the rows to calculate the pearson correlation on

    Returns
    -------
    res_arr: 2d numpy array
        The pearson correlation between the rows at indices_list and all rows in np_arr_2d
    """
    demeaned = np_arr_2d - np_arr_2d.mean(axis=1)[:, None]
    row_norms = np.sqrt((demeaned ** 2).sum(axis=1))[:, None]
    res = np.dot(demeaned, demeaned.T[:, indices_list])
    res = res / row_norms
    res = res / row_norms[indices_list, 0]

    return res


def get_corr_ROI(adata, ad_ROI):
    """ Calculate the pearson correlation between a subset of cells and all other cells

    Parameters
    ----------
    adata: AnnData
        The AnnData object containing all cells
    ad_ROI: AnnData
        The AnnData object containing the subset of cells

    Returns
    -------
    corr_ROI: pandas DataFrame
        The pearson correlation between the subset of cells and all other cells. Rows are all cells and columns are the subset of cells.
    """
    df = adata.obs[['hist_PC1','hist_PC2','hist_PC3','expr_PC1','expr_PC2','expr_PC3']]
    ROI_iloc = df.index.get_indexer(ad_ROI.obs.index)
    corr_ROI = pearson_corr(df.values, ROI_iloc)
    corr_ROI = pd.DataFrame(corr_ROI, columns=ad_ROI.obs.index, index=adata.obs.index)
    return corr_ROI


def select_cell(adata, ad_ROI, thresh=1.2):
    corr_ROI = get_corr_ROI(adata, ad_ROI)
    corr_ROI['corr_mean'] = corr_ROI.mean(axis=1)
    sele_cell = corr_ROI[corr_ROI['corr_mean']>corr_ROI['corr_mean'].max()/thresh]
    df_sele_mask = pd.DataFrame(adata.obs.index.isin(sele_cell.index),index=adata.obs_names,columns=["cell_select"])
    return df_sele_mask


def cell_selection_main(adata, ad_ROI, obs_features_list=None):
    if obs_features_list is None:
        obs_features_list = ['mean_gray', 'std_gray', 'entropy_img', 'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b']
    adata = get_feature(adata, obs_features_list)
    df_sele_mask = select_cell(adata, ad_ROI)
    return df_sele_mask

