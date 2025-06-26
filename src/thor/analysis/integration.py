import os
import matplotlib.pyplot as plt
import scanpy as sc


def sc_prediction_integration(labelled_ads_dict, outdir=""):
    ad_combined = sc.concat(labelled_ads_dict, label='batch')
    ad_combined.obs_names_make_unique()
    sc.pp.normalize_total(ad_combined)
    sc.tl.pca(ad_combined, n_comps=50)
    sc.external.pp.harmony_integrate(ad_combined, key='batch')
    try:
        ad_combined.obsm['X_umap_prior'] = ad_combined.obsm['X_umap'].copy()
    except:
        pass
    ad_combined.obsm['X_pca_prior'] = ad_combined.obsm['X_pca'].copy()
    ad_combined.obsm['X_pca'] = ad_combined.obsm['X_pca_harmony']
    sc.pp.neighbors(ad_combined, n_neighbors=10)
    sc.tl.umap(ad_combined)

    sample_palette = {'Ground truth': '#334356', 'fineST': '#e3c8aa'}
    dotsize = 10
    fig, axes = plt.subplots(figsize=(12, 3), ncols=3)
    sc.pl.embedding(
        ad_combined,
        basis='X_umap',
        color='batch',
        show=False,
        frameon=False,
        title='Overlayed',
        size=dotsize,
        ax=axes[2],
        palette=sample_palette,
        alpha=0.5
    )
    sc.pl.embedding(
        ad_combined[ad_combined.obs.batch == 'Ground truth'],
        basis='X_umap',
        color='region',
        show=False,
        frameon=False,
        title='Ground truth',
        size=dotsize,
        ax=axes[0],
        legend_loc='none'
    )
    sc.pl.embedding(
        ad_combined[ad_combined.obs.batch == 'fineST'],
        basis='X_umap',
        color='region',
        show=False,
        frameon=False,
        title='fineST',
        size=dotsize,
        ax=axes[1],
        legend_loc='none'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'umap_integration_separate.pdf'))
    return ad_combined


# Example:
#ad_pred = create_umap(ad_sc_out_path, layer='snn_genemode_20_corr', apply_normalization=False, plot=False, return_adata=True)
#labelled_ads_dict = {"Ground truth":ad_true_roi, "fineST": ad_pred}
#ad = sc_prediction_integration(labelled_ads_dict)

def create_umap(
    adata_path,
    layer=None,
    apply_normalization=False,
    name=None,
    color_key=None,
    palette=None,
    plot=True,
    return_adata=False
):
    ad = sc.read_h5ad(adata_path)
    if layer is not None:
        ad.X = ad.layers[layer]
        ad.X[ad.X < 0] = 0

    if apply_normalization:
        sc.pp.normalize_total(ad)
        sc.pp.log1p(ad)

    sc.tl.pca(ad)
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)

    if plot:
        ax = sc.pl.umap(
            ad,
            color=color_key,
            show=False,
            frameon=False,
            title=name,
            palette=palette
        )
        plt.savefig(f"{name}.pdf")
    if return_adata:
        return ad
