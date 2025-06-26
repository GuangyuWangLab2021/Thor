import operator
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import issparse


def create_subset_ad(adata, adata_bg, n_hists=10, relate='>='):
    """

    Parameters
    ----------
    adata :
        
    adata_bg :
        
    n_hists :
         (Default value = 10)
    relate :
         (Default value = '>=')

    Returns
    -------

    """

    ops = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq
    }

    clusters = adata.obs['clusters'].cat.categories.to_list()
    n_clusters = len(clusters)
    ad_subsets = {}
    ad_subsets_bg = {}
    for hetero_threshold in np.linspace(0, np.log(n_clusters),
                                        n_hists + 1)[:-1]:
        hetero_cells = ops[relate](
            adata.obs['spot_heterogeneity'], hetero_threshold
        )
        ad_subsets[hetero_threshold] = adata[hetero_cells].copy()
        ad_subsets_bg[hetero_threshold] = adata_bg[hetero_cells].copy()
    return ad_subsets, ad_subsets_bg


def boxplot_cluster(
    ad,
    layer_key=None,
    genes_list=None,
    cell_list=None,
    ax=None,
    return_data=True,
    y='expression',
    x='marker',
    hue='cluster',
    calculate_only=False,
    sample_label=None
):
    """

    Parameters
    ----------
    ad :
        
    layer_key :
         (Default value = None)
    genes_list :
         (Default value = None)
    cell_list :
         (Default value = None)
    ax :
         (Default value = None)
    return_data :
         (Default value = True)
    y :
         (Default value = 'expression')
    x :
         (Default value = 'marker')
    hue :
         (Default value = 'cluster')
    calculate_only :
         (Default value = False)
    sample_label :
         (Default value = None)

    Returns
    -------

    """

    ad1 = ad.copy()

    if genes_list is not None:
        ad1 = ad1[:, genes_list].copy()
    if cell_list is not None:
        ad1 = ad1[cell_list, :].copy()

    if layer_key is not None:
        ad1.X = ad1.layers[layer_key]

    if issparse(ad1.X):
        ad1.X = ad1.X.toarray()

    group = OrderedDict()
    group_labels = []
    cluster_labels = []
    marker_labels = []

    for c_g in ad1.obs.clusters.cat.categories:
        c_g_short = c_g[0]
        for c_c in ad1.obs.clusters.cat.categories:
            c_c_short = c_c[0]
            k = f"m{c_g_short}@{c_c_short}"
            v = ad1[ad1.obs.clusters.isin([c_c]),
                    ad1.var.assigned_cluster.isin([c_g])].X.ravel()
            group[k] = v
            group_labels += [k] * len(v)
            cluster_labels += [c_c_short] * len(v)
            marker_labels += [c_g_short] * len(v)

    df = pd.DataFrame(
        np.concatenate(list(group.values())), columns=['expression']
    )
    df['group'] = pd.Categorical(group_labels)
    df['cluster'] = pd.Categorical(cluster_labels)
    df['marker'] = pd.Categorical(marker_labels)
    df['metagene'] = (
        df['cluster'] == df['marker']
    ).apply(lambda x: f'{sample_label}_high' if x else f'{sample_label}_low')

    if calculate_only:
        return df

    ax = sns.boxplot(
        df,
        y=y,
        x=x,
        hue=hue,
        whis=[5, 95],
        ax=ax,
        dodge=True,
        width=0.5,
        fliersize=0.05,
        order=None
    )
    #fg=sns.catplot(x="marker", hue="cluster", y="expression", data=_df, kind="box", whis=[5,95], fliersize=0.05, dodge=True, width=0.5, ax=ax)
    #ax.legend(loc='best')
    #ax.legend(loc='upper center', bbox_to_anchor=(0.3, 0.98), fancybox=True, shadow=True, ncol=1)
    if return_data:
        return ax, df


def boxplot_group(
    ad,
    ad_bg,
    hetero=0.0,
    layer_key='y_50',
    ncols=3,
    figsize=(20, 4),
    label=None,
    **kwargs
):
    """

    Parameters
    ----------
    ad :
        
    ad_bg :
        
    hetero :
         (Default value = 0.0)
    layer_key :
         (Default value = 'y_50')
    ncols :
         (Default value = 3)
    figsize :
         (Default value = (20)
    4) :
        
    label :
         (Default value = None)
    **kwargs :
        

    Returns
    -------

    """


    fig, axes = plt.subplots(ncols=ncols, figsize=figsize, sharey=True)

    _, _ = boxplot_cluster(ad[hetero], ax=axes[0], **kwargs)
    _, _ = boxplot_cluster(
        ad[hetero], layer_key=layer_key, ax=axes[2], **kwargs
    )
    _, _ = boxplot_cluster(ad_bg[hetero], ax=axes[1], **kwargs)

    axes[0].set_title('Spot')
    axes[1].set_title('Truth')
    axes[2].set_title('Recovered')

    axes[0].yaxis.grid(True)
    axes[1].yaxis.grid(True)
    axes[2].yaxis.grid(True)

    plt.suptitle(f'heterogeneity {label} {round(hetero,2)}')

    plt.show()


def boxplot_all(ad_sub, layer_key=None, relate=">=", title="", ylim=None):
    """

    Parameters
    ----------
    ad_sub :
        
    layer_key :
         (Default value = None)
    relate :
         (Default value = ">=")
    title :
         (Default value = "")
    ylim :
         (Default value = None)

    Returns
    -------

    """
    hetero_thresholds = sorted(list(ad_sub.keys()))
    nbins = len(hetero_thresholds)

    fig, axes = plt.subplots(
        ncols=nbins, figsize=(30, 4), sharey=True, sharex=True
    )

    for i in range(nbins):
        hetero_threshold = hetero_thresholds[i]
        ad = ad_sub[hetero_threshold]
        _, _ = boxplot_cluster(ad, ax=axes[i], layer_key=layer_key)
        axes[i].set_title(f"S {relate} {round(hetero_threshold,2)}")
        axes[i].legend().set_visible(False)
        axes[i].yaxis.grid(True)

    plt.suptitle(f'{title}')
    plt.ylim(ylim)
    plt.show()


def check_PCA(ad_true, ad_sc_out, layers=['y_50', 'y_100']):
    """

    Parameters
    ----------
    ad_true :
        
    ad_sc_out :
        
    layers :
         (Default value = ['y_50')
    'y_100'] :
        

    Returns
    -------

    """

    def merge_list(lst):
        """

        Parameters
        ----------
        lst :
            

        Returns
        -------

        """
        ml = []
        for l in lst:
            ml += l
        return ml

    ad_combined = sc.AnnData(
        np.concatenate(
            [ad_true.X, ad_sc_out.X.toarray()] +
            [ad_sc_out.layers[i] for i in layers]
        )
    )
    ad_combined.obs['source'] = ['ground_truth'] * ad_true.shape[0] + [
        'spot'
    ] * ad_sc_out.shape[0] + merge_list(
        [[i] * ad_sc_out.shape[0] for i in layers]
    )
    ad_combined.obs['clusters'] = ad_true.obs['clusters'].to_list(
    ) * (2 + len(layers))
    sc.tl.pca(ad_combined)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
    sc.pl.pca(
        ad_combined[ad_combined.obs['source'].isin(['ground_truth', 'spot'])],
        color='source',
        size=5,
        show=False,
        ax=axes[0],
        legend_loc=None
    )
    sc.pl.pca(
        ad_combined[ad_combined.obs['source'].isin(['ground_truth', 'spot'])],
        color='clusters',
        size=5,
        show=False,
        ax=axes[1],
        legend_loc='on data'
    )
    plt.show()

    for layer in layers:
        fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
        sc.pl.pca(
            ad_combined[ad_combined.obs['source'].isin(['ground_truth',
                                                        layer])],
            color='source',
            size=5,
            show=False,
            ax=axes[0],
            legend_loc=None
        )
        sc.pl.pca(
            ad_combined[ad_combined.obs['source'].isin(['ground_truth',
                                                        layer])],
            color='clusters',
            size=5,
            show=False,
            ax=axes[1],
            legend_loc='on data'
        )
        plt.show()


def grouped_obs_mean(
    adata, group_key, layer=None, sample_label=None
):
    """mean values according to cell labels.
    """

    if layer is not None:
        getX = lambda x: x.layers[layer]
    else:
        getX = lambda x: x.X

    grouped = adata.obs.groupby(group_key)
    df = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=adata.var_names
    )

    for group, idx in grouped.indices.items():
        X = getX(adata[idx])
        if issparse(X):
            X = X.toarray()
        df[group] = np.ravel(X.mean(axis=0, dtype=np.float64)).tolist()

    df = pd.melt(
        df.reset_index(),
        id_vars=['index'],
        var_name=group_key,
        value_name='mean expression',
        ignore_index=True
    )
    df['marker'] = df['index'].map(
        lambda x: adata.var.loc[x, 'assigned_cluster']
    )
    df['metagene'] = (
        df[group_key] == df['marker']
    ).apply(lambda x: f'{sample_label}_high' if x else f'{sample_label}_low')
    return df


def get_relative_distance(a, b):
    """

    Parameters
    ----------
    a :
        
    b :
        

    Returns
    -------

    """
    return np.linalg.norm(a - b) / np.linalg.norm(a)
