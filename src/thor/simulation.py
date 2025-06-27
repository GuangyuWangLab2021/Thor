import logging
import random
from statistics import mode

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

# local package
from thor.utils import (estimate_spot_from_cells, get_adata_layer_array,
                        get_spot_heterogeneity_entropy)

logger = logging.getLogger(__name__)


def call_cells_TopUMI(ad, quantile=0.5, clusters=None):
    """ Select cells with top UMI counts in each cluster
    Parameters
    ----------
    ad: :class:`anndata.AnnData`
        AnnData object with adata.obs["clusters"] and adata.obs["UMI"]
    quantile: :py:class:`float`
        quantile of UMI counts to select cells
    clusters: :py:class:`list`
        list of cluster labels to select cells
    """

    select = np.array([False] * ad.shape[0], dtype=bool)

    for c in clusters:
        select_c = ad.obs["clusters"].isin([c])
        topUMIcells = ad.obs["UMI"] > ad[select_c].obs["UMI"].quantile(quantile)
        select = select | (select_c & topUMIcells)

    ad_select = ad[select]

    return ad_select.copy()


def adata_drop_cells(adata, cell_array):
    all_cells = adata.shape[0]
    keep_cells = [i for i in range(all_cells) if i not in cell_array]

    return adata[keep_cells].copy()


def assign_spot_label(spot_cells_dict, cell_label_dict):

    d = {}
    for spot, cells_array in spot_cells_dict.items():
        if cells_array is None:
            d[spot] = None
        else:
            d[spot] = mode([cell_label_dict[c] for c in cells_array])
    return d


def find_sep(x):
    """ Find the nearest distance between cells or spots
    Parameters
    ----------
    x: :class:`numpy.ndarray` (n_cells, n_dims)
        Spatial coordinates of cells or spots

    Returns
    -------
    :py:class:`float`
        Minimum distance between neighboring cells or spots
    """

    adj = kneighbors_graph(
        x, n_neighbors=1, mode="distance", include_self=False
    )

    beads_sep = np.min(adj.data)
    return beads_sep


def find_sep_most_probable(x):
    """ Find the most probable distance between neighboring cells or spots
    Parameters
    ----------
    x: :class:`numpy.ndarray` (n_cells, n_dims)
        Spatial coordinates of cells or spots

    Returns
    -------
    :py:class:`float`
        Most probable distance between neighboring cells or spots
    """
    adj = kneighbors_graph(
        x, n_neighbors=1, mode="distance", include_self=False
    )
    bins = int(np.max(adj.data) - np.min(adj.data))  # resolution ~ 1

    cnt, dist = np.histogram(adj.data, bins=bins)
    beads_sep = dist[np.argmax(cnt)]

    return beads_sep


def find_bound(x):
    """ function to find the boundary of the cells

    Parameters
    ----------
    x: :class:`numpy.ndarray` (n_cells, n_dims)
        Spatial coordinates of a cell (pixels)

    Returns
    -------
    :py:class:`tuple` of :py:class:`float`
        Boundary of the cells (xmin, xmax, ymin, ymax)
    """

    xmin, ymin = np.min(x, axis=0)
    xmax, ymax = np.max(x, axis=0)

    return (xmin, xmax, ymin, ymax)


def simulate_spot(
    adata_path,
    spot_sep=20,
    layer_key="X_counts",
    cell_label_keys=["clusters"],
    unit_length=1
):
    """
    The adata read from `adata_path` should have spatial coordinates.
    The unit of the spatial coordinates is mu micrometers.
    Spots are created according to the input spatial resolution (`spot_sep`)
    """

    ad = sc.read_h5ad(adata_path)

    n_cells = ad.shape[0]
    n_genes = ad.shape[1]

    cnts = get_adata_layer_array(ad, layer_key=layer_key)

    xy = ad.obsm["spatial"]
    logger.info(
        "Creating spots. Spot positions are estimated according to the single cell positions in obsm[\"spatial\"]."
    )
    xmin, xmax, ymin, ymax = find_bound(xy)
    cell_sep = find_sep(xy)

    # leave 0.5*cell_sep from each boundary
    xmin = xmin - 0.5 * cell_sep
    xmax = xmax + 0.5 * cell_sep
    ymin = ymin - 0.5 * cell_sep
    ymax = ymax + 0.5 * cell_sep
    xr = (xmax - xmin) + cell_sep
    yr = (ymax - ymin) + cell_sep

    # we'd better to have more spots than less (might losing coverage of boundary cells)
    ny = int(np.ceil(yr / spot_sep))
    nx = int(np.ceil(xr / spot_sep))
    n_spots = nx * ny

    spot_cnts_mean = np.zeros(shape=(n_spots, n_genes), dtype=np.float32)
    spot_cnts_sum = np.zeros(shape=(n_spots, n_genes), dtype=np.float32)

    spot_names = []
    spot_locs = []

    for i in range(nx):
        for j in range(ny):
            x_ij = xmin + spot_sep * (i + 0.5)
            y_ij = ymin + spot_sep * (j + 0.5)
            spot_names.append(f"SPOT_{i}_{j}")
            spot_locs.append((x_ij, y_ij))
    spot_locs = np.array(spot_locs)

    def grid_to_spot(x):
        spot_name = f"SPOT_{x[0]}_{x[1]}"
        return (spot_names.index(spot_name))

    grid_index = (xy - np.array([xmin, ymin])) / np.array([spot_sep, spot_sep])
    grid_index = grid_index.astype(int)

    n_cells_in_spot = [[] for i in range(n_spots)]

    cell_cnt = 0
    for grid in grid_index:
        spot_index = grid_to_spot(grid)
        n_cells_in_spot[spot_index].append(cell_cnt)
        cell_cnt += 1

    all_cells_in_spot = {}
    for i in range(n_spots):
        if len(n_cells_in_spot[i]) > 0:
            spot_cnts_mean[i, :] += np.mean(cnts[n_cells_in_spot[i]], axis=0)
            spot_cnts_sum[i, :] += np.sum(cnts[n_cells_in_spot[i]], axis=0)
            all_cells_in_spot[spot_names[i]
                             ] = ad[n_cells_in_spot[i]].obs.index.to_numpy()
        else:
            all_cells_in_spot[spot_names[i]] = None

    spot_cnts_mean = csr_matrix(spot_cnts_mean)
    spot_cnts_sum = csr_matrix(spot_cnts_sum)

    ad_spot = sc.AnnData(spot_cnts_mean)
    ad_spot.layers["X_sum"] = spot_cnts_sum

    ad_spot.obs.index = spot_names

    for k in cell_label_keys:
        cell_label_dict = ad.obs[k].to_dict()
        ad_spot.obs[f"major {k}"] = pd.Series(
            assign_spot_label(all_cells_in_spot, cell_label_dict)
        )

    ad_spot.var.index = ad.var.index
    ad_spot.obsm["spatial"] = spot_locs
    ad_spot.uns["all_cells_in_spot"] = all_cells_in_spot
    try:
        ad_spot.uns["template"] = ad.uns["template"]
        ad_spot.uns["truth"] = adata_path
    except:
        pass

    nonEmptySpots = [
        True if v is not None else False
        for k, v in ad_spot.uns["all_cells_in_spot"].items()
    ]
    ad_spot = ad_spot[nonEmptySpots]

    return ad_spot


def simulate_poisson(
    template_adata_path,
    gene_population=[600, 300, 100],
    lambda_range_low=[1, 3],
    lambda_range_high=[10, 30],
    random_seed=None,
):

    np.random.seed(random_seed)

    ad = sc.read_h5ad(template_adata_path)

    n_genes = np.sum(gene_population)
    gene_names = [f"g_{g+1}" for g in range(n_genes)]

    cell_cluster = ad.obs["clusters"]
    cell_pos = ad.obsm["spatial"]
    n_cells = ad.shape[0]

    clusters = cell_cluster.cat.categories.to_list()
    n_clusters = len(clusters)
    n_cells_in_cluster = cell_cluster.value_counts().to_dict()

    # number of genes which are highly expressed in the cluster
    population = dict(zip(clusters, gene_population))

    gene_cluster_list = []
    for i in range(n_clusters):
        gene_cluster_list += [clusters[i]] * population[clusters[i]]

    gene_cluster = pd.Series(data=gene_cluster_list, index=gene_names)
    lambda_genes = pd.DataFrame(columns=clusters, index=gene_names)

    for c in clusters:
        lambda_genes.loc[gene_cluster == c, c] = np.random.randint(
            lambda_range_high[0],
            high=lambda_range_high[1],
            size=(population[c])
        )
        lambda_genes.loc[~(gene_cluster == c), c] = np.random.randint(
            lambda_range_low[0],
            high=lambda_range_low[1],
            size=(n_genes - population[c])
        )

    X = np.zeros((n_cells, n_genes)).astype(np.float32)
    ad_sim = sc.AnnData(X)
    ad_sim.obs.index = cell_cluster.index
    ad_sim.obs["clusters"] = pd.Categorical(cell_cluster)
    ad_sim.var.index = gene_names
    ad_sim.var["assigned_cluster"] = pd.Categorical(gene_cluster)
    for c in clusters:
        ad_sim.var[f"lambda_{c}"] = lambda_genes[c].tolist()

    ad_sim.obsm["spatial"] = cell_pos
    ad_sim.uns["template"] = template_adata_path

    # sample expressions
    for g in gene_names:
        for c in clusters:
            cells_c = cell_cluster[cell_cluster == c].index.to_list()
            ex_c = np.random.poisson(
                lam=lambda_genes.loc[g, c], size=n_cells_in_cluster[c]
            ).tolist()
            ad_sim[cells_c, g].X = ex_c

    return ad_sim


def generate_cellwise_adata_simulation(truth_adata_path, spot_adata_path):
    ad_sim = sc.read_h5ad(truth_adata_path)
    ad_spot = sc.read_h5ad(spot_adata_path)

    # spots - > cells according to known assignment
    identified_cells = ad_sim.obs.index.to_list()

    spot_assigned = {}
    for spot_id in ad_spot.obs.index:
        m_list = ad_spot.uns["all_cells_in_spot"][spot_id]
        for m in m_list:
            spot_assigned[m] = spot_id

    spots = [spot_assigned[cell] for cell in identified_cells]
    ad_sc = ad_spot[spots]
    ad_sc.uns["spots_mapped_to_cell"] = spot_assigned

    ad_sc.obsm["spatial_spot"] = ad_sc.obsm["spatial"]
    ad_sc.obs["spot_barcodes"] = spots

    ad_sc.obsm["spatial"] = ad_sim[identified_cells].obsm["spatial"]
    ad_sc.obs.index = identified_cells
    ad_sc.var = ad_sim[identified_cells].var
    ad_sc.obs["clusters"] = ad_sim[identified_cells].obs["clusters"]
    ad_sc.obs = pd.concat(
        (
            ad_sc.obs,
            pd.DataFrame(
                ad_sc.obsm["spatial"],
                columns=["x", "y"],
                index=ad_sc.obs.index
            ), pd.get_dummies(ad_sc.obs["clusters"])
        ),
        axis=1
    )
    ad_sc.uns["cell_image_props"] = np.array(
        ["x", "y"] + list(ad_sc.obs["clusters"].cat.categories), dtype=object
    )
    hetero = get_spot_heterogeneity_entropy(ad_sc.obs)
    ad_sc.obs["spot_heterogeneity"] = hetero

    # .copy() is necessary!
    ad_sc.layers["X_counts"] = ad_sc.X.copy()

    sc.pp.normalize_total(ad_sc, inplace=True)
    sc.pp.log1p(ad_sc)
    return ad_sc


def generate_cellwise_adata(truth_adata_path, spot_adata_path):
    ad_sim = sc.read_h5ad(truth_adata_path)
    ad_spot = sc.read_h5ad(spot_adata_path)

    # spots - > cells according to known assignment
    identified_cells = ad_sim.obs.index.to_list()

    spot_assigned = {}
    for spot_id in ad_spot.obs.index:
        m_list = ad_spot.uns["all_cells_in_spot"][spot_id]
        for m in m_list:
            spot_assigned[m] = spot_id

    spots = [spot_assigned[cell] for cell in identified_cells]
    ad_sc = ad_spot[spots]
    ad_sc.uns["spots_mapped_to_cell"] = spot_assigned

    ad_sc.obsm["spatial_spot"] = ad_sc.obsm["spatial"]
    ad_sc.obs["spot_barcodes"] = spots

    ad_sc.obsm["spatial"] = ad_sim[identified_cells].obsm["spatial"]
    ad_sc.obs.index = identified_cells
    ad_sc.var = ad_sim[identified_cells].var
    ad_sc.obs["clusters"] = ad_sim[identified_cells].obs["clusters"]
    ad_sc.obs = pd.concat(
        (
            ad_sc.obs,
            pd.DataFrame(
                ad_sc.obsm["spatial"],
                columns=["x", "y"],
                index=ad_sc.obs.index
            ), pd.get_dummies(ad_sc.obs["clusters"])
        ),
        axis=1
    )
    ad_sc.uns["cell_image_props"] = np.array(
        ["x", "y"] + list(ad_sc.obs["clusters"].cat.categories), dtype=object
    )
    hetero = get_spot_heterogeneity_entropy(ad_sc.obs)
    ad_sc.obs["spot_heterogeneity"] = hetero

    # .copy() is necessary!
    ad_sc.layers["X_counts"] = ad_sc.X.copy()

    sc.pp.normalize_total(ad_sc, inplace=True)
    sc.pp.log1p(ad_sc)
    return ad_sc


def drop_cells(adata, cluster=None, proportion_drop=None, random_seed=2023):

    if cluster is not None:
        ad = adata[adata.obs["clusters"] == cluster]
    else:
        ad = adata

    n_cells = ad.shape[0]
    n_keep = n_cells - int(n_cells * proportion_drop)

    random.seed(random_seed)
    cells_keep = random.sample(adata.obs_names.tolist(), k=n_keep)

    ad_w_missouts = ad[cells_keep].copy()
    get_spot_heterogeneity_entropy(ad_w_missouts)
    return ad_w_missouts


def adj_to_neighborslist(adj, cells=None):

    if cells is None:
        cells = range(adj.shape[0])
    return [np.nonzero(adj[i])[1] for i in cells]


def add_neighborslist_to_knn(adj, neighbors_list, cells=None):
    adj = adj.toarray()

    for i in range(len(cells)):
        neigh = neighbors_list[i]
        cell = cells[i]
        adj[cell] = 0
        adj[cell, neigh] = 1

    return adj


def get_CV(a, b):
    return np.linalg.norm(a - b,
                          axis=0) / (np.sqrt(a.shape[0]) * np.mean(a, axis=0))


def get_cell_mean_neighbors(neighbors_list):
    n_neighs = [len(neigh) for neigh in neighbors_list]
    return np.mean(n_neighs)


def get_average_network_connection_quality(adj, cluster_labels):
    """
    Parameters
    ---------
    adj is a connectivity matrix (csr_matrix)
    cluster_labels is list-like labels of clusters for the cells indexed from 0 to n_cells.

    Returns
    -------
    mean proportion of edges across clusters.
    proportion of cells which have at least 1 inter-cluster connection.
    """

    n_cells = len(cluster_labels)
    neighbors_list = adj_to_neighborslist(adj)

    percentage_contaminated_edges = []

    for c in range(n_cells):
        neigh_c = neighbors_list[c]
        clusters_c = np.array([cluster_labels[i] for i in neigh_c])
        bad_neighbors = (clusters_c != cluster_labels[c])
        percentage_contaminated_edges.append(np.mean(bad_neighbors))

    percentage_contaminated_edges = np.array(percentage_contaminated_edges)
    return np.mean(percentage_contaminated_edges), np.mean(
        percentage_contaminated_edges > 0
    )


def perturb_KNN_neighbors(
    adj,
    adj_large,
    proportion_cells_perturb=0.1,
    poisson_mean_drop=1,
    neighbor_num=None,
    random_seed=2023
):
    """
    adj: adjacency matrix
    adj_search: adjacency matrix with a much large number of neighbors (for reconnecting)
    """
    np.random.seed(random_seed)

    if neighbor_num is None:
        neighbor_num = mode(adj.toarray().sum(axis=1))

    ori_net = adj.copy()

    n_nodes = ori_net.shape[0]
    n_cells_perturb = int(n_nodes * proportion_cells_perturb)
    logger.info(
        f"Add edge perturbations to {n_cells_perturb} out of {n_nodes} cells."
    )

    cells_perturb = np.random.choice(
        range(n_nodes), size=n_cells_perturb, replace=False
    )  # vector index

    neighbors_list = adj_to_neighborslist(adj, cells=cells_perturb)
    #print(f"Before perturbation, there are on average {get_cell_mean_neighbors(neighbors_list)} edges for each cell")

    neighbors_list_large = adj_to_neighborslist(adj_large, cells=cells_perturb)
    neighbors_list_search = [
        np.array(list(set(neighbors_list_large[i]) - set(neighbors_list[i])))
        for i in range(n_cells_perturb)
    ]

    # number of edges to drop for the selected cells
    # cap it to the total number of neighbors
    n_edges_drop = np.random.poisson(
        lam=poisson_mean_drop, size=n_cells_perturb
    )
    n_edges_drop[n_edges_drop > neighbor_num] = neighbor_num

    for i in range(n_cells_perturb):
        neighs_keep = np.random.choice(
            neighbors_list[i],
            size=neighbor_num - n_edges_drop[i],
            replace=False
        )
        neighs_add = np.random.choice(
            neighbors_list_search[i], size=n_edges_drop[i], replace=False
        )
        neighbors_list[i] = np.append(neighs_keep, neighs_add)

    #print(f"After perturbation, there are on average {get_cell_mean_neighbors(neighbors_list)} edges for each cell")

    ori_net = add_neighborslist_to_knn(
        ori_net, neighbors_list=neighbors_list, cells=cells_perturb
    )
    ori_net = csr_matrix(ori_net)
    ori_net.eliminate_zeros()
    return ori_net


def mislabel_cells(
    adata, cluster=None, proportion_mislabel=None, random_seed=2023
):

    np.random.seed(random_seed)

    clusters = adata.obs["clusters"].cat.categories.tolist()

    if cluster is not None:
        ad = (adata[adata.obs["clusters"] == cluster]).copy()
    else:
        ad = adata.copy()

    n_cells = ad.shape[0]
    n_mislabel = int(n_cells * proportion_mislabel)

    cells_mislabel = np.random.choice(range(n_cells), size=n_mislabel)

    c_new = []
    for c in adata.obs["clusters"][cells_mislabel]:
        other_cluster = [cl for cl in clusters if cl != c]
        c_new.append(np.random.choice(other_cluster))

    ad.obs["clusters"][cells_mislabel] = c_new

    # change relevant entries
    ad.obs = ad.obs.assign(
        **(pd.get_dummies(ad.obs["clusters"]).to_dict(orient="series"))
    )
    get_spot_heterogeneity_entropy(ad)
    return ad


def get_gene_dropout_ratio(adata, layer=None):
    """
    Returns a list-like dropout ratios for each gene (n_genes)
    """

    X = get_adata_layer_array(adata, layer_key=layer)
    return np.mean(1 - np.array(X != 0), axis=0)


def get_cell_lambda(ad_sim):

    clusters = ad_sim.obs["clusters"].cat.categories
    cells = ad_sim.obs_names
    lambda_clusters = [f"lambda_{c}" for c in clusters]

    df = pd.get_dummies(ad_sim.obs["clusters"])
    cellxcluster = df.loc[cells, clusters].values
    genexcluster = ad_sim.var.loc[:, lambda_clusters].values
    cellxgene = cellxcluster * csr_matrix(genexcluster.T)
    return cellxgene


def estimate_spot_lambda(ad_spot, mapping_method="mean"):
    """
    return spot x gene lambda values (numpy array).
    """

    try:
        ad_sim = sc.read_h5ad(ad_spot.uns["truth"])
    except:
        logger.error("truth path not correctly set in the spot adata")
        return None

    cells = ad_sim.obs_names
    spots = ad_spot.obs_names

    spot_assigned = {}
    for spot_id in ad_spot.obs.index:
        m_list = ad_spot.uns["all_cells_in_spot"][spot_id]
        for m in m_list:
            spot_assigned[m] = spot_id

    # cell x spot
    df = pd.get_dummies(pd.Series(spot_assigned))
    cellxspot = df.loc[cells, spots].values

    cellxgene = get_cell_lambda(ad_sim)
    spotxgene = estimate_spot_from_cells(
        cellxgene, cellxspot, mapping_method=mapping_method
    )
    return spotxgene


def logistic_func(x, midpoint, lshape):
    """
    Logistic function for simulating gene dropouts.

    Parameters
    ----------
    x: numpy.ndarray (n_cells, n_genes)
        Input matrix
    midpoint: list-like (n_cells)
        Midpoint of the logistic function for each cell (spot)
    lshape: list-like (n_cells)
        Shape of the logistic function for each cell (spot)

    Returns
    -------
    probs: numpy.ndarray (n_cells, n_genes)
        Probability matrix
    """

    probs = 1 / (1 + np.exp(-lshape * (x.T - midpoint)))
    return probs.T


def simulate_gene_dropouts(
    ad_spot,
    dropout_midpoints,
    dropout_shapes,
    counts_layer=None,
    mapping_method="mean"
):
    """ Simulate gene dropouts for each spot. Adapted from Splatter [https://github.com/Oshlack/splatter]; specifically splatSimDropout function.

    Parameters
    ----------
    ad_spot: AnnData
        Spot AnnData object
    dropout_midpoints: list-like
        Midpoint of the logistic function for each spot
    dropout_shapes: list-like
        Shape of the logistic function for each spot
    counts_layer: str
        Layers key for the counts matrix
    mapping_method: str
        Method to map cell expression to spot expression. Available options are "mean" and "sum". Default is "mean".    

    Returns
    -------
    counts: np.ndarray
        Simulated counts matrix
    drop_prob: np.ndarray
        Dropout probability matrix
    1 - keep: np.ndarray
        Dropout matrix
    """

    true_counts = get_adata_layer_array(ad_spot, layer_key=counts_layer)

    n_spots, n_genes = ad_spot.shape
    spot_lambda = estimate_spot_lambda(ad_spot, mapping_method=mapping_method)

    #assert len(dropout_midpoints) == n_spots
    #assert len(dropout_shapes) == n_spots

    drop_prob = logistic_func(
        np.log(spot_lambda), dropout_midpoints, dropout_shapes
    )
    keep = np.random.binomial(1, 1 - drop_prob, ad_spot.shape)
    counts = true_counts * keep

    return counts, drop_prob, 1 - keep
