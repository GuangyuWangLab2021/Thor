import warnings
from statistics import mode

import numexpr as ne
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, StandardScaler


warnings.filterwarnings("ignore")

import logging

# local package
from thor.utils import mask_large_sparse_matrix, row_normalize_sparse, kneighbors_graph_with_geometrical_constraint, kneighbors_conndist_graph


logger = logging.getLogger(__name__)


def promote_good_flow(G, K=None, alpha=5):
    """ Compute the flow from good cells to bad cells. The flow is asymmetric, i.e., the flow from good cells to bad cells is larger than the other direction.
        Good cells are defined as cells with high node weights (low heterogeneity scores).
        The flow is computed as a sigmoid function of the difference of node weights between two cells.

    Parameters
    ----------
    G: numpy vector (n_cells)
        Node weights of each cell.
    K: csr sparse matrix (n_cells x n_cells)
        The connectivity matrix. K[i, j] is the weight of the edge from cell i to cell j.
    alpha: float
        The steepness of the sigmoid function. The larger the alpha, the more asymmetric the flow is.

    Returns
    -------
    flow: csr sparse matrix (n_cells x n_cells)
        The flow matrix. flow[i, j] is the flow from cell i to cell j.
    """

    MANY_CELLS = 20000
    if len(G) > MANY_CELLS:
        assert K is not None
        return promote_good_flow_sparse(G, K, alpha=alpha)
    else:
        return promote_good_flow_dense(G, alpha=alpha)


def promote_good_flow_dense(G, alpha=5):

    Garr = G[:, np.newaxis]
    flavored_displacement = ne.evaluate("G * (G - Garr)")
    flow = ne.evaluate("1/(1+exp(-flavored_displacement * alpha))")
    return flow


def promote_good_flow_sparse(G, K, alpha=5):
    flow = K.copy()
    i, j = K.nonzero()

    flow[i, j] = G[j] - G[i]

    flavored_displacement = flow.multiply(G)
    flow[i, j] = 1 / (1 + np.exp(-flavored_displacement.tocsr()[i, j] * alpha))
    return flow


def compute_node_weights(S, kinv=1):
    """ Compute node weights from the heterogeneity score S.

    Parameters
    ----------
    S: numpy array (n_cells x n_features)
        Heterogeneity score of each cell. Each row is a cell, each column is a feature (only supports 1 feature at the moment). The larger the score, the more heterogeneous the cell is.
    kinv: float
        The inverse of the kernel bandwidth. The larger the kinv, the more sensitive to the heterogeneity score.

    Returns
    -------
    G:  numpy vector
        Node weights of each cell.
    """

    S_norm = MinMaxScaler().fit_transform(S)
    G = np.exp(-1 * S_norm * kinv)

    G = G / G.max()

    return G


def quantile_normalization(arr, qr=(0.15, 0.85), alpha=5):
    """
    Parameters
    ---------
    arr: n_obs x n_feats (numpy array)

    Returns
    -------
    numpy array: numbers are in range (0, 1)
    """

    logger.info(
        f"Apply quantile normalization, so data between the quantiles {qr} are scaled to [-1, 1]."
    )
    ql = np.quantile(arr, qr[0], axis=0)
    qu = np.quantile(arr, qr[1], axis=0)

    if ql == qu:
        logger.error("Data is too uniform for the quantile normalization.")
        return arr

    # ql --> -1, qu --> 1
    arr_trans = (arr - ql) / (qu - ql) * 2 - 1
    arr_trans = 1 / (1 + np.exp(-1 * alpha * arr_trans))

    return arr_trans


def compute_transition_matrix(adj_matrix, edge_weights=None, node_weights=None, lmda=None):
    """
    A row is the in- probability. Normalize each row to 1 (or 0 if the node does not have in- edges).

    Parameters
    ---------
    adj_matrix: csr_matrix (n_cells x n_cells)
        The connectivity matrix. Edge weights are not included.
    edge_weights: csr_matrix (n_cells x n_cells) or float
        The edge weights. If float, all edges are weighted equally.
    node_weights: numpy 1d array (n_cells)
        The node weights. Homogeneous spot ~ high quality cells ~ large node weights ~  large inertia ~ low incoming flow.
    lmda: float
        1 - self_weight. The smaller the lmda, the more likely a cell maintains its own expression.

    Returns
    -------
    T: csr_matrix (n_cells x n_cells)
        The transition matrix.
    """

    logger.info("Compute transition matrix")
    # pointwise multiplication (asymmetric directed flow; Favor flow from high- to low- quality cells, and deter the other direction)
    T = adj_matrix.multiply(edge_weights)

    # normalize each row to 1
    T = row_normalize_sparse(T)

    # self-weight: probability (fraction) of maintaining own expression
    self_weight = 1 - lmda
    logger.info(f"self weight scale is set to: {self_weight:.3f}")
    self_weight = self_weight * node_weights
    T = T.multiply((1 - self_weight)[:, np.newaxis])

    # adata does not like coo_matrix, slicing fails.
    T = T.tocsr()
    T.setdiag(self_weight)

    return T


def compute_transition_matrix_update(adj_matrix, edge_weights=None, node_weights=None, lmda=None):
    """
    A row is the in- probability. Normalize each row to 1 (or 0 if the node does not have in- edges).

    Parameters
    ---------
    adj_matrix: csr_matrix (n_cells x n_cells)
        The connectivity matrix. Edge weights are not included.
    edge_weights: csr_matrix (n_cells x n_cells) or float
        The edge weights. If float, all edges are weighted equally.
    node_weights: numpy 1d array (n_cells)
        The node weights. Homogeneous spot ~ high quality cells ~ large node weights ~  large inertia ~ low incoming flow.
    lmda: float
        1 - self_weight. The smaller the lmda, the more likely a cell maintains its own expression.

    Returns
    -------
    T: csr_matrix (n_cells x n_cells)
        The transition matrix.
    """

    logger.info("Compute transition matrix")
    # pointwise multiplication (asymmetric directed flow; Favor flow from high- to low- quality cells, and deter the other direction)
    T = adj_matrix.multiply(edge_weights)

    # normalize each row to 1 (L1)
    T = row_normalize_sparse(T) 
    T = T.multiply((1 - node_weights)[:, np.newaxis])

    # self-weight: probability (fraction) of maintaining own expression
    self_weight = 1 - lmda
    logger.info(f"self weight scale is set to: {self_weight:.3f}")
    T = T.multiply(lmda)

    # adata does not like coo_matrix, slicing fails.
    T = T.tocsr()
    T.setdiag(self_weight)

    return T


def add_transition_matrix_to_adata(
    adata,
    conn_key=None,
    snn_threshold=None,
    preferential_flow=True,
    weigh_cells=True,
    balance_cell_quality=False,
    bcq_IQR=(0.15, 0.85),
    node_features_obs_list=["spot_heterogeneity"],
    diffusion_extent=None,
    inflation_percentage=None
):
    """
    node_features_obs_list: List
    """

    # compute node_weights: vector-like
    if weigh_cells:
        logger.info("Weigh cells according to the spot heterogeneity.")
        nfeats = adata.obs[node_features_obs_list].values
        node_weights = compute_node_weights(nfeats, kinv=1)
        if balance_cell_quality:
            logger.info("Balance cell quality!")
            logger.info(f"Stereotype cells outside of quantiles {bcq_IQR} to bad or good cells")
            node_weights = quantile_normalization(node_weights, qr=bcq_IQR, alpha=5)

        node_weights = node_weights[:, 0]
    else:
        node_weights = np.zeros(adata.shape[0])

    conn = adata.obsp[f"{conn_key}_connectivities"]

    # using a mask to filter out low quality connections
    nones_mask = np.array(conn[conn.nonzero()] < snn_threshold)[0]
    conn = mask_large_sparse_matrix(conn, nones_mask)

    # compute latency matrix
    if preferential_flow:
        logger.info("Promote flow of information from more homogeneous cells to less.")
        latency = promote_good_flow(node_weights, K=conn, alpha=5)
    else:
        latency = 1

    # applying SNN threshold to the edges
    logger.info(f"Eliminate low quality edges (<{snn_threshold}) between cells.")

    T_csr = compute_transition_matrix(
        conn, edge_weights=latency, node_weights=node_weights, lmda=diffusion_extent
    )

    if inflation_percentage is not None:
        # The diffusion transition matrix is defined as:
        # T = I - lmda * K 
        # the inflation transition matrix
        # R = I - mu * K

        mu = -(100 + inflation_percentage) * diffusion_extent * 0.01
        T_inflat = compute_transition_matrix(
            conn, edge_weights=latency, node_weights=node_weights, lmda=mu
        )
        T_csr = T_inflat @ T_csr

    if conn_key not in adata.uns:
        adata.uns[conn_key] = {}

    adata.obs["node_weights"] = node_weights.tolist()
    adata.obsp[f"{conn_key}_transition_matrix"] = T_csr
    logger.info(f"Added transition matrix to adata.obsp[\"{conn_key}_transition_matrix\"]")


def knn_to_snn(knn_graph, neighbor_num=None, keep_neighbors=True):
    """
    knn_graph is csr_matrix
    """
    assert neighbor_num > 0

    # notice knn_graph is csr_matrix and the * is actually matrix multiplication.
    snn_connectivity = knn_graph * knn_graph.T / neighbor_num

    # exclude self-connections.
    snn_connectivity.setdiag(0)

    snn_connectivity = csr_matrix(snn_connectivity, dtype=float)
    if keep_neighbors:
        return snn_connectivity.multiply(knn_graph)
    else:
        return snn_connectivity


def construct_SNN(
    img_props,
    neighbor_num,
    select_cell=None,
    select_feat=None,
    save_conn_path=None,
    geom_morph_ratio=1,
    geom_constraint=0,
    reduce_morph_dim=False,
    reduced_dimension_image_features_pcs=15,
    theta=0,
    transcriptome_array=None
):
    """
    Parameters
    ---------

    theta: contribution of the transcriptome to the KNN distance matrix in relative to the H&E image features. 
            Valid values range from 0 (no contribution) to inf.
    """

    MANY_FEATURES = max(20, reduced_dimension_image_features_pcs)  

    if select_feat is None:
        select_feat = img_props.columns.to_list()

    logger.info(f"Construct SNN with morphological features: {select_feat[2:]}.")

    if select_cell is None:
        X_raw = img_props.loc[:, select_feat].values
    else:
        X_raw = img_props.loc[select_cell, select_feat].values

    X = StandardScaler().fit_transform(X_raw)

    if geom_morph_ratio is not None:
        assert (geom_morph_ratio >= 0), "geom_morph_ratio must be non-negative."
        if select_feat[0] != "x" or select_feat[1] != "y":
            logger.error("The first two columns need to be [\"x\", \"y\"]")
            return None

        pos = X[:, :2]
        morph = X[:, 2:]
        pos = pos * geom_morph_ratio / np.sqrt(pos.shape[1])

        # reduce the dimension of morphological features if there are many
        # using first two components of PCA
        reduce_morph_dim = reduce_morph_dim and (morph.shape[1] > reduced_dimension_image_features_pcs)
        reduce_morph_dim = reduce_morph_dim or (morph.shape[1] > MANY_FEATURES)

        if reduce_morph_dim:

            pca = PCA(n_components=reduced_dimension_image_features_pcs)
            morph = pca.fit_transform(morph)
            morph = StandardScaler().fit_transform(morph)

            logger.info(f"Reduced morphological features to {morph.shape[1]} dimensions.")


        morph = morph / np.sqrt(morph.shape[1])
        X = np.concatenate((pos, morph), axis=1)

    # Note by P.Z.:
    # Instead of doing this, perhaps using a gaussian kernel for the transcriptome part as a factor of the edge weights. [ MAGIC way ]
    if transcriptome_array is not None and theta > 0:
        logger.info(f"Incorporate the effect of transcriptome. The dimension of the transcriptome is {transcriptome_array.shape[1]}.")
        # normalize the transcriptome_array (reduced space) to the standard normal distribution: N(0,1)
        ts = StandardScaler().fit_transform(transcriptome_array[:, :])
        ts = ts * theta / np.sqrt(ts.shape[1]) * 100
        X = np.concatenate((X, ts), axis=1)

    if geom_constraint is None or geom_constraint <= 0:
        knn_graph, distance_graph = kneighbors_conndist_graph(
            X, n_neighbors=neighbor_num, include_self=False
        )
    else:
        logger.info(f"Apply geom_constraint {geom_constraint} pixels")
        knn_graph, distance_graph = kneighbors_graph_with_geometrical_constraint(X, neighbor_num, X_raw[:, :2], geom_constraint)

    snn_connectivity = knn_to_snn(knn_graph, neighbor_num=neighbor_num, keep_neighbors=False)
    snn_distance = snn_connectivity.multiply(distance_graph)
    logger.info("Finish constructing SNN")

    if save_conn_path is not None:
        save_npz(save_conn_path, snn_connectivity)
        save_npz(save_conn_path+"-dist.npz", snn_distance)
        logger.info(f"SNN matrix saved at {save_conn_path}")
    return snn_connectivity, snn_distance, knn_graph, distance_graph


def add_snn_to_adata(
    adata,
    obs_keys=None,
    neighbor_num=None,
    geom_morph_ratio=1.,
    geom_constraint=0,
    theta=0,
    conn_key="snn",
    reduced_dimension_transcriptome_obsm_key="X_pca",
):

    obsm_key = reduced_dimension_transcriptome_obsm_key
    key_added = f"{conn_key}_connectivities"
    associated_knn_key = f"knn_connectivities"
    X = adata.obs[obs_keys]
    transcriptome_array = adata.obsm[obsm_key] if obsm_key in adata.obsm else None

    _snn, _snn_dist, _knn, _knn_dist = construct_SNN(
        X,
        neighbor_num,
        select_feat=obs_keys,
        geom_morph_ratio=geom_morph_ratio,
        geom_constraint=geom_constraint,
        theta=theta,
        transcriptome_array=transcriptome_array,
    )

    # if geom_constraint > 0:
    #     logger.info(f"Applying geom_constraint {geom_constraint} pixels")
    #     nn = radius_neighbors_graph(X.iloc[:, :2], radius=geom_constraint, mode="connectivity")
    #     _snn = _snn.multiply(nn)

    adata.obsp[key_added], adata.obsp[associated_knn_key] = _snn, _knn
    adata.obsp[key_added.replace("_connectivities", "_distances")], adata.obsp[associated_knn_key.replace("_connectivities", "_distances")] = _snn_dist, _knn_dist
    logger.info(f"Add adata.obsp[\"{key_added}\"]")
    logger.info(f"Add adata.obsp[\"{associated_knn_key}\"]")
    if conn_key not in adata.uns:
        adata.uns[conn_key] = {}
    adata.uns[conn_key]["neighbor_num"] = {key_added: neighbor_num}
    adata.uns[conn_key]["features"] = obs_keys
    adata.uns[conn_key]["geom_morph_ratio"] = geom_morph_ratio
    adata.uns[conn_key]["geom_constraint"] = geom_constraint
    adata.uns[conn_key]["theta"] = theta
    adata.uns[conn_key]["transcriptome"] = obsm_key
    logger.info(f"Add adata.uns[\"{conn_key}\"]")
