import logging

import numpy as np
from scipy.sparse import csr_array, csr_matrix
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph, NearestNeighbors
from collections import Counter
from statistics import median

from ._math import row_normalize_sparse
from ._adata import get_adata_layer_array

logger = logging.getLogger(__name__)


def spatial_smooth(pos, attr, n_neighbors=100, radius=None):
    """ smoothing the data `attr` using the nearest neighbours in spatial space `pos`

    Parameters
    ----------
    pos: :class:`numpy.ndarray` (n_cells, 2)
        spatial coordinates of cells
    attr: :class:`numpy.ndarray` (n_cells, n_vars)
        data to be smoothed
    n_neighbors: :py:class:`int`
        number of neighbours to use for smoothing
    radius: :py:class:`float`
        radius of the neighbourhood to use for smoothing. If n_neighbors is not :py:obj:`None`, this will be ignored.

    Returns
    -------
    :class:`numpy.ndarray` (n_cells, n_vars)
        smoothed data
    """

    if n_neighbors is not None:

        nn_graph = kneighbors_graph(pos,
                                    n_neighbors=n_neighbors,
                                    include_self=True)
    elif radius is not None:
        nn_graph = radius_neighbors_graph(pos,
                                          radius=radius,
                                          include_self=True)
    else:
        raise ValueError("Either n_neighbors or radius must be specified.")

    nn_graph = row_normalize_sparse(nn_graph)

    attr_smooth = nn_graph * attr
    return attr_smooth


def knn_smooth(pos, attr, n_neighbors=100):
    """ smoothing the data `attr` using the k-nearest neighbours in spatial space `pos`
    Wrapper for spatial_smooth.
    """

    return spatial_smooth(pos, attr, n_neighbors=n_neighbors)


def nn_to_dict(nn):
    """ Convert neighbour graph array to dictionary
    Parameters
    ----------
    nn: :class:`scipy.sparse.csr_matrix`
            cell-cell neighbours matrix
    Returns
    -------
    :py:class:`dict`
            dictionary of neighbours
    """

    NNdict = {
        cellIndex: nn[cellIndex].indices
        for cellIndex in range(nn.shape[0])
    }
    return NNdict


def getNN2_dict(nn):
    """ Get the second level neighbouring cells of each cell
    Parameters
    ----------
    nn: :class:`scipy.sparse.csr_matrix`
            cell-cell neighbours matrix
    Returns
    -------
    :py:class:`dict`
                dictionary of second level neighbours
    """

    NNdict = nn_to_dict(nn)
    NN2_dict = {
        cellIndex: np.sort(np.unique(nn[NNdict[cellIndex]].indices))
        for cellIndex in range(nn.shape[0])
    }
    return NN2_dict


def dict_to_nn(NN_dict):
    """ Convert dictionary of neighbours to neighbour graph array
    Parameters
    ----------
    NN_dict: :py:class:`dict`
                dictionary of neighbours
    Returns
    -------
    :class:`scipy.sparse.csr_matrix`
                cell-cell neighbours matrix
    """

    row = []
    col = []
    size = 0

    n_cells = len(NN_dict)

    for s, cs in NN_dict.items():
        ncs = len(cs)
        row += [s] * ncs
        col += list(cs)
        size += ncs

    data = np.ones(size)
    NN_graph_arr = csr_array((data, (row, col)), shape=(n_cells, n_cells))
    NN_graph = csr_matrix(NN_graph_arr)
    return NN_graph


def get_2nd_spot_neigh_graph(nn, cellxspot):
    """ Get the spots in which at least one of their cells are first or second level neighbours of a cell.
    Parameters
    ----------
    nn: :class:`scipy.sparse.csr_matrix`
                cell-cell neighbours matrix
    cellxspot: :class:`scipy.sparse.csr_matrix`
                cell-spot mapping sparse matrix
    Returns
    -------
    :class:`scipy.sparse.csr_matrix`
                immediate and second-level neighboring spots (neighbor graph)
    """
    NN2_dict = getNN2_dict(nn)
    NN2_graph = dict_to_nn(NN2_dict)
    cellxspotNN2 = NN2_graph * cellxspot
    cellxspotNN = nn * cellxspot

    cellxspotNN_1_and_2 = cellxspotNN + cellxspotNN2

    return cellxspotNN_1_and_2 > 0


def get_stats_in_neighboring_cells(x, nn):
    """ Get the mean and standard deviation of gene expression in neighboring cells
    Parameters
    ----------
    x: :class:`numpy.ndarray` (n_cells, n_genes)
            gene expression
    nn: :class:`scipy.sparse.csr_matrix`
            cell-cell neighbours matrix
    Returns
    -------
    x_: :class:`numpy.ndarray` (n_cells, n_genes)
            mean gene expression in neighboring cells
    x_sigma: :class:`numpy.ndarray` (n_cells, n_genes)
            standard deviation of gene expression in neighboring cells
    """

    nn = row_normalize_sparse(nn)
    x2_ = nn * (x * x)
    x_ = nn * x
    x_sigma = np.sqrt(x2_ - x_ * x_)

    return x_, x_sigma


def get_2nd_spot_neighbours(nn, cellxspot):
    """ Get the second level neighbouring cells of each cell
    Parameters
    ----------
    nn: :class:`scipy.sparse.csr_matrix`
                cell-cell neighbours matrix
    cellxspot: :class:`scipy.sparse.csr_matrix`
                cell-spot mapping sparse matrix
    Returns
    -------
    :class:`scipy.sparse.csr_matrix`
                        cell-cell neighbours matrix of second-level spot neighbours
    """
    # get the second level neighbouring spots of each cell
    cellxspot2 = get_2nd_spot_neigh_graph(nn, cellxspot)

    # get all cells in neighbouring spots of each cell
    cellxspot2xcell = cellxspot2 * cellxspot.T
    cellxspot2xcell = (cellxspot2xcell > 0)
    
    return cellxspot2xcell


def get_2nd_cell_neighbours(nn):
    """ Get the second level neighbouring cells of each cell
    Parameters
    ----------
    nn: :class:`scipy.sparse.csr_matrix`
                cell-cell neighbours matrix
    Returns
    -------
    :class:`scipy.sparse.csr_matrix`
                cell-cell neighbours matrix of second-level neighbours
    """
    NN2_dict = getNN2_dict(nn)
    NN2_graph = dict_to_nn(NN2_dict)

    return NN2_graph


def median_num_cells_mapped_to_spots(df, spot_identifier='spot_barcodes'):
    """ Calculate the median number of cells mapped to each spot.
    
    Parameters
    ----------
    df: pd.DataFrame
        cell-level dataframe
    spot_identifier: str
        The column name of the spot identifier. Default: 'spot_barcodes'
        
    Returns
    -------
    median_num_cells_mapped_to_spots: int
        The median number of cells mapped to each spot.
    """

    assert spot_identifier in df.columns, f"{spot_identifier} not in df.columns."
    spots_barcodes = df[spot_identifier].values
    C = Counter(spots_barcodes)
    return int(median(C.values()))


def get_adata_layer_array_smooth(adata, layer_key=None):
    X = get_adata_layer_array(adata, layer_key=layer_key)
    n_neigh = 4 * median_num_cells_mapped_to_spots(adata.obs, spot_identifier='spot_barcodes')
    x_ref = knn_smooth(adata.obsm['spatial'], X, n_neigh)
    return x_ref


def kneighbors_graph_with_geometrical_constraint(X, n, pos, geom_constraint):
    """ Get the k-nearest neighbors graph with a geometrical constraint.
    The expected usage is to get the distance matrix from an embedding and filter the connections so that only local connections are included, then use this function to get the k-nearest neighbors graph.

    Parameters
    ----------
    X: numpy.ndarray
        Feature matrix. Expected to be column normalized.
    n: int
        Number of neighbors.
    pos: numpy.ndarray
        Spatial coordinates of cells.
    geom_constraint: float
        Geometrical constraint. The distance between two cells must be less than this value to be considered as a connection.
    
    Returns
    -------
    knn_graph: scipy.sparse.csr_matrix
        k-nearest neighbors connectivity graph in scipy.sparse.csr_matrix format.
    distance_graph: scipy.sparse.csr_matrix
        k-nearest neighbors distance graph in scipy.sparse.csr_matrix format.
    """

    assert geom_constraint > 0, "geom_constraint must be positive."

    n_cells = X.shape[0]

    radius_graph = radius_neighbors_graph(
        pos, radius=geom_constraint, mode="connectivity"
    )

    i, j = radius_graph.nonzero()

    d_ij = X[j] - X[i]
    d_ij = np.linalg.norm(d_ij, axis=1)

    feature_dist = csr_matrix((d_ij, (i, j)), shape=(n_cells, n_cells))

    return kneighbors_graph_sparse(feature_dist, n)


def kneighbors_graph_sparse(distance_matrix, n_neighbors):
    """ Get the k-nearest neighbors graph from a distance matrix in scipy.sparse.csr_matrix format.
    The expected usage is to get the distance matrix from an embedding and filter the connections so that only local connections are included, then use this function to get the k-nearest neighbors graph.

    Parameters
    ----------
    distance_matrix: scipy.sparse.csr_matrix
        Distance matrix in scipy.sparse.csr_matrix format.
    n_neighbors: int
        Number of neighbors. If the number of neighbors is greater than the number of available connections in the distance matrix, all available connections will be used.

    Returns
    -------
    knn_graph: scipy.sparse.csr_matrix
        k-nearest neighbors graph in scipy.sparse.csr_matrix format.
    distance_graph: scipy.sparse.csr_matrix
        modified distance matrix with only the k-nearest neighbors.

    Note
    ----
    This function is brutal force and is not recommended for distance matrices of low sparsity.
    """

    n_cells = distance_matrix.shape[0]
    indices = distance_matrix.indices
    indptr = distance_matrix.indptr

    row = []
    col = []

    for row_index in np.arange(distance_matrix.shape[0]):
        row_data = distance_matrix.data[indptr[row_index]:indptr[row_index+1]]
        row_indices = indices[indptr[row_index]:indptr[row_index+1]]

        # Find the indices of the top n values using argpartition
        n = n_neighbors if len(row_data) >= n_neighbors else len(row_data)
        if n == 0:
            logger.warning("No neighbors found for cell ", row_index)
            continue
        partition_indices = np.argpartition(row_data, n-1)[:n]

        # Get the smallest n values and their corresponding indices
        top_n_values = row_data[partition_indices]
        top_n_indices = row_indices[partition_indices]

        for i in range(n):
            row.append(row_index)
            col.append(top_n_indices[i])

    knn_graph = csr_matrix((np.ones(len(row)), (row, col)), shape=(n_cells, n_cells))
    distances = distance_matrix[row, col].A1
    distance_graph = csr_matrix((distances, (row, col)), shape=(n_cells, n_cells))

    return knn_graph, distance_graph

def kneighbors_conndist_graph(X, n_neighbors, include_self=False):
    """
    This function returns both the connectivity graph and the distance graph of the k-nearest neighbors of each cell. It is intended to be used
    only when you need both the connectivity graph and the distance graph of the k-nearest neighbors at the same time to avoid recomputing the
    neighbors.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1 if not include_self else n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)

    if not include_self:
        # Exclude the first neighbor (self)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    # Create connectivity graph as csr_matrix
    rows = np.repeat(np.arange(X.shape[0]), n_neighbors)
    cols = indices.flatten()
    connectivity_data = np.ones(len(rows))
    connectivity_graph = csr_matrix((connectivity_data, (rows, cols)), shape=(X.shape[0], X.shape[0]))

    # Create distance graph as csr_matrix
    distance_data = distances.flatten()
    distance_graph = csr_matrix((distance_data, (rows, cols)), shape=(X.shape[0], X.shape[0]))

    return connectivity_graph, distance_graph
