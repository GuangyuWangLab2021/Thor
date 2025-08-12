import logging
import pandas as pd

from anndata import AnnData

logger = logging.getLogger(__name__)

def spot2cell(cell_locs, spot_locs, cell_radius):
    """
    Aggregate the spots to the cells based on the distance between the spots and the cells.

    Parameters:
    - cell_locs: numpy array of shape (n_cells, 2) with the x, y coordinates of the cells.
    - spot_locs: numpy array of shape (n_spots, 2) with the x, y coordinates of the spots.
    - cell_radius: float, the radius of the cells.

    Returns:
    - assignments: A csr_matrix of shape (n_cells, n_spots) with the assignments of the spots to the cells.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # Initialize the NearestNeighbors model with the specified radius
    neigh = NearestNeighbors(radius=cell_radius)

    # Fit the model on the spot locations
    neigh.fit(spot_locs)

    # Create the radius neighbors graph
    A = neigh.radius_neighbors_graph(cell_locs, mode='connectivity')

    # Exclude empty rows
    non_empty_rows = np.diff(A.indptr) != 0
    A = A[non_empty_rows]

    return A, non_empty_rows


def aggregate_features(spot_features, cellxspot, method='mean'):
    """
    Aggregate the spot features to the cells based on the assignments.

    Parameters:
    - spot_features: csr_matrix of shape (n_spots, n_features) with the features of the spots.
    - assignments: list of lists, the list of spots assigned to each cell.

    Returns:
    - cell_features: csr_matrix of shape (n_cells, n_features) with the aggregated features of the cells.
    """

    print("The shape of the transformed cellxspot is: ", cellxspot.shape)
    assert method in ['mean', 'sum']

    # When you transpose a csr_matrix, it becomes a dense array...
    if method == 'mean':
        cellxspot = row_normalize_sparse(cellxspot)

    cell_features = cellxspot @ spot_features

    return cell_features


def HD2cell(adata_spot, node_feat, margin=1, cell_radius=10):
    """ Assigning ST of segmented cells by the nearest spots (1 or 0 for each cell). If the nearest spot is within the margin, the cell is assigned to the spot.
    
    Parameters
    ----------
    adata_spot: AnnData
        The AnnData object containing the spot data.
    node_feat: pd.DataFrame
        The dataframe containing the cell features.
    margin: float
        The scale for assigning the spots to the cells. 1 means the spot is within the cell radius.
    cell_radius: float
        The radius of the cells in microns.

    Returns
    -------
    adata_cell: AnnData
        The AnnData object containing the cell data.
    assignments: list
        The list of spot indices that are mapped to cells.
    """

    spot_pos = pd.DataFrame(
        adata_spot.obsm['spatial'],
        columns=['x', 'y'],
        index=adata_spot.obs.index
    )

    scalf = get_scalefactors(adata_spot)
    microns_per_pixel = scalf['microns_per_pixel']

    cell_r = cell_radius / microns_per_pixel

    cell_pos = node_feat[['x', 'y']]

    # Assign spots to cells
    assignments, non_empty_rows = spot2cell(cell_pos.values, spot_pos.values, cell_r * margin)

    # Aggregate the features
    cell_features = aggregate_features(adata_spot.X, assignments, method='mean')
    node_feat = node_feat[non_empty_rows]

    # Create the cell AnnData object
    adata_cell = AnnData(
        X=cell_features, obs=node_feat, var=adata_spot.var, uns=adata_spot.uns
    )

    adata_cell.obsm['spatial'] = node_feat.iloc[:, :2].values

    return adata_cell, assignments


def get_library_id(adata):
    assert 'spatial' in adata.uns, "spatial not present in adata.uns"
    library_ids = adata.uns['spatial'].keys()
    try:
        library_id = list(library_ids)[0]
        return library_id
    except IndexError:
        logger.error('No library_id found in adata')


def get_scalefactors(adata, library_id=None):
    if library_id is None:
        library_id = get_library_id(adata)
    try:
        scalef = adata.uns['spatial'][library_id]['scalefactors']
        return scalef
    except IndexError:
        logger.error('scalefactors not found in adata')


def row_normalize_sparse(csr):
    """
    Each row sums to 1
    """
    from sklearn.preprocessing import normalize

    csr_normalized = normalize(csr, norm='l1', axis=1)
    return csr_normalized


if __name__ == '__main__':
    import pandas as pd
    import scanpy as sc

    spot_adata_path = 'path to processed_002micro.h5ad'
    cell_feature_path = 'path to cell_features.csv'

    adata_spot = sc.read_h5ad(spot_adata_path)
    cell_features = pd.read_csv(cell_feature_path, index_col=0)

    adata_cell, assignments = HD2cell(adata_spot, cell_features)

    print(adata_cell)
    print(assignments.shape)
