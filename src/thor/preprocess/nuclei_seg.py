import logging

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


def load_nuclei(nuclei_path=None, source_format='cellpose'):
    """Load nuclei segmentation result from a file.

    Parameters
    ----------
    nuclei_path : str
        Path to the nuclei segmentation result file.
    source_format : str
        Format of the nuclei segmentation result file. Options: 'cellprofiler', 'cellpose', 'mask_array_npz'.

    Returns
    -------
    labels : numpy array
        Nuclei segmentation labels (numpy array: n_cells)
    nuclei_centroids : numpy array
        Nuclei positions (numpy array: n_cells x 2)
    """

    read_nuclei = {
        'cellprofiler': load_cellprofiler,
        'cellpose': load_cellpose,
        'mask_array_npz': load_mask_npz
    }

    func = read_nuclei.get(source_format.lower(), None)

    try:
        return func(nuclei_path)
    except Exception as e:
        logger.error(f"Failed to load nuclei segmentation result from file: {nuclei_path}")
        logger.error(e)
        raise e


def load_cellpose(nuclei_path):
    """Load nuclei segmentation result from a cellpose output file.
    
    Parameters
    ----------
    nuclei_path : str
        Path to the nuclei segmentation result file.
    
    Returns
    -------
    labels : numpy array
        Nuclei segmentation labels (numpy array: n_cells)
    nuclei_centroids : numpy array
        Nuclei positions (numpy array: n_cells x 2)
    """

    seg = np.load(nuclei_path, allow_pickle=True).item()
    cell_masks = seg['masks']  
    labels, nuclei_centroids = get_nuclei_centroids(cell_masks)
    return labels, nuclei_centroids


def load_cellprofiler(nuclei_path):
    """Load nuclei segmentation result from a cellprofiler output file.

    Parameters
    ----------
    nuclei_path : str
        Path to the nuclei segmentation result file.

    Returns
    -------
    labels : numpy array
        Nuclei segmentation labels (numpy array: n_cells)
    nuclei_centroids : numpy array
        Nuclei positions (numpy array: n_cells x 2)
    """

    df = pd.read_csv(nuclei_path, sep=",")
    labels = np.array(df['ObjectNumber'].values) 
    centroids = np.array(df[['AreaShape_Center_X', 'AreaShape_Center_Y']].values)
    return labels, centroids


def load_mask_npz(nuclei_path):
    """Load nuclei segmentation result from a mask array npz file.
    
    Parameters
    ----------
    nuclei_path : str
        Path to the nuclei segmentation result file.

    Returns
    -------
    labels : numpy array
        Nuclei segmentation labels (numpy array: n_cells)
    nuclei_centroids : numpy array
        Nuclei positions (numpy array: n_cells x 2)
    """
    cmask = load_npz(nuclei_path)
    labels, nuclei_centroids = get_nuclei_centroids(cmask.toarray())
    return labels, nuclei_centroids


def get_nuclei_centroids(cell_masks):
    """Get nuclei centroids from nuclei segmentation masks.
    
    Parameters
    ----------
    cell_masks : numpy array
        Nuclei segmentation masks
    
    Returns
    -------
    labels : numpy array
        Nuclei segmentation labels (numpy array: n_cells)
    centroids : numpy array
        Nuclei positions (numpy array: n_cells x 2)
    """

    regions = regionprops(cell_masks)
    centroids = np.array([region.centroid for region in regions])
    labels = np.array([region.label for region in regions])
    return labels, centroids[:, [1, 0]]

