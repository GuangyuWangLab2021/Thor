import logging
import os

import numpy as np
import squidpy as sq
from PIL import Image
from scipy.sparse import csr_matrix, save_npz

# Disable tensorflow logging
logging.getLogger('tensorflow').disabled = True

from thor.utils import require_packages


Image.MAX_IMAGE_PIXELS = None

def nuclei_segmentation(image_path, save_dir=None, tile_size=(1000, 1000), method='stardist', **segment_kwds):
    """Segment nuclei from H&E stained images using stardist, cellpose or histocartography.
    The segmentation mask is saved as a sparse matrix in .npz format.

    Parameters
    ----------
    image_path : str
        Path to the full resolution image to segment.
    save_dir : str
        Path to the directory where to save the segmentation mask.
    tile_size : tuple 
        Size of the tiles to use for stardist segmentation.
    method : str
        Segmentation method to use. Can be 'stardist', 'cellpose' or 'histocartography'.
    segment_kwds : dict
        Keyword arguments to pass to the segmentation function.
    """

    PIL_im = Image.open(image_path).convert('RGB')
    im = np.array(PIL_im)
    return nuclei_segmentation_from_image_array(im, save_dir=save_dir, tile_size=tile_size, method=method, **segment_kwds)


def nuclei_segmentation_from_image_array(im_array, save_dir=None, tile_size=(1000, 1000), method='stardist', **segment_kwds):
    """Segment nuclei from H&E stained images using stardist, cellpose or histocartography.
    The segmentation mask is saved as a sparse matrix in .npz format.

    Parameters
    ----------
    im_array : np.ndarray
        Image to segment.
    save_dir : str, optional
        Path to the directory where to save the segmentation mask. The default is None (saving to current directory).
    tile_size : tuple, optional
        Size of the tiles to use for stardist segmentation. The default is (1000, 1000).
    method : str
        Segmentation method to use. Can be 'stardist', 'cellpose' or 'histocartography'.
    segment_kwds : dict
        Keyword arguments to pass to the segmentation function.
    """

    seg = {'stardist': stardist_2D_versatile_he, 
           'cellpose': cellpose_he,
           'histocartography': histocartography_2d_he
           }

    whole_img = sq.im.ImageContainer(im_array, layer="img1")

    if method.lower() == 'stardist' and tile_size is not None:
        n_tiles = (int(im_array.shape[0]/tile_size[0]), int(im_array.shape[1]/tile_size[1]), 1)
        segment_kwds.update({'n_tiles':n_tiles})

    seg_func = seg[method.lower()] 

    if method.lower() == 'histocartography':
        mask = seg_func(im_array, **segment_kwds)
    else:
        sq.im.segment(
            img=whole_img,
            layer="img1",
            channel=None,
            method=seg_func,
            layer_added='segmented_default',
            **segment_kwds
        )
        mask = whole_img['segmented_default'].data[:,:,0,0]
        mask = csr_matrix(mask)

    if save_dir == None:
        save_dir = os.getcwd()
    cell_seg_path = os.path.join(save_dir, 'nuclei_mask.npz')
    save_npz(cell_seg_path, mask)


@require_packages('histocartography')
def histocartography_2d_he(image_array, pretrained_model="monusac"):
    """ nuclei segmentation using histocartography pretrained model.
    Parameters
    ----------
    image_array : np.ndarray
        Image to segment.
    pretrained_model : str
        Name of the pretrained model to use. Can be 'monusac' or 'pannuke'.
    
    Returns
    -------
    cell_masks : np.ndarray
        Segmentation mask of the image.
    """

    from histocartography.preprocessing import NucleiExtractor
    nuclei_detector = NucleiExtractor(pretrained_data="monusac")
    cell_masks, _ = nuclei_detector.process(image_array)
    return cell_masks


@require_packages('stardist')
# Many thanks to squidpy developpers for providing examples for running nuclei segmentation from H&E staining images.
def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=0.3, n_tiles=None, verbose=True):
    # Import the StarDist 2D segmentation models.
    from stardist.models import StarDist2D

    # Import the recommended normalization technique for stardist.
    from csbdeep.utils import normalize

    #axis_norm = (1)   # normalize channels independently
    axis_norm = (0,1,2) # normalize channels jointly
    img = normalize(img, 0, 99.8, axis=axis_norm)
    model = StarDist2D.from_pretrained('2D_versatile_he')
    labels, _ = model.predict_instances(img, nms_thresh=nms_thresh, prob_thresh=prob_thresh, n_tiles=n_tiles, verbose=verbose)
    return labels


@require_packages('cellpose')
def cellpose_he(img, min_size=15, flow_threshold=0.4, cellprob_threshold=0.0, channel_cellpose=0, use_gpu=True):
    from cellpose import models
    model = models.Cellpose(model_type='nuclei', gpu=use_gpu)
    res, _, _, _ = model.eval(
        img,
        channels=[channel_cellpose, 0],
        diameter=None,
        min_size=min_size,
        invert=True,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return res
