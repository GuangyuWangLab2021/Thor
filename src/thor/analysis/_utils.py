import logging
import json
import numpy as np

from thor.utils import resample_polygon, get_scalefactors
logger = logging.getLogger(__name__)


def clean_keys(d):
    c_dict = {}
    for k, v in d.items():
        k = k.split('.')[-1]
        c_dict[k] = v
    return c_dict


def get_polygon_from_path(p):
    p = clean_keys(p)
    coors = p['path'].split('L')
    # starts with M
    coors[0] = coors[0][1:]
    # ends with Z
    coors[-1] = coors[-1][:-1]

    if not np.array_equal(coors[0], coors[-1]):
        coors.append(coors[0])
    return np.array(
        list(map(lambda x: np.array(x.split(','), dtype=float), coors))
    )


def get_polygon_from_rect(r):

    rect_dict = clean_keys(r)
    #assert len(rect_dict) == 4
    try:
        x0 = rect_dict['x0']
        x1 = rect_dict['x1']
        y0 = rect_dict['y0']
        y1 = rect_dict['y1']
    except:
        return None
    coors = [(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)]

    return np.array(coors)


def json_parser(json_path, scalefactor=1):
    # THIS ONLY WORKS FOR REGIONS DRAWN USING PLOTLY
    with open(json_path) as json_file:
        data = json.load(json_file)
    scalef = 1 / scalefactor

    shapes = {}
    if 'shapes' in data:
        # Not in editing mode
#        if len(data['shapes']) > 1:
#            print("Multiple active shapes!")

        polygons = []
        for shape in data['shapes']:
            if shape['type'] == 'path':
                polygons.append(
                    scalef * get_polygon_from_path({'path': shape['path']})
                )
            if shape['type'] == 'rect':
                polygons.append(scalef * get_polygon_from_rect(shape))
            shapes['poly'] = polygons
    else:
        # in editing mode
        isPath = any(['path' in k for k in data])
        isRect = (len(data) == 4)
        if isPath:
            shapes['poly'] = [scalef * get_polygon_from_path(data)]
        if isRect:
            shapes['poly'] = [scalef * get_polygon_from_rect(data)]
    return shapes


def read_polygon_ROI(json_path, adata, img_key=None):
    """Read polygon ROI from json file.

    Parameters
    ----------
    json_path : :py:class:`str`
        Path to json file.
    adata : :class:`anndata.AnnData`
        Annotated data matrix.
    img_key : :py:class:`str`, optional
        Key for image in `adata.uns['spatial']` where the ROI was drawn. :py:obj:`None` for full-resolution image. Valid keys are:
        'hires', 'lowres', 'fullres'

    Returns
    -------
    roi_shape : :class:`numpy.ndarray`
        Numpy array of shape (n_vertices, 2) containing the coordinates of the ROI polygon.
    """

    if img_key == 'fullres':
        img_key = None

    scalef = get_scalefactors(adata)[f"tissue_{img_key}_scalef"] if img_key else 1
    roi_shape = json_parser(json_path, scalefactor=scalef)["poly"]

    if len(roi_shape) > 1:
        logger.warning(
            "Multiple regions (polygons) selected! Using the first region drawn."
        )
    roi_shape = roi_shape[0]

    # evenly sample along the picked polygon boundary.
    roi_shape = resample_polygon(roi_shape)

    return roi_shape
