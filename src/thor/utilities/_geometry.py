import numpy as np
from scipy.stats import zscore
from shapely import geometry
from sklearn.neighbors import kneighbors_graph


def detect_outlier(pos_array, zscore_cutoff=4, n_neigh=10, return_outlier=True):
    """detect outliers based on average distance of the nearest neighbours. Any dots outside of zscore > zscore_cutoff will be excluded (too far
    away from its neighbours).

    Parameters
    ----------
    pos_array : numpy.ndarray
        2D array of x and y coordinates of dots.
    zscore_cutoff : float, optional
        zscore cutoff for outlier detection. The default is 4.
    n_neigh : int, optional
        number of nearest neighbours to consider. The default is 10.
    return_outlier : bool, optional
        whether to return the outlier indices. The default is True. If False, return the indices of non-outliers.

    Returns
    -------
    Depending on return_outlier, either,
    outlier_indices : numpy.ndarray
        indices of outliers.
    non_outlier_indices : numpy.ndarray
        indices of non-outliers.
    
    """

    adj = kneighbors_graph(
        pos_array, n_neighbors=n_neigh, mode='distance', include_self=False
    )
    closeness = adj.mean(axis=1)
    closeness = zscore(np.array(closeness)[:, 0])

    if return_outlier:
        outlier_indices = np.where(closeness > zscore_cutoff)[0]
        return outlier_indices
    else:
        non_outlier_indices = np.where(closeness <= zscore_cutoff)[0]
        return non_outlier_indices


def inside_polygon(xy, polygon):
    points_inside_polygon = list(map(polygon.contains, map(geometry.Point, xy)))

    return np.array(points_inside_polygon)


def get_region(ad, polygon_coors):
    if type(polygon_coors) in (geometry.polygon.Polygon, geometry.multipolygon.MultiPolygon):
        polygon = polygon_coors
    else:
        assert np.array_equal(
            polygon_coors[0], polygon_coors[-1]
        ), "Polygon is not closed. Closing it by connecting the first and last vertices."
        #polygon = [(1, 5), (2, 5), (2, 2), (1, 2), (1, 5)]
        polygon = geometry.Polygon(polygon_coors)

    xy = ad.obsm['spatial']
    ad_roi = ad[inside_polygon(xy, polygon)]
    return ad_roi.copy()


def get_ROI_tuple_from_polygon(polygon_coors, extend_pixels=100):
    xmax, ymax = polygon_coors.max(axis=0).astype(int)
    xmin, ymin = polygon_coors.min(axis=0).astype(int)

    ROI_tuple = (xmin-extend_pixels, ymin-extend_pixels, xmax-xmin+2*extend_pixels, ymax-ymin+2*extend_pixels)
    return ROI_tuple


def resample_polygon(polygon_coors, n_points=100):
    """
    resample the boundary line of the polygon by interpolation.
    """

    d = np.cumsum(
        np.r_[0, np.sqrt((np.diff(polygon_coors, axis=0)**2).sum(axis=1))]
    )

    d_sampled = np.linspace(0, d.max(), n_points)

    polygon_coors_interp = np.c_[
        np.interp(d_sampled, d, polygon_coors[:, 0]),
        np.interp(d_sampled, d, polygon_coors[:, 1]),
    ]

    return polygon_coors_interp


def on_patch_rect(xy, xy_range):
    x_min, x_max, y_min, y_max = xy_range
    x = xy[:, 0]
    y = xy[:, 1]

    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

def rect_from_ROI_tuple(ROI_tuple):
    xmin, ymin, width, height = ROI_tuple
    xmax = xmin + width
    ymax = ymin + height

    lower_left = (xmin, ymin)
    lower_right = (xmax, ymin)
    upper_left = (xmin, ymax)
    upper_right = (xmax, ymax)

    return np.array((lower_left, upper_left, upper_right, lower_right, lower_left))



