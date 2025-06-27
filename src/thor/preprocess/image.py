import logging
import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from thor.utils import detect_outlier, get_spot_diameter_in_pixels, update_kwargs
from .seg_models import nuclei_segmentation_from_image_array 
from .nuclei_seg import load_nuclei

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

MIN_CELLS = 10


class WholeSlideImage:
    """Whole slide image class. 


    .. raw:: html

        <span style="color: red;">New: Now support external cell features as csv file.</span>


    Parameters
    ----------
    image_path: :py:class:`str`
        Path to the whole slide image.
    name: :py:class:`str`, optional
        Name of the image.
    color_image: :py:class:`bool`
        Whether the image is a color image. Default is :py:obj:`True`.
    nuclei_seg_path: :py:class:`str`, optional
        Path to the nuclei segmentation result file.
    nuclei_seg_format: :py:class:`str`, optional
        Format of the nuclei segmentation result file. Can be 'cellpose', 'mask_array_npz' or 'cellprofiler'.
    nuclei_remove_outlier: :py:class:`bool`, optional
        Whether to remove outlier cells. Default is :py:obj:`True`.
    external_cell_features_csv_path: :py:class:`str`, optional
        Pre-extracted cell features from the image. If provided, the cell features will be loaded from the csv file. Note, the externally provided cell features need to be paired with the nuclei segmentation results. The indices of the nuclei should be the segmentation labels.
    save_dir: :py:class:`str`, optional
        Path to save the results.

    Examples
    --------
    >>> from thor.pp import WholeSlideImage
    >>> wsi = WholeSlideImage(
    ...     image_path="path/to/image.tif",
    ...     name="wsi_1",
    ...     color_image=True,)
    >>> wsi.process(method="stardist", tile_size=(500, 500), min_size=10, flow_threshold=0, prob_thresh=0.1)

    """

    def __init__(
        self,
        image_path,
        name=None,
        color_image=True,
        nuclei_seg_path=None,
        nuclei_seg_format=None,
        nuclei_remove_outlier=True,
        context_size='mean',
        external_cell_features_csv_path=None,
        save_dir=None,
    ):
        self.image_path = os.path.abspath(image_path)
        self.name = name
        self.color_image = color_image
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), f"WSI_{name}")

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.nuclei_seg_path = nuclei_seg_path
        self.nuclei_seg_format = nuclei_seg_format
        self.nuclei_remove_outlier = nuclei_remove_outlier
        self.context_size = context_size

        self.im = Image.open(image_path)
        PIL_im = Image.open(image_path).convert('RGB') if color_image else Image.open(image_path).convert('L')
        self.im = np.array(PIL_im)
        self.height = self.im.shape[0]
        self.width = self.im.shape[1]

        if external_cell_features_csv_path is not None:
            assert nuclei_seg_path is not None, "Paired `nuclei_seg_path` must be provided if `external_cell_features_csv_path` is provided."
        self.external_cell_features_csv_path = external_cell_features_csv_path
        
    def nuclei_segmentation(self, **kwargs):
        """Wrapper function for nuclei segmentation :func:`thor.preprocessing.nuclei_seg.nuclei_segmentation_from_image_array`.
        """

        logger.info("Performing nuclei segmentation...")
        kwargs = update_kwargs(nuclei_segmentation_from_image_array, kwargs)
        kwargs.update({"save_dir": self.save_dir})
        return nuclei_segmentation_from_image_array(self.im, **kwargs)

    @staticmethod
    def get_bbox(self, spot_adata_path):
        """Get the bounding box of the image according to the mapped spot positions.
        
        Parameters
        ----------
        spot_adata_path : :py:class:`str`
            Path to the spot adata object.

        Returns
        -------
        LB : :py:class:`tuple`
            Lower bound of the image (lower left corner).
        UB : :py:class:`tuple`
            Upper bound of the image (upper right corner).
        """
        ad_spot = sc.read_h5ad(spot_adata_path)
        try:
            spot_diameter = get_spot_diameter_in_pixels(ad_spot)
        except:
            logger.warning(
                "spot diameter not found in the adata object, use default margin 50."
            )
            spot_diameter = 100
        spot_radius = 0.5 * spot_diameter
        LB = ad_spot.obsm['spatial'].min(axis=0) - spot_radius
        UB = ad_spot.obsm['spatial'].max(axis=0) + spot_radius
        self.bbox = (LB, UB)
        return LB, UB

    def process(self, **kwargs):
        """Process the staining image to get the cell features and nuclei segmentation results.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the `nuclei_segmentation` function.
        """

        if self.nuclei_seg_path is None:
            # Perform cell segmentation
            self.nuclei_segmentation(**kwargs)
            self.nuclei_seg_path = os.path.join(
                self.save_dir, "nuclei_mask.npz"
            )
            self.nuclei_seg_format = "mask_array_npz"

        bbox = self.bbox if hasattr(self, "bbox") else None

        self.cell_features_csv_path = os.path.join(
            self.save_dir, "cell_features.csv"
        )
        preprocess_image(
            self.image_path,
            bbox=bbox,
            color_image=self.color_image,
            nuclei_seg_path=self.nuclei_seg_path,
            nuclei_seg_format=self.nuclei_seg_format,
            context_size=self.context_size,
            remove_outlier=self.nuclei_remove_outlier,
            save_path=self.cell_features_csv_path
        )

        if self.external_cell_features_csv_path is not None:
            self.load_external_cell_features()

    def load_external_cell_features(self, exclusive=False):
        """
        Load external cell features and combine them with default features.

        Parameters
        ----------
        exclusive : :py:class:`bool`, default: :py:obj:`False`
            If :py:obj:`True`, only use external features (must include 'x' and 'y' coordinates in the first two columns).
            If :py:obj:`False`, join external features with the default cell features.

        Raises
        ------
        ValueError
            If exclusive=True and the first two columns are not named 'x' and 'y'.

        Notes
        -----
        The external CSV file is expected to have cell IDs as the index (first column).
        When exclusive=True, the first two columns must be 'x' and 'y'.
        The result is saved to `self.cell_features_csv_path`, either replacing or extending
        the existing features.
        """
        external_cell_features_df = pd.read_csv(self.external_cell_features_csv_path, index_col=0)

        if exclusive:
            logger.warning("Only the external cell features will be used. We expected the first columns to be the cell positions [x, y].")

            # quit if the first two columns are not x and y
            if external_cell_features_df.columns[0] != 'x' or external_cell_features_df.columns[1] != 'y':
                raise ValueError("The first two columns should be the cell positions [x, y].")

            external_cell_features_df.to_csv(self.cell_features_csv_path)

        else:
            # Join the external cell features with the default cell features and rewrite the csv file.
            default_cell_features_df = pd.read_csv(self.cell_features_csv_path, index_col=0)
            cell_features_df = default_cell_features_df.join(external_cell_features_df, how='left', lsuffix='_default', rsuffix='_external')
            cell_features_df.to_csv(self.cell_features_csv_path)

    def split(self, split=(2, 2), output_path=None):
        """Split the image into tiles.

        Parameters
        ----------
        split : tuple
            Number of tiles to split the image. Default is (2, 2).
        output_path : str
            Path to save the tiles. Default is None.

        Returns
        -------
        None
            The tiles will be saved to the provided path in .png format.
        """
        im = self.im
        M = self.width // split[0]  # x
        N = self.height // split[1]  # y
        tiles = [
            im[x:x + M, y:y + N] for x in range(0, self.height, M)
            for y in range(0, self.width, N)
        ]

        output_path = os.path.join(
            self.save_dir, "tiles"
        ) if output_path is None else output_path
        os.makedirs(output_path, exist_ok=True)

        k = 0
        for i in tiles:
            Image.fromarray(i).save(f"{output_path}/subset_{k}.png")
            k += 1


def preprocess_image(
    image_path,
    bbox=None,
    color_image=True,
    nuclei_seg_path=None,
    nuclei_seg_format=None,
    nuclei_centroids_path=None,
    context_size='mean',
    extract_image_feature_custom_func=None,
    remove_outlier=True,
    save_path=None,
):
    """Preprocess the image and extract features from the cells.

    Parameters
    ----------
    image_path : str
        Path to the full resolution image to process.
    bbox : tuple, optional
        Bounding box of the image. Format is (`lower_left`, `upper_right`). If not provided, the whole image will be processed. `lower_left` and `upper_right` are tuples of x and y coordinates.
    color_image : bool, optional
        Whether to extract color features from the image. Default is True.
    nuclei_seg_path : str
        Path to the nuclei segmentation result file.
    nuclei_seg_format : str
        Format of the nuclei segmentation result file. Can be 'cellpose', 'mask_array_npz' or 'cellprofiler'.
    nuclei_centroids_path : str
        Path to the nuclei centroids. The format should be a csv file with columns `x` and `y` for the x and
        y coordinates of the nuclei centroids. Index of the dataframe should be the cell labels.
    context_size : numeric or str, optional
        Radius of the square image patch to extract around each cell (unit: pixel). Valid values are numeric or 'median', 'mean', 'min', 'max'.
        If provided as `str`, the radius is estimated from the nearest cell distances.
    extract_image_feature_custom_func : function
        Custom function to extract additional features from the image patches. The function should be written in a way that takes a list of
        image patches as input and return a dataframe of features.
    remove_outlier : bool
        Whether to remove outlier cells. Default is True.
    save_path : str
        Path to save the extracted features.
    
    """

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    if nuclei_centroids_path is not None:
        nuclei_centroids = pd.read_csv(nuclei_centroids_path, index_col=0)
        labels = nuclei_centroids.index
        nuclei_centroids = nuclei_centroids[['x','y']].values
    else:
        labels, nuclei_centroids = load_nuclei(
            nuclei_path=nuclei_seg_path,
            source_format=nuclei_seg_format,
        )

    n_cells = nuclei_centroids.shape[0]
    logger.info(f"{n_cells} cells detected.")

    if bbox is not None:
        # remove cells outside of the spots range
        nuclei_to_keep = get_inbound(nuclei_centroids, bbox)
        nuclei_centroids = nuclei_centroids[nuclei_to_keep]
        labels = labels[nuclei_to_keep]
        LB, UB = bbox
    else:
        LB, UB = None, None

    n_cells_mapped = len(labels)
    logger.info(f"{n_cells_mapped} cells mapped to spots.")
    # remove outlier cells
    if remove_outlier and n_cells_mapped > MIN_CELLS:
        inlier_indices = detect_outlier(
            nuclei_centroids, n_neigh=MIN_CELLS, return_outlier=False
        )
        nuclei_centroids = nuclei_centroids[inlier_indices]
        labels = labels[inlier_indices]

    logger.info(f"{len(labels)} cells kept.")
    extract_image_features(
        labels,
        nuclei_centroids,
        image_path,
        LB=LB,
        UB=UB,
        color_image=color_image,
        context_size=context_size,
        extract_image_feature_custom_func=extract_image_feature_custom_func,
        save_path=save_path,
    )


def extract_image_features(
    labels,
    nuclei_centroids,
    image_path,
    LB=None,
    UB=None,
    color_image=True,
    context_size='mean',
    extract_image_feature_custom_func=None,
    save_path=None
):
    """Extract image features from the image patches centered at the nuclei centroids.
    Parameters
    ----------
    labels: list
        List of cell labels as in the original segmentation array.
    nuclei_centroids : numpy array
        Nuclei centroids.
    image_path : str
        Path to the full resolution image to process.
    LB : tuple
        Lower bound of the image.
    UB : tuple
        Upper bound of the image.
    color_image : bool
        Whether to extract color features from the image.
    context_size : numeric or str, optional
        Radius of the square image patch to extract around each cell (unit: pixel). Valid values are numeric or 'median', 'mean', 'min', 'max'.
        If provided as `str`, the radius is estimated from the nearest cell distances.
    extract_image_feature_custom_func : function
        Custom function to extract additional features from the image patches. The function should be written in a way that takes a list of
        image patches as input and return a dataframe of features.
    save_path : str
        Path to save the extracted features.

    Notes
    -----
    nuclei coordinates in closest integers.
    """

    context_size_usage_message = "Use 'median', 'mean', 'min', 'max' or a positive number in pixels."
    if isinstance(context_size, str):
        context_size = context_size.lower()
        assert context_size in ['median', 'mean', 'min', 'max'], context_size_usage_message
    
        meas_funcs = {'median': np.median, 'min': np.min, 'max': np.max, 'mean': np.mean}
        nn_1 = kneighbors_graph(
            nuclei_centroids,
            n_neighbors=1,
            mode='distance',
            include_self=False
        )

        #print(meas_funcs[context_size])
        context_size = 2 * meas_funcs[context_size](nn_1.data)
        context_size = int(np.round(context_size))
    elif isinstance(context_size, (int, float)):
        assert context_size > 0, context_size_usage_message
        context_size = int(context_size)
    else:
        raise ValueError(context_size_usage_message)

    logger.info(
        f"Extracting image features from a square image patch of size {2*context_size} pixels centered at the cell."
    )
    
    PIL_im_all_chn = Image.open(image_path)
    PIL_im_gray = PIL_im_all_chn.convert('L')
    cell_image_patches = crop_image(
        nuclei_centroids,
        PIL_im_gray,
        LB=LB,
        UB=UB,
        coverage=context_size
    )
    #assert len(cell_image_patches) == len(nuclei_centroids)

    gray_feature_df = extract_gray_image_features(cell_image_patches)
    feature_df = gray_feature_df

    if color_image:
        PIL_im_RGB = PIL_im_all_chn.convert('RGB')
        cell_image_patches = crop_image(
            nuclei_centroids,
            PIL_im_RGB,
            LB=LB,
            UB=UB,
            coverage=context_size
        )
        color_feature_df = extract_color_image_features(cell_image_patches)
        feature_df = pd.concat([feature_df, color_feature_df], axis='columns')

    if callable(extract_image_feature_custom_func):
        # Here the patches match the original images (color or gray)
        additional_feature_df = extract_image_feature_custom_func(
            cell_image_patches
        )
        feature_df = pd.concat(
            [feature_df, additional_feature_df], axis='columns'
        )

    feature_df.index = labels
    pos_df = pd.DataFrame(nuclei_centroids, columns=['x', 'y'], index=labels)
    feature_df = pd.concat([pos_df, feature_df], axis='columns')
    feature_df.to_csv(save_path)


def extract_gray_image_features(images_list):
    """
    Extract cell features from a color image read from provided path.
    """

    n_images = len(images_list)

    props_gray = [
        'mean_gray',
        'std_gray',
        'entropy_img',
    ]

    props = props_gray
    props_dict = {i: np.zeros(n_images) for i in props}

    # centroid starts from 0
    for cell_idx in tqdm(range(n_images), desc="Extracting gray image features"):
        cropped_im_gray = images_list[cell_idx]
        props_dict['entropy_img'][cell_idx] = cropped_im_gray.entropy()
        props_dict['mean_gray'][cell_idx] = np.mean(cropped_im_gray)
        props_dict['std_gray'][cell_idx] = np.std(cropped_im_gray)

    img_props = pd.DataFrame(props_dict)
    return img_props


def extract_color_image_features(images_list):
    """
    Extract cell features from a color image read from provided path.
    """
    n_images = len(images_list)

    props_color = [
        'mean_r',
        'mean_g',
        'mean_b',
        'std_r',
        'std_g',
        'std_b',
    ]

    props = props_color
    props_dict = {i: np.zeros(n_images) for i in props}

    # centroid starts from 0
    for cell_idx in tqdm(range(n_images), desc="Extracting RGB features"):
        # color properties
        cropped_im = images_list[cell_idx]
        mean_rgb = np.mean(
            cropped_im, axis=tuple(range(np.ndim(cropped_im) - 1))
        )
        std_rgb = np.std(cropped_im, axis=tuple(range(np.ndim(cropped_im) - 1)))
        props_dict['mean_r'][cell_idx] = mean_rgb[0]
        props_dict['std_r'][cell_idx] = std_rgb[0]
        props_dict['mean_g'][cell_idx] = mean_rgb[1]
        props_dict['std_g'][cell_idx] = std_rgb[1]
        props_dict['mean_b'][cell_idx] = mean_rgb[2]
        props_dict['std_b'][cell_idx] = std_rgb[2]

    img_props = pd.DataFrame(props_dict)
    return img_props


def crop_image(cells, PIL_im, LB=None, UB=None, coverage=5):
    """Crop image patches around the cells.
    """

    dim = PIL_im.size
    LB_im = (0, 0)
    UB_im = (dim[0] - 1, dim[1] - 1)  # x -> horizontal, y -> verticle

    if LB is None:
        LB = LB_im
    else:
        LB = np.max((LB, LB_im), axis=0)

    if UB is None:
        UB = UB_im
    else:
        UB = np.min((UB, UB_im), axis=0)

    pieces = []
    for cell_pos in cells:
        cell_pos = np.array(cell_pos).astype(int)

        cell_pos_lower = cell_pos - coverage
        cell_pos_upper = cell_pos + coverage

        area = np.clip(np.array([cell_pos_lower, cell_pos_upper]), LB,
                       UB).ravel()
        pieces.append(PIL_im.crop(area))
    return pieces


def get_inbound(cell_pos, bound):
    LB, UB = bound
    x_min, y_min = LB
    x_max, y_max = UB

    select = np.logical_and(
        np.logical_and(cell_pos[:, 0] >= x_min, cell_pos[:, 0] <= x_max),
        np.logical_and(cell_pos[:, 1] >= y_min, cell_pos[:, 1] <= y_max)
    )
    return select

