import os
from glob import glob
import scanpy as sc
import logging

logger = logging.getLogger(__name__)


class Spatial:
    """ Class for spatial transcriptomics data.
    This class is used to minimally process spatial transcriptomics data.

    Parameters
    ----------
    name : :py:class:`str`
        Name of the spatial transcriptomics data.
    st_dir : :py:class:`str`
        Path to the directory containing the spatial transcriptomics data. This directory should contain the 10x spaceranger output.
    image_path : :py:class:`str`, optional
        Path to the full-size image file used for spatial transcriptomics.
    save_dir : :py:class:`str`, optional
        Path to the directory where the processed spatial transcriptomics data will be saved.

    """
    def __init__(self, name, st_dir, image_path=None, save_dir=None):
        self.name = name
        self.st_dir = st_dir
        self.image_path = image_path if image_path else os.path.join(st_dir, "spatial")
        self.save_dir = save_dir if save_dir else f"Spatial_{self.name}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.spot_adata_path = None

    def process_transcriptome(self, perform_QC=True, min_cells=10, min_counts=1000, max_counts=35000, max_mt_pct=20, max_rb_pct=100):
        """Process the spatial transcriptome data (sequence-based).
        This function will read the 10x spaceranger output and perform basic preprocessing steps.

        Parameters
        ----------
        perform_QC : :py:class:`bool`, optional
            Whether to perform quality control. Default is :py:obj:`True`.
        min_cells : :py:class:`int`, optional
            Minimum number of cells. Default is 10.
        min_counts : :py:class:`int`, optional
            Minimum number of counts. Default is 1000.
        max_counts : :py:class:`int`, optional
            Maximum number of counts. Default is 35000.
        max_mt_pct : :py:class:`float`, optional
            Maximum percentage of mitochondrial genes. Default is 20.
        max_rb_pct : :py:class:`float`, optional
            Maximum percentage of ribosomal genes. Default is 100.

        Note
        ----
        It is recommended that this preprocessing step is done by the users using Scanpy or Seurat.
        This function only provides very basic level of preprocessing following Scanpy.
        Please refer to the `Scanpy documentation <https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html>`_.
        """
        transdir = self.st_dir
        filtered_count_file_path = os.path.join(
            transdir, "*filtered_feature_bc_matrix.h5"
        )
        filtered_count_file = os.path.basename(
            glob(filtered_count_file_path)[0]
        )
        adata = sc.read_visium(
            transdir,
            count_file=filtered_count_file,
            source_image_path=self.image_path
        )

        try:
            adata.obsm["spatial"] = adata.obsm["spatial"].astype(float)
        except:
            pass
        adata.var_names_make_unique()

        # Filter genes and cells
        if perform_QC:
            adata = QC(adata, min_cells=min_cells, min_counts=min_counts, max_counts=max_counts, max_mt_pct=max_mt_pct, max_rb_pct=max_rb_pct)

        out_path = os.path.join(
            self.save_dir, f"{self.name}_spots.h5ad"
        )
        adata.write_h5ad(out_path)
        self.spot_adata_path = out_path


def QC(adata, min_counts=5000, max_counts=35000, min_cells=10, max_mt_pct=20, max_rb_pct=100):
    """
    This function will perform QC on the spatial transcriptomics data.

    Parameters
    ----------
    adata : :class:`anndata.AnnData`
        Annotated data matrix.
    min_counts : :py:class:`int`, optional
        Minimum number of counts. Default is 5000.
    max_counts : :py:class:`int`, optional
        Maximum number of counts. Default is 35000.
    min_cells : :py:class:`int`, optional
        Minimum number of cells. Default is 10.
    max_mt_pct : :py:class:`float`, optional
        Maximum percentage of mitochondrial genes. Default is 20.
    max_rb_pct : :py:class:`float`, optional
        Maximum percentage of ribosomal genes. Default is 100.

    Returns
    -------
    :class:`anndata.AnnData`
        Filtered annotated data matrix.
    """
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    adata.var["rb"] = adata.var_names.str.startswith(
        ("RPS", "MRP", "RPL", "rps", "mrp", "rpl")
    )
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "rb"], inplace=True)

    # Filter cells
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, max_counts=max_counts)
    adata = adata[adata.obs["pct_counts_mt"] < max_mt_pct, :]
    logger.info(f"Number of cells after MT filtering: {adata.n_obs}")

    adata = adata[adata.obs["pct_counts_rb"] < max_rb_pct, :]
    logger.info(f"Number of cells after RB filtering: {adata.n_obs}")

    return adata
