import logging
import os
import subprocess
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from thor.utils import get_adata_layer_array

logger = logging.getLogger(__name__)


def prepare_and_run_copykat(
    adata: AnnData,
    datadir: Optional[str] = None,
    layer: Optional[str] = None,
    batch_size: int = 10,
    id_type: str = "S",
    ngene_chr: int = 5,
    win_size: int = 25,
    KS_cut: float = 0.1,
    sam_name: str = "",
    distance: str = "euclidean",
    norm_cell_names: str = "",
    output_seg: str = "FALSE",
    plot_genes: str = "TRUE",
    genome: str = "hg20",
    n_cores: int = 1,
    copykat: bool = True
) -> None:
    """
    Run CopyKAT on the input data. Requires R and the CopyKAT package to be installed. Refer to the CopyKAT documentation for more information:
    `CopyKAT <https://github.com/navinlabcode/copykat/blob/b795ff793522499f814f6ae282aad1aab790902f/R/copykat.R>`_.

    Parameters
    ----------
    adata : AnnData
        Gene expression data.
    datadir : str, optional
        Directory where data will be saved. Default is None. If None, data will be saved in the current working directory.
    layer : str, optional
        Name of the layer in `adata` to use as input data. Default is None. If None, the X layer will be used.
    batch_size : int
        Number of subfolders to process in parallel. Default is 10.
    id_type : str
        CopyKAT parameter `id.type`. Gene identification type. Default is "S"("s"), which refers to gene symbol. Other option is "E"("e") for Ensembl ID. 
    ngene_chr : int
        CopyKAT parameter `ngene.chr`. Minimal number of genes per chromosome for cell filtering. Default is 5.
    win_size : int
        CopyKAT parameter `win.size`. Minimal window sizes for segmentation. Default is 25.
    KS_cut : float
        CopyKAT parameter `KS.cut`. Segmentation parameter ranging from 0 to 1; larger value means looser criteria. Default is 0.1.
    sam_name : str
        CopyKAT parameter `sam.name`. Sample name. Default is "".
    distance : str
        CopyKAT parameter `distance`. Distance metric. Default is "euclidean". Other options are "pearson" and "spearman".
    norm_cell_names : str
        CopyKAT parameter `norm.cell.names`. A vector of normal cell names. Default is "".
    output_seg : str 
        CopyKAT parameter `output.seg`. Whether to output segmentation results for IGV visualization. Default is "FALSE". Other option is
        "TRUE". Note that it is a string and not a boolean.
    plot_genes : str
        CopyKAT parameter `plot.genes`. Whether to output heatmap of CNAs with genename labels. Default is "TRUE". Other option is "FALSE". Note that it is a string and not a boolean.
    genome : str
        CopyKAT parameter `genome`. Genome name. Default is "hg20" for human genome version 20. Other option is "mm10" for mouse genome version 10.
    n_cores : int
        CopyKAT parameter `n.cores`. Number of CPU cores for parallel computing. Default is 1. Recommended to use 1 core if batch_size > 1.
    copykat : bool
        Whether to run the CopyKAT analysis. Default is True. If False, the function will only split the data into smaller chunks and prepare the R script for CopyKAT.

    Returns
    -------
    None
        Results are saved in the current working directory.

    """

    # Warn user that CopyKAT need to be installed
    logger.warning(
        "Rscript and CopyKAT needs to be installed. Please refer to the CopyKAT documentation for more information:`CopyKAT <https://github.com/navinlabcode/copykat/blob/b795ff793522499f814f6ae282aad1aab790902f/R/copykat.R>`_."
    )


    if datadir is None:
        datadir = os.getcwd()
    else:
        datadir = datadir
    source = os.path.join(datadir, "split_data_forcopykat")

    
    # Run split function and get folder names
    out_dirs = prepare_data_for_parallel_processing(
        adata, source, layer, id_type, ngene_chr, win_size, KS_cut, sam_name,
        distance, norm_cell_names, output_seg, plot_genes, genome, n_cores
    )
    if copykat:
        processes = []
        num_folders = len(out_dirs)
        processes = []
        if num_folders < batch_size:
            batch_size = num_folders
        logger.info(f"Running CopyKAT on {num_folders} batches of data")

        # Run R files in parallel batches
        for num_batch in range(0, num_folders, batch_size):
            batch = out_dirs[num_batch:num_batch + batch_size]

            for name in batch:
                folder_matrix_path = os.path.join(source, name)
                r_path = os.path.join(folder_matrix_path, "copykat_R.R")
                process = subprocess.Popen(["Rscript", r_path])
                processes.append(process)

            # Wait for the current batch of processes to finish
            for process in processes:
                process.wait()

            # Clear the list of processes
            processes = []
        logger.info("Merging CopyKAT results")
        merge_copykat_result(out_dirs, source, copykat, sam_name)
    else:
        pass


def prepare_data_for_parallel_processing(
    adata: AnnData, source: str, layer: str = None, id_type: str = "S", ngene_chr: int = 5, win_size: int = 25, KS_cut: float = 0.1, sam_name: str = "",
    distance: str = "euclidean", norm_cell_names: str = "", output_seg: str = "FALSE", plot_genes: str = "TRUE", genome: str = "hg20", n_cores: int = 1
) -> List[str]:
    """
    Split the input data into smaller chunks and create subfolders for parallel processing.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix.
    source : str
        Directory where subfolders will be created.
    layer : str or None, optional
        Name of the layer in 'adata' to use as input data. Default is None.
    id_type : str, optional
        Identifier type. Default is "S".
    ngene_chr : int, optional
        Number of genes per chromosome. Default is 5.
    win_size : int, optional
        Window size. Default is 25.
    KS_cut : float, optional
        KS cut-off value. Default is 0.1.
    sam_name : str, optional
        Sample name. Default is "".
    distance : str, optional
        Distance metric. Default is "euclidean".
    norm_cell_names : str, optional
        Normalized cell names. Default is "".
    output_seg : str, optional
        Output segment. Default is "FALSE".
    plot_genes : str, optional
        Plot genes. Default is "TRUE".
    genome : str, optional
        Genome name. Default is "hg20".
    n_cores : int, optional
        Number of CPU cores to use. Default is 1.

    Returns
    -------
    out_dirs : list
        List of subfolder names.
    """
    
    adata_copy = adata.copy()
    adata_copy.X = np.expm1(adata_copy.X)
    out_dirs: List[str] = []
    # Initialize variables
    cell_split = 2000

    obs_all = list(adata_copy.obs.index)

    num_folders,random_obs = sample_data(data_list=obs_all,number_sampling=cell_split)

    for folder_num in range(num_folders):
        folder_matrix = f"filtered_feature_bc_matrix_{cell_split}_{folder_num}"
        out_dirs.append(folder_matrix)
        folder_matrix_path = os.path.join(source, folder_matrix)
        os.makedirs(folder_matrix_path, exist_ok=True)
        obs_list=list(random_obs[folder_num])
        # Save each cell list to each folder
        pd.DataFrame(index=obs_list).to_csv(
            os.path.join(folder_matrix_path, "cell_list.txt"), sep="\t"
        )

        # Create AnnData object for the current batch
        adata_batch = adata_copy[obs_list]
        adata_to_mtx_conversion(
                adata_batch,
                save_path=folder_matrix_path,
                layer=layer,
            )


        # Create R copykat script for the current subfolder
        generate_copykat_r_script(
            source, folder_matrix_path, id_type, ngene_chr, win_size, KS_cut,
            sam_name, distance, norm_cell_names, output_seg, plot_genes, genome,
            n_cores
        )

    #print("number of split folders:", len(out_dirs))
    return out_dirs


def sample_data(data_list: List, number_sampling: int) -> Tuple[int, List[List]]:
    """
    Randomly sample and split a list of data into multiple sublists.

    This function takes a list of data and divides it into smaller sublists to facilitate
    parallel processing or batching. It ensures that each sublist has roughly the same number
    of elements, with the exception of the last sublist, which may contain fewer elements if
    the total number of elements is not evenly divisible by `number_sampling`.

    Parameters:
    -----------
    data_list : List
        A list of data to be split and sampled.
    number_sampling : int
        The desired number of member in each sublists to create.

    Returns:
    --------
    Tuple[int, List[List]]
        The number of sublists created and a list of sublists containing randomly sampled elements from `data_list`.
        The length of this list is equal to `number_sampling`, with each sublist containing approximately
        len(data_list) / number_sampling elements. The last sublist may contain fewer elements if len(data_list) is
        not evenly divisible by `number_sampling`.
    """
    # calculate how many sublist will be create
    sampling_time=np.ceil(len(data_list)/number_sampling).astype(int)

    #check and handle if the data can be divide all equally or it have some remain
    if len(data_list) % number_sampling == 0:
        random_list=list(np.random.choice(data_list, replace=False, size=[sampling_time,number_sampling]))
    else:
        random_list=np.random.choice(data_list, replace=False, size=[sampling_time-1,number_sampling])
        random_list_remain=set(data_list)-set(random_list.flatten())
        random_list=list(random_list)
        random_list_remain=list(random_list_remain)
        random_list.append(random_list_remain)

    return sampling_time,random_list


def adata_to_mtx_conversion(adata: AnnData, save_path: Optional[str] = None, layer: Optional[str] = None) -> None:
    """Convert AnnData object to matrix market format.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    save_path : str or None, optional
        Directory to save the output files. Default is None. If None, data will be saved in the current working directory.
    layer : str or None, optional
        Name of the layer in `adata` to use as input data. Default is None. If None, the X layer will be used.

    Returns
    -------
    None
        Results are saved in the current working directory.

        matrix.mtx : file
            Matrix market format file containing gene expression data.
        genes.tsv : file
            Gene names file.
        barcodes.tsv : file
            Cell barcode names file.
    """

    matrix = get_adata_layer_array(adata, layer)

    # Transpose matrix and create sparse CSR matrix
    gbm = matrix.transpose()

    # Set save path if not provided
    if save_path is None:
        current_path = os.getcwd()
        folder_matrix_path = f"{current_path}/filtered_feature_bc_matrix"
        os.makedirs(folder_matrix_path, exist_ok=True)
        save_path = folder_matrix_path
    else:
        folder_matrix_path = f"{save_path}/filtered_feature_bc_matrix"
        os.makedirs(folder_matrix_path, exist_ok=True)
        save_path = folder_matrix_path

    # Convert to a sparse matrix and save
    sparse_gbm = csr_matrix(gbm)
    mmwrite(os.path.join(save_path, "matrix.mtx"), sparse_gbm)

    # Save gene and cell names
    gene = pd.DataFrame(list(adata.var.index))
    gene[1] = list(adata.var.index)
    gene.to_csv(
        os.path.join(save_path, "genes.tsv"),
        header=False,
        index=False,
        sep="\t"
    )
    adata.obs.index.to_series().to_csv(
        os.path.join(save_path, "barcodes.tsv"),
        header=False,
        index=False,
        sep="\t"
    )


def generate_copykat_r_script(
    source: str, 
    folder_matrix_path: str, 
    id_type: str, 
    ngene_chr: int, 
    win_size: int, 
    KS_cut: float, 
    sam_name: str,
    distance: str, 
    norm_cell_names: str, 
    output_seg: str, 
    plot_genes: str, 
    genome: str, 
    n_cores: int
) -> None:
    """
    Create R copykat script for a subfolder.

    Parameters
    ----------
    source : str
        Main data source directory.
    folder_matrix_path : str
        Subfolder path.
    id_type : str
        Identifier type.
    ngene_chr : int
        Number of genes per chromosome.
    win_size : int
        Window size.
    KS_cut : float
        KS cut-off value.
    sam_name : str
        Sample name.
    distance : str
        Distance metric.
    norm_cell_names : str
        Normalized cell names.
    output_seg : str
        Output segment.
    plot_genes : str
        Plot genes.
    genome : str
        Genome name.
    n_cores : int
        Number of CPU cores to use.

    Output Files
    ------------
    copykat_R.R : R script for running copykat.
    """
    # Create R script content
    R_content = f"""
    setwd("{source}")
    library(anndata)
    library(Seurat)
    data.dir = "{folder_matrix_path}"
    raw <- Read10X(data.dir = "{folder_matrix_path}", gene.column = 1)

    raw <- CreateSeuratObject(counts = raw, project = "copycat.{sam_name}", min.cells = 0, min.features = 0)
    exp.rawdata <- as.matrix(raw@assays$RNA@counts)
    dir <- paste0(data.dir, "/")
    data.dir2 <- paste0(dir, "result")
    if (!dir.exists(data.dir2)) {{
        dir.create(data.dir2)
    }} else {{
        print("")
    }}
    setwd(data.dir2)

    library(copykat)
    copykat.{sam_name} <- copykat(rawmat = exp.rawdata, id.type = "{id_type}", ngene.chr = {ngene_chr}, win.size = {win_size}, 
    KS.cut = {KS_cut}, sam.name = "{sam_name}", distance = "{distance}", norm.cell.names = "{norm_cell_names}", 
    output.seg = "{output_seg}", plot.genes = "{plot_genes}", genome = "{genome}", n.cores = {n_cores})
    """

    # Write R script content to file
    R_path = os.path.join(folder_matrix_path, "copykat_R.R")
    with open(R_path, "w") as r_file:
        r_file.write(R_content)


def merge_copykat_result(out_dirs: List[str], source: str = None, copykat: bool = False, sam_name: str = "") -> None:
    """
    Merge CopyKat results from multiple subfolders into consolidated result files.

    Parameters
    ----------
    out_dirs : List[str]
        List of subfolder names.
    source : str, optional
        Directory of the main data source. Default is None.
    copykat : bool, optional
        Flag indicating whether CopyKAT was used. Default is False.
    sam_name : str, optional
        Sample name for creating output files. Default is an empty string.

    Output Files
    ------------
    sam_name_copykat_CNA_results_all.parquet : pd.DataFrame
        Merged CopyKat CNA results.
    sam_name_copykat_CNA_raw_results_gene_by_cell_all.parquet : pd.DataFrame
        Merged CopyKat raw CNA results.
    sam_name_copykat_prediction_all.parquet : pd.DataFrame
        Merged CopyKat predictions.

    Returns
    -------
    None

    """
    #print("merging copykat results...")
    # Merge {sam_name}_copykat_CNA_results
    if source is None:
        source=os.getcwd()
    else:
        source=source
    if copykat:
 
        CNA_results = pd.read_csv(
            os.path.join(
                source, f"{out_dirs[0]}/result/{sam_name}_copykat_CNA_results.txt"
            ),
            sep="\t",
            index_col=0
        )
    else:
        CNA_results = pd.read_csv(
            os.path.join(
                source, f"{out_dirs[0]}/{sam_name}_copykat_CNA_results.txt"
            ),
            sep="\t",
            index_col=0
        )

    CNA_results = CNA_results.transpose()
    for folder_ind in range(1, len(out_dirs)):
        folder = out_dirs[folder_ind]
        CNA_results_addin = pd.read_csv(
            os.path.join(
                source, f"{folder}/result/{sam_name}_copykat_CNA_results.txt"
            ),
            sep="\t",
            index_col=0
        )
        del CNA_results_addin["chrompos"]
        del CNA_results_addin["abspos"]
        CNA_results_addin = CNA_results_addin.transpose()
        CNA_results = pd.concat([CNA_results, CNA_results_addin])
    CNA_results = CNA_results.transpose()
    CNA_results.to_parquet(
        os.path.join(source, f"{sam_name}_copykat_CNA_results_all.parquet")
    )

    # Merge {sam_name}_copykat_CNA_raw_results_gene_by_cell
    if copykat:
        CNA_results = pd.read_csv(
            os.path.join(
                source,
                f"{out_dirs[0]}/result/{sam_name}_copykat_CNA_raw_results_gene_by_cell.txt"
            ),
            sep="\t",
            index_col=0
        )
    else:
        CNA_results = pd.read_csv(
            os.path.join(
                source,
                f"{out_dirs[0]}/{sam_name}_copykat_CNA_raw_results_gene_by_cell.txt"
            ),
            sep="\t",
            index_col=0
        )
    CNA_results = CNA_results.transpose()
    for folder_ind in range(1, len(out_dirs)):
        folder = out_dirs[folder_ind]
        CNA_results_addin = pd.read_csv(
            os.path.join(
                source,
                f"{folder}/result/{sam_name}_copykat_CNA_raw_results_gene_by_cell.txt"
            ),
            sep="\t",
            index_col=0
        )
        del CNA_results_addin['start_position']
        del CNA_results_addin['end_position']
        del CNA_results_addin['band']
        del CNA_results_addin['ensembl_gene_id']
        del CNA_results_addin['chromosome_name']
        del CNA_results_addin['hgnc_symbol']
        CNA_results_addin = CNA_results_addin.transpose()
        CNA_results = pd.concat([CNA_results, CNA_results_addin])
    CNA_results = CNA_results.transpose()
    CNA_results.to_parquet(
        os.path.join(
            source, f"{sam_name}_copykat_CNA_raw_results_gene_by_cell_all.parquet"
        )
    )

    # Merge {sam_name}_copykat_prediction
    if copykat:
        CNA_results = pd.read_csv(
            os.path.join(
                source, f"{out_dirs[0]}/result/{sam_name}_copykat_prediction.txt"
            ),
            sep="\t",
            index_col=0
        )
    else:
        CNA_results = pd.read_csv(
            os.path.join(
                source, f"{out_dirs[0]}/{sam_name}_copykat_prediction.txt"
            ),
            sep="\t",
            index_col=0
        )
    for folder_ind in range(1, len(out_dirs)):
        folder = out_dirs[folder_ind]
        CNA_results_addin = pd.read_csv(
            os.path.join(
                source, f"{folder}/result/{sam_name}_copykat_prediction.txt"
            ),
            sep="\t",
            index_col=0
        )
        CNA_results = pd.concat([CNA_results, CNA_results_addin])
    CNA_results.to_parquet(
        os.path.join(source, f"{sam_name}_copykat_prediction_all.parquet")
    )
