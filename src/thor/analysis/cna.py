import glob
import logging
import os
import shutil
import subprocess
from typing import Dict, List, TextIO, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from thor.utils import get_adata_layer_array

logger = logging.getLogger(__name__)

def prepare_copykat(
    adata,
    datadir,
    cell_split=2000,
    layer=None,
    id_type="S",
    ngene_chr=5,
    win_size=25,
    KS_cut=0.1,
    sam_name="",
    distance="euclidean",
    norm_cell_names="",
    output_seg="FALSE",
    plot_genes="TRUE",
    genome="hg20",
    n_cores=1,
    seed=42,
):
    """
    Prepare copykat input

    Parameters
    ----------
    adata: AnnData object
        Gene expression data.
    data_dir: str or None
        Directory where data will be saved. Default is None. If None, data will be saved in the current working directory.
    layer: str or None
        Name of the layer in `adata` to use as input data. Default is None. If None, the X layer will be used.
    batch_size: int
        Number of subfolders to process in parallel. Default is 10.
    id_type: str
        copykat parameter `id.type`. Gene identification type. Default is "S"("s"), which refers to gene symbol. Other option is "E"("e") for Ensembl ID. 
    ngene_chr: int
        copykat parameter `ngene.chr`. Minimal number of genes per chromosome for cell filtering.. Default is 5.
    win_size: int
        copykat parameter `win.size`. Minimal window sizes for segmentation. Default is 25.
    KS_cut: float
        copykat parameter `KS.cut`. Segmentation parameter ranging from 0 to 1; larger value means looser criteria. Default is 0.1.
    sam_name: str
        copykat parameter `sam.name`. Sample name. Default is "".
    distance: str
        copykat parameter `distance`. Distance metric. Default is "euclidean". Other options are "pearson" and "spearman".
    norm_cell_names: str
        copykat parameter `norm.cell.names`. A vector of normal cell names. Default is "".
    output_seg: str 
        copykat parameter `output.seg`. Whether to output segmentation results for IGV visualization. Default is "FALSE". Other option is
        "TRUE". Note that it is a string and not a boolean.
    plot_genes: str
        copykat parameter `plot.genes`. Whether to output heatmap of CNAs with genename labels. Default is "TRUE". Other option is "FALSE". Note that it is a string and not a boolean.
    genome: str
        copykat parameter `genome`. Genome name. Default is "hg20" for human genome version 20. Other option is "mm10" for mouse genome version 10.
    n_cores: int
        copykat parameter `n.cores`. Number of CPU cores for parallel computing. Default is 1. Recommended to use 1 core if batch_size > 1.
        copykat (bool): Whether to run the CopyKat analysis. Default is True.

    Returns
    -------
    None. Results are saved in the current working directory.

    """
    if datadir is None:
        datadir = os.getcwd()
    datadir = os.path.abspath(datadir)
    
    # If main output directory exists, rename the existing one with incremental suffix
    if os.path.exists(datadir):
        original_datadir = datadir
        base_dir = os.path.dirname(original_datadir)
        base_name = os.path.basename(original_datadir)
        
        # Find the highest existing number for this base name
        counter = 1
        while os.path.exists(os.path.join(base_dir, f"{base_name}.{counter}")):
            counter += 1
        
        # Rename the existing directory to the next available number
        renamed_datadir = os.path.join(base_dir, f"{base_name}.{counter}")
        os.rename(original_datadir, renamed_datadir)
        logger.info(f"Renamed existing directory to {os.path.basename(renamed_datadir)}")
    
    source = os.path.join(datadir, "split_data_forcopykat")
    os.makedirs(source, exist_ok=True)

    obs_all = np.array(adata.obs_names)

    # deterministic shuffle
    rng = np.random.default_rng(seed)
    rng.shuffle(obs_all)

    # balanced splits
    n_chunks = int(np.ceil(len(obs_all) / cell_split))
    splits = np.array_split(obs_all, n_chunks)

    out_dirs = []
    for idx, obs_subset in enumerate(splits):
        folder = f"filtered_feature_bc_matrix_{cell_split}_{idx}"
        folder_path = os.path.join(source, folder)
        os.makedirs(folder_path, exist_ok=True)

        # save cell list
        pd.DataFrame(index=obs_subset).to_csv(
            os.path.join(folder_path, "cell_list.txt"), sep="\t"
        )

        # export to mtx
        _export_to_mtx(adata[obs_subset].copy(), folder_path, layer)

        # write R script
        _write_r_script(
            folder_path, id_type, ngene_chr, win_size, KS_cut,
            sam_name, distance, norm_cell_names, output_seg,
            plot_genes, genome, n_cores
        )

        logger.info(f"Prepared folder {folder_path}")


def _export_to_mtx(adata, save_path, layer=None):
    matrix = get_adata_layer_array(adata, layer)
    sparse_gbm = csr_matrix(matrix.T)  # genes x cells

    mmwrite(os.path.join(save_path, "matrix.mtx"), sparse_gbm)
    pd.DataFrame(adata.var_names).to_csv(
        os.path.join(save_path, "genes.tsv"), header=False, index=False, sep="\t"
    )
    pd.Series(adata.obs_names).to_csv(
        os.path.join(save_path, "barcodes.tsv"), header=False, index=False, sep="\t"
    )


def _write_r_script(folder_path, id_type, ngene_chr, win_size, KS_cut,
                    sam_name, distance, norm_cell_names, output_seg,
                    plot_genes, genome, n_cores):
    script = f"""
    library(Seurat)
    library(copykat)

    raw <- Read10X(data.dir="{folder_path}", gene.column=1)
    raw <- CreateSeuratObject(raw, project="copykat.{sam_name}", min.cells=0, min.features=0)
    exp.rawdata <- as.matrix(raw@assays$RNA$counts)

    result_dir <- file.path("{folder_path}", "result")
    if (!dir.exists(result_dir)) dir.create(result_dir)
    setwd(result_dir)

    ck <- copykat(
      rawmat=exp.rawdata,
      id.type="{id_type}",
      ngene.chr={ngene_chr},
      win.size={win_size},
      KS.cut={KS_cut},
      sam.name="{sam_name}",
      distance="{distance}",
      norm.cell.names="{norm_cell_names}",
      output.seg="{output_seg}",
      plot.genes="{plot_genes}",
      genome="{genome}",
      n.cores={n_cores}
    )
    """
    with open(os.path.join(folder_path, "copykat_R.R"), "w") as f:
        f.write(script.strip() + "\n")


def _check_copykat_installation():
    """Check if CopyKAT R package is installed."""
    check_script = """
    if (!require(copykat, quietly=TRUE)) {
        cat("ERROR: CopyKAT package not found. Please install it using:\\n")
        cat("if (!require(devtools)) install.packages('devtools')\\n")
        cat("devtools::install_github('navinlabcode/copykat')\\n")
        quit(status=1)
    }
    cat("CopyKAT package found\\n")
    """
    
    try:
        result = subprocess.run(
            ["Rscript", "-e", check_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            logger.error("CopyKAT R package is not installed.")
            logger.error("Please install it using:")
            logger.error("if (!require(devtools)) install.packages('devtools')")
            logger.error("devtools::install_github('navinlabcode/copykat')")
            raise RuntimeError("CopyKAT package not found. Please install it first.")
        else:
            logger.info("CopyKAT package verified.")
    except subprocess.TimeoutExpired:
        logger.error("Timeout checking CopyKAT installation.")
        raise RuntimeError("Timeout checking CopyKAT installation.")
    except FileNotFoundError:
        logger.error("Rscript not found. Please install R and ensure it's in your PATH.")
        raise RuntimeError("R not found. Please install R first.")


def run_copykat(datadir, batch_size=1, max_retries=2):
    """
    Run CopyKAT on the input data. Requires R and the CopyKAT package to be installed. 
    Refer to the CopyKAT documentation for details:
    `copykat <https://github.com/navinlabcode/copykat/blob/b795ff793522499f814f6ae282aad1aab790902f/R/copykat.R>`_.

    Parameters
    ----------
    datadir : str or None
        Directory where data will be saved. Default is None. If None, data will be saved in the current working directory.
    batch_size : int
        Number of samples to run simultaneously in each batch.
    max_retries : int
        Maximum number of retries for failed jobs. Default is 2.
    """
    
    # Check CopyKAT installation first
    _check_copykat_installation()
    
    if datadir is None:
        datadir = os.getcwd()
    datadir = os.path.abspath(datadir)
    source = os.path.join(datadir, "split_data_forcopykat")
    
    if not os.path.exists(source):
        logger.error(f"Source directory not found: {source}")
        raise FileNotFoundError(f"Source directory not found: {source}")
    
    out_dirs = [
        d for d in os.listdir(source)
        if d.startswith("filtered_feature_bc_matrix") and os.path.isdir(os.path.join(source, d))
    ]

    if not out_dirs:
        logger.error(f"No split folders found in {source}")
        return

    # Sort directories for consistent processing order
    out_dirs.sort()
    batch_size = min(batch_size, len(out_dirs))
    logger.info(f"Found {len(out_dirs)} split folders. Running in batches of {batch_size}.")

    failed_jobs = []
    
    for i in range(0, len(out_dirs), batch_size):
        batch = out_dirs[i:i+batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Starting batch {batch_num}: {batch}")

        # Launch all processes in the batch
        procs = []
        for folder in batch:
            r_path = os.path.join(source, folder, "copykat_R.R")
            log_path = os.path.join(source, folder, "copykat.log")
            
            if not os.path.exists(r_path):
                logger.error(f"R script not found: {r_path}")
                failed_jobs.append(folder)
                continue

            logger.info(f"Launching Rscript for {folder} (logging to {log_path})")
            try:
                with open(log_path, "w") as log:
                    proc = subprocess.Popen(
                        ["Rscript", r_path],
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                        cwd=os.path.dirname(r_path)  # Set working directory
                    )
                    procs.append((folder, proc))
            except Exception as e:
                logger.error(f"Failed to launch Rscript for {folder}: {e}")
                failed_jobs.append(folder)

        # Wait for all processes in the batch to complete
        for folder, proc in procs:
            try:
                ret = proc.wait()
                if ret == 0:
                    logger.info(f"Finished {folder} successfully")
                else:
                    logger.error(f"Rscript failed for {folder} (exit code: {ret})")
                    failed_jobs.append(folder)
            except Exception as e:
                logger.error(f"Error waiting for {folder}: {e}")
                failed_jobs.append(folder)

    # Retry failed jobs if any
    if failed_jobs and max_retries > 0:
        logger.warning(f"Retrying {len(failed_jobs)} failed jobs: {failed_jobs}")
        retry_failed = failed_jobs.copy()
        failed_jobs = []
        
        for folder in retry_failed:
            r_path = os.path.join(source, folder, "copykat_R.R")
            log_path = os.path.join(source, folder, "copykat.log")
            
            logger.info(f"Retrying {folder}")
            try:
                with open(log_path, "a") as log:  # Append to existing log
                    log.write(f"\n--- RETRY ATTEMPT ---\n")
                    proc = subprocess.Popen(
                        ["Rscript", r_path],
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                        cwd=os.path.dirname(r_path)
                    )
                    ret = proc.wait()
                    if ret == 0:
                        logger.info(f"Retry successful for {folder}")
                    else:
                        logger.error(f"Retry failed for {folder} (exit code: {ret})")
                        failed_jobs.append(folder)
            except Exception as e:
                logger.error(f"Error during retry for {folder}: {e}")
                failed_jobs.append(folder)

    # Final status report
    if failed_jobs:
        logger.error(f"Final failed jobs: {failed_jobs}")
        raise RuntimeError(f"CopyKAT failed for {len(failed_jobs)} folders: {failed_jobs}")
    else:
        logger.info("All CopyKAT jobs completed successfully!")


# -----------------------------
# Results combiner (CNA matrix)
# -----------------------------

def _list_result_files(base_dir: str, filename: str) -> List[str]:
    pattern = os.path.join(base_dir, "*", "result", filename)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return files


def _open_files(file_paths: List[str]) -> List[TextIO]:
    return [open(p, "r", encoding="utf-8") for p in file_paths]


def _read_header_columns(fh: TextIO) -> List[str]:
    header = fh.readline().rstrip("\n\r")
    if not header:
        raise ValueError("Encountered empty header line")
    return header.split("\t")


def _combine_headers(headers: List[List[str]], key_cols: int) -> List[str]:
    combined = list(headers[0])
    for hdr in headers[1:]:
        if hdr[:key_cols] != headers[0][:key_cols]:
            raise ValueError(
                "Key column names mismatch across files: "
                f"{hdr[:key_cols]} vs {headers[0][:key_cols]}"
            )
        combined.extend(hdr[key_cols:])
    return combined


def _write_line(out_fh: TextIO, fields: List[str]) -> None:
    out_fh.write("\t".join(fields) + "\n")


def combine_copykat_results(base_dir: str, filename: str, output_path: str, key_cols: int = 3) -> str:
    """
    Combine CopyKAT split result matrices by horizontal concatenation, streaming line-by-line.

    Parameters
    ----------
    base_dir : str
        Path to the parent directory that contains all split subfolders, typically
        `<datadir>/split_data_forcopykat` produced by the preparation step. This
        function searches `*/result/<filename>` under this directory.
    filename : str
        The per-split result filename to combine (e.g. `BC_copykat_CNA_results.txt`).
    output_path : str
        Destination path for the combined TSV to be written.
    key_cols : int, optional
        Number of leading key columns shared by all files (default: 3). These are
        used for alignment and are not duplicated in the combined output.

    Returns
    -------
    str
        The `output_path` where the combined file was written.
    """
    file_paths = _list_result_files(base_dir, filename)
    fhs = _open_files(file_paths)
    try:
        headers = [_read_header_columns(fh) for fh in fhs]
        combined_header = _combine_headers(headers, key_cols)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_fh:
            _write_line(out_fh, combined_header)

            line_num = 1
            while True:
                lines = [fh.readline() for fh in fhs]
                if all(l == "" for l in lines):
                    break
                if any(l == "" for l in lines):
                    raise ValueError(
                        "Files have different number of data lines; mismatch at line "
                        f"{line_num}"
                    )

                rows = [l.rstrip("\n\r").split("\t") for l in lines]
                first_keys = rows[0][:key_cols]
                for r in rows[1:]:
                    if r[:key_cols] != first_keys:
                        raise ValueError(
                            "Key columns mismatch at data line "
                            f"{line_num}: {r[:key_cols]} vs {first_keys}"
                        )

                out_fields = list(rows[0])
                for r in rows[1:]:
                    out_fields.extend(r[key_cols:])
                _write_line(out_fh, out_fields)
                line_num += 1
    finally:
        for fh in fhs:
            fh.close()

    return output_path


# ---------------------------------
# Prediction combiner (cell -> type)
# ---------------------------------

def _list_prediction_files(base_dir: str, sample_name: str) -> List[str]:
    pattern = os.path.join(base_dir, "*", "result", f"{sample_name}_copykat_prediction.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")
    return files


def _load_prediction_file(path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline()
        if header == "":
            return rows
        for line in fh:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 2:
                continue
            cell_id, label = parts[0], parts[1]
            rows.append((cell_id, label))
    return rows


def _merge_predictions(file_paths: List[str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for p in file_paths:
        for cell_id, label in _load_prediction_file(p):
            if cell_id in merged and merged[cell_id] != label:
                raise ValueError(
                    f"Conflicting labels for cell {cell_id}: {merged[cell_id]} vs {label} in {p}"
                )
            merged[cell_id] = label
    return merged


def _read_cell_list(cell_list_path: str) -> List[str]:
    cells: List[str] = []
    with open(cell_list_path, "r", encoding="utf-8") as fh:
        for line in fh:
            cell = line.rstrip("\n\r").split("\t")[0]
            if cell:
                cells.append(cell)
    return cells


def combine_copykat_predictions(
    base_dir: str,
    sample_name: str,
    out_path: str,
) -> str:
    """
    Merge per-split CopyKAT prediction files into a two-column TSV (cell_id, cell_type).

    Parameters
    ----------
    base_dir : str
        Path to the parent directory that contains all split subfolders, typically
        `<datadir>/split_data_forcopykat`.
    sample_name : str
        Sample name prefix used for prediction files (e.g. `BC` -> `BC_copykat_prediction.txt`).
    out_path : str
        Destination path for the merged two-column TSV.

    Returns
    -------
    str
        The `out_path` where the merged mapping was written.
    """
    files = _list_prediction_files(base_dir, sample_name)
    merged = _merge_predictions(files)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("cell_id\tcell_type\n")
        for cell_id, label in merged.items():
            out.write(f"{cell_id}\t{label}\n")

    return out_path

def plot_cna_heatmap_with_pred(
    tsv_path: str,
    pred_path: str,
    label_col: str | None = None,            # e.g. "cell_type"; if None, uses 2nd column
    label_colors: dict | None = None,        # e.g. {"aneuploid":"crimson","diploid":"steelblue"}
    label_order: dict | None = None,         # e.g. {"aneuploid":0, "diploid":1}
    vmin: float = -0.3,
    vmax: float = 0.3,
    cmap: str = "coolwarm",
    figsize=(16, 6),
    rasterized: bool = True,
    show_group_labels: bool = True,          # add text labels next to the left bar
    group_label_fontsize: int = 10,
    group_label_color: str = "black",
):
    # Load CNA matrix
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    # Meta columns for your schema
    meta_cols = [
        "abspos", "chromosome_name", "start_position",
        "end_position", "ensembl_gene_id", "hgnc_symbol", "band"
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    if "chromosome_name" not in df.columns:
        raise ValueError("Expected 'chromosome_name' in the header.")

    # Cell columns
    value_cols = [c for c in df.columns if c not in meta_cols]
    if not value_cols:
        raise ValueError("No cell columns found after excluding meta columns.")

    # Chromosome bar (top) with alternating black/gray
    chrom = df["chromosome_name"].astype(str).to_numpy()
    codes, chrom_levels = pd.factorize(chrom)  # codes aligned with chrom, levels are unique chroms
    black = (0.0, 0.0, 0.0)
    gray = (0.7, 0.7, 0.7)
    level_colors = np.array([black if i % 2 == 0 else gray for i in range(len(chrom_levels))], dtype=float)
    chrom_rgb = level_colors[codes]

    # Matrix: cells x genes
    mat = df[value_cols].to_numpy(dtype=float).T  # rows=cells, cols=genes

    # Load predictions (cell -> label)
    pred = pd.read_csv(pred_path, sep="\t", low_memory=False)
    if label_col is None:
        if pred.shape[1] < 2:
            raise ValueError("pred_combined.tsv must have at least two columns (cell_id, label).")
        cell_col = pred.columns[0]
        label_col = pred.columns[1]
    else:
        non_label_cols = [c for c in pred.columns if c != label_col]
        cell_col = non_label_cols[0] if non_label_cols else pred.columns[0]

    cell_to_label = dict(zip(pred[cell_col].astype(str), pred[label_col].astype(str)))

    # Defaults
    if label_colors is None:
        label_colors = {"aneuploid": "crimson", "diploid": "steelblue"}
    if label_order is None:
        label_order = {"aneuploid": 0, "diploid": 1}  # group order

    # Row labels and colors (value_cols order)
    labels = [cell_to_label.get(c, "NA") for c in value_cols]
    uniq_labels = pd.unique(labels)
    default_palette = plt.cm.Set2.colors
    auto_map = {
        lab: (label_colors.get(lab) if lab in label_colors else default_palette[i % len(default_palette)])
        for i, lab in enumerate(uniq_labels)
    }
    row_rgb = np.array([
        to_rgb(auto_map[lab]) if isinstance(auto_map[lab], str) else auto_map[lab]
        for lab in labels
    ])

    # Sort rows (cells) by annotation groups
    row_order = np.argsort([label_order.get(l, 99) for l in labels])
    value_cols = [value_cols[i] for i in row_order]
    mat = mat[row_order, :]
    labels = [labels[i] for i in row_order]
    row_rgb = row_rgb[row_order]

    # Figure grid: optionally allocate an extra left column for group text labels
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    if show_group_labels:
        gs = fig.add_gridspec(
            nrows=2, ncols=3,
            height_ratios=[1, 19], width_ratios=[2, 1, 30], hspace=0.05, wspace=0.05
        )
        ax_label = fig.add_subplot(gs[1, 0])  # text labels for aneuploid/diploid groups
        ax_left  = fig.add_subplot(gs[1, 1])  # left color strip (cell annotations)
        ax       = fig.add_subplot(gs[1, 2])  # heatmap
        ax_top   = fig.add_subplot(gs[0, 2])  # top bar over heatmap column only
    else:
        gs = fig.add_gridspec(
            nrows=2, ncols=2,
            height_ratios=[1, 19], width_ratios=[1, 30], hspace=0.05, wspace=0.05
        )
        ax_top  = fig.add_subplot(gs[0, 1])   # top bar over heatmap column
        ax_left = fig.add_subplot(gs[1, 0])   # left strip (cell annotations)
        ax      = fig.add_subplot(gs[1, 1])   # heatmap

    # Top chromosome strip (aligned to heatmap columns)
    ax_top.imshow(chrom_rgb.reshape(1, -1, 3), aspect="auto")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)

    # Left cell-type strip (aligned to heatmap rows)
    ax_left.imshow(row_rgb.reshape(-1, 1, 3), aspect="auto")
    ax_left.set_xticks([]); ax_left.set_yticks([])
    for s in ax_left.spines.values():
        s.set_visible(False)

    # Optional: add text labels (e.g., 'aneuploid', 'diploid') next to the left strip
    if show_group_labels:
        # Find contiguous groups after sorting
        groups = []  # (label, start_idx, end_idx) with end exclusive
        start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[i - 1]:
                groups.append((labels[start], start, i))
                start = i

        # Prepare label axis
        ax_label.set_axis_off()
        # Match y-limits/orientation to the strip image for proper alignment
        ax_label.set_ylim(ax_left.get_ylim())
        ax_label.set_xlim(0, 1)

        for lab, s, e in groups:
            mid = (s + e - 1) / 2.0
            ax_label.text(
                0.98, mid, str(lab),
                ha="right", va="center",
                fontsize=group_label_fontsize, color=group_label_color,
                clip_on=False,
            )

    # Heatmap
    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        rasterized=rasterized,
    )
    ax.set_xticks([]); ax.set_yticks([])

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(length=0)

    # Align top bar width to the heatmap axes (exclude colorbar and left strip)
    pos_heat = ax.get_position()
    pos_top = ax_top.get_position()
    ax_top.set_position([pos_heat.x0, pos_top.y0, pos_heat.width, pos_top.height])

    # Ensure x-limits match the heatmap image extent exactly
    x0, x1, _, _ = im.get_extent()
    ax_top.set_xlim(x0, x1)

    return fig, (ax_top, ax_left, ax)


def plot_segment_cna_heatmap_with_pred(
    tsv_path: str,
    pred_path: str,
    label_col: str | None = None,            # e.g. "cell_type"; if None, uses 2nd column
    label_colors: dict | None = None,        # e.g. {"aneuploid":"crimson","diploid":"steelblue"}
    label_order: dict | None = None,         # e.g. {"aneuploid":0, "diploid":1}
    vmin: float = -0.3,
    vmax: float = 0.3,
    cmap: str = "coolwarm",
    figsize=(16, 6),
    rasterized: bool = True,
    show_group_labels: bool = True,          # add text labels next to the left bar
    group_label_fontsize: int = 10,
    group_label_color: str = "black",
):
    """
    Plot a CNA heatmap for segment-level CNV results with a left bar reflecting aneuploid/diploid
    predictions and a top strip marking chromosomes in alternating black/gray.

    Expected TSV columns (at minimum):
      - 'chrom' (or 'chromosome_name'), 'chrompos' (optional), 'abspos' (recommended)
      - One column per cell containing segment-level CNA values
    Example cell column names might look like 'TTGACATGAACGTGGA.1.11' while predictions might use 'TTGACATGAACGTGGA-1-11'.
    This function attempts to auto-match these styles.

    Returns
    -------
    fig, (ax_top, ax_left, ax)
    """

    # Load segment-level CNA matrix
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    # Meta columns for this schema
    meta_candidates = ["chrom", "chromosome_name", "chrompos", "abspos", "start", "end", "band", "ensembl_gene_id", "hgnc_symbol"]
    meta_cols = [c for c in meta_candidates if c in df.columns]

    chrom_col = "chrom" if "chrom" in df.columns else ("chromosome_name" if "chromosome_name" in df.columns else None)
    if chrom_col is None:
        raise ValueError("Expected a chromosome column named 'chrom' or 'chromosome_name'.")

    # Cell columns
    value_cols = [c for c in df.columns if c not in meta_cols]
    if not value_cols:
        raise ValueError("No cell columns found after excluding meta columns.")

    # Top chromosome strip: alternating black/gray by chromosome blocks
    chrom = df[chrom_col].astype(str).to_numpy()
    codes, levels = pd.factorize(chrom)  # codes align to rows (segments), levels are unique chromosomes in order
    black = (0.0, 0.0, 0.0)
    gray = (0.7, 0.7, 0.7)
    level_colors = np.array([black if i % 2 == 0 else gray for i in range(len(levels))], dtype=float)
    chrom_rgb = level_colors[codes]

    # Matrix: cells x segments
    mat = df[value_cols].to_numpy(dtype=float).T  # rows=cells, cols=segments

    # Load predictions (cell -> label)
    pred = pd.read_csv(pred_path, sep="\t", low_memory=False)
    if label_col is None:
        if pred.shape[1] < 2:
            raise ValueError("pred_combined.tsv must have at least two columns (cell_id, label).")
        cell_col = pred.columns[0]
        label_col = pred.columns[1]
    else:
        non_label_cols = [c for c in pred.columns if c != label_col]
        cell_col = non_label_cols[0] if non_label_cols else pred.columns[0]

    # Optionally normalize IDs in predictions to match matrix columns
    # If no normalizer provided, we'll try exact, then flip '.'<->'-' automatically.
    pred_cells = pred[cell_col].astype(str)
    pred_labels = pred[label_col].astype(str)

    cell_to_label = dict(zip(pred_cells, pred_labels))

    # If few matches, try simple dot/dash flip heuristics as fallback without overwriting existing
    def flip_dot_dash(s: str) -> str:
        if "." in s or "-" in s:
            return s.replace(".", "#").replace("-", ".").replace("#", "-")
        return s

    # Test coverage: count initial matches
    initial_matches = sum(c in cell_to_label for c in value_cols)
    if initial_matches < max(1, int(0.1 * len(value_cols))):  # lazy heuristic
        # Build a second map with flipped styles; don't clobber existing keys
        flipped_map = {flip_dot_dash(k): v for k, v in zip(pred_cells, pred_labels)}
        for k, v in flipped_map.items():
            cell_to_label.setdefault(k, v)

    # Defaults for strip coloring and row ordering
    if label_colors is None:
        label_colors = {"aneuploid": "crimson", "diploid": "steelblue"}
    if label_order is None:
        label_order = {"aneuploid": 0, "diploid": 1}  # group order

    # Row labels and colors (value_cols order from the matrix)
    def lookup_label(cell_id: str) -> str:
        if cell_id in cell_to_label:
            return cell_to_label[cell_id]
        # Try flipped style for matrix-side if not found
        fd = flip_dot_dash(cell_id)
        return cell_to_label.get(fd, "NA")

    labels = [lookup_label(c) for c in value_cols]
    uniq_labels = pd.unique(labels)
    default_palette = plt.cm.Set2.colors
    auto_map = {
        lab: (label_colors.get(lab) if lab in label_colors else default_palette[i % len(default_palette)])
        for i, lab in enumerate(uniq_labels)
    }
    row_rgb = np.array([
        to_rgb(auto_map[lab]) if isinstance(auto_map[lab], str) else auto_map[lab]
        for lab in labels
    ])

    # Sort rows (cells) by annotation groups
    row_order = np.argsort([label_order.get(l, 99) for l in labels])
    value_cols = [value_cols[i] for i in row_order]
    mat = mat[row_order, :]
    labels = [labels[i] for i in row_order]
    row_rgb = row_rgb[row_order]

    # Figure grid: optionally allocate an extra left column for group text labels
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    if show_group_labels:
        gs = fig.add_gridspec(
            nrows=2, ncols=3,
            height_ratios=[1, 19], width_ratios=[2, 1, 30], hspace=0.05, wspace=0.05
        )
        ax_label = fig.add_subplot(gs[1, 0])  # text labels for aneuploid/diploid groups
        ax_left  = fig.add_subplot(gs[1, 1])  # left color strip (cell annotations)
        ax       = fig.add_subplot(gs[1, 2])  # heatmap
        ax_top   = fig.add_subplot(gs[0, 2])  # top bar over heatmap column only
    else:
        gs = fig.add_gridspec(
            nrows=2, ncols=2,
            height_ratios=[1, 19], width_ratios=[1, 30], hspace=0.05, wspace=0.05
        )
        ax_top  = fig.add_subplot(gs[0, 1])   # top bar over heatmap column
        ax_left = fig.add_subplot(gs[1, 0])   # left strip (cell annotations)
        ax      = fig.add_subplot(gs[1, 1])   # heatmap

    # Top chromosome strip (aligned to heatmap columns = segments)
    ax_top.imshow(chrom_rgb.reshape(1, -1, 3), aspect="auto")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)

    # Left cell-type strip (aligned to heatmap rows)
    ax_left.imshow(row_rgb.reshape(-1, 1, 3), aspect="auto")
    ax_left.set_xticks([]); ax_left.set_yticks([])
    for s in ax_left.spines.values():
        s.set_visible(False)

    # Optional: add text labels (e.g., 'aneuploid', 'diploid') next to the left strip
    if show_group_labels:
        groups = []  # (label, start_idx, end_idx) with end exclusive
        start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[i - 1]:
                groups.append((labels[start], start, i))
                start = i

        ax_label.set_axis_off()
        ax_label.set_ylim(ax_left.get_ylim())
        ax_label.set_xlim(0, 1)

        for lab, s, e in groups:
            mid = (s + e - 1) / 2.0
            ax_label.text(
                0.98, mid, str(lab),
                ha="right", va="center",
                fontsize=group_label_fontsize, color=group_label_color,
                clip_on=False,
            )

    # Heatmap
    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        rasterized=rasterized,
    )
    ax.set_xticks([]); ax.set_yticks([])

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(length=0)

    # Align top bar width to the heatmap axes (exclude colorbar and left strip)
    pos_heat = ax.get_position()
    pos_top = ax_top.get_position()
    ax_top.set_position([pos_heat.x0, pos_top.y0, pos_heat.width, pos_top.height])

    # Ensure x-limits match the heatmap image extent exactly
    x0, x1, _, _ = im.get_extent()
    ax_top.set_xlim(x0, x1)

    return fig, (ax_top, ax_left, ax)
