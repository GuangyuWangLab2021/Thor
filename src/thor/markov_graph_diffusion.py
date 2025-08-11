import os
import warnings

warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger(__name__)

import multiprocessing as mp
from collections import OrderedDict

import numpy as np
import pandas as pd
from qnorm import quantile_normalize
from scipy.sparse import csr_matrix, issparse, save_npz
from tqdm import tqdm

# local package
from thor.graph_construction import (add_snn_to_adata,
                                     add_transition_matrix_to_adata)
from thor.utils import (estimate_spot_from_cells, get_2nd_cell_neighbours,
                        get_2nd_spot_neighbours, get_adata_layer_array,
                        get_stats_in_neighboring_cells, inverse_robustnorm,
                        robustnorm, arr_to_csr, knn_smooth, median_num_cells_mapped_to_spots)
from thor.VAE import IdentityGenerator

# global variables
MANY_GENES = 200
MIN_NUM_CELLS = 10

DATA_MATRIX_SAVE_DTYPE = np.float32


def markov_graph_diffusion_initialize(
    adata_input,
    n_neighbors=5,
    conn_key="snn",
    geom_morph_ratio=0.5,
    geom_constraint=0,
    obs_keys=None,
    reduced_dimension_transcriptome_obsm_key="X_pca",
    reduced_dimension_transcriptome_obsm_dims=2,
    adjust_cell_network_by_transcriptome_scale=1,
    snn_threshold=0.1,
    conn_csr_matrix=None,
    smoothing_scale=0.8,
    inflation_percentage=None,
    node_features_obs_list=["spot_heterogeneity"],
    preferential_flow=True,
    weigh_cells=True,
    balance_cell_quality=False,
    bcq_IQR=(0.15, 0.85),
):
    """ Initialize the adata object for markov graph diffusion.

    Parameters
    ----------
    adata_input : :class:`anndata.AnnData`
        Annotated data matrix. 
    n_neighbors : :py:class:`int`, optional
        The number of neighbors for the Nearest Neighbors graph construction. Defaults to 5.
    conn_key : :py:class:`str`, optional
        The prefix of the key for storing the cell-cell adjacency matrix and transition matrix in adata.obsp. Defaults to "snn".
    geom_morph_ratio : :py:class:`float`, optional
        The ratio of geometric distance and morphological distance for the Nearest Neighbors graph construction. Defaults to 1. Meaningful range
        is between 0 (neglect geometric constraint) to infinity (depends only on physical locations).
    geom_constraint : :py:class:`float`, optional
        The geometric constraint for the Nearest Neighbors graph construction. Defaults to 0. Meaningful range is between 0 (neglect geometric constraint) to
        infinity (depends only on physical locations).
    obs_keys : :py:class:`list`, optional
        The list of keys in adata.obs for morphological features to be used for the Nearest Neighbors graph construction. Defaults to :py:obj:`None`,
        and adata.uns["cell_image_props"] will be used.
    reduced_dimension_transcriptome_obsm_key : :py:class:`str`, optional
        The key in adata.obsm for low-dimension embedding of the transcriptomic data to be used for the Nearest Neighbors graph construction.
        Defaults to "X_pca".
    reduced_dimension_transcriptome_obsm_dims : :py:class:`int`, optional
        The number of dimensions for the low-dimension embedding of the transcriptomic data to be used for the Nearest Neighbors graph construction.
        Defaults to 2.
    adjust_cell_network_by_transcriptome_scale : :py:class:`float`, optional
        The scale of the transcriptome-based cell-cell adjacency matrix to be used for the Nearest Neighbors graph construction. Defaults to
        0 (not used).
    snn_threshold : :py:class:`float`, optional
        The threshold for the cell-cell adjacency matrix to be used for the Nearest Neighbors graph construction. Defaults to 0.1.
    conn_csr_matrix : :class:`scipy.sparse.csr_matrix` or :py:class:`str`, optional
        The cell-cell adjacency matrix in compressed sparse row format. If set to "force", the function will force to recalculate the
        connectivities. Defaults to :py:obj:`None`.
    smoothing_scale : :py:class:`float`, optional
        The extent of smoothing for the transition matrix. Defaults to 0.8.
    inflation_percentage : :py:class:`float`, optional
        The percentage of inflation for the reverse transition matrix. Defaults to :py:obj:`None` (no inflation).
    node_features_obs_list : :py:class:`list`, optional
        The list of keys in adata.obs for node features to be used for the transition matrix. Defaults to ["spot_heterogeneity"].
    preferential_flow : :py:class:`bool`, optional
        Whether to use preferential flow for the transition matrix. Defaults to :py:obj:`True`.
    weigh_cells : :py:class:`bool`, optional
        Whether to weigh cells for the transition matrix (diagonal). Defaults to :py:obj:`True`.
    balance_cell_quality : :py:class:`bool`, optional
        Whether to balance cell quality for the transition matrix. Defaults to :py:obj:`False`.
    bcq_IQR : :py:class:`tuple`, optional
        The interquartile range for the cell quality balancing. Defaults to (0.15, 0.85).
    """

    obsm_key = reduced_dimension_transcriptome_obsm_key
    if adjust_cell_network_by_transcriptome_scale > 0:
        if obsm_key not in adata_input.obsm:
            logger.error(
                f"{obsm_key} not in adata obsm. Please generate it before running markov graphical diffusion."
            )
            return None
        else:
            adata_input.obsm[obsm_key] = adata_input.obsm[obsm_key][:, :reduced_dimension_transcriptome_obsm_dims].copy()
            
    if not set(node_features_obs_list).issubset(adata_input.obs.columns):
        logger.warning(
            f"{node_features_obs_list} not in adata.obs. Please generate it before running markov graphical diffusion."
        )
        logger.info("Now not distinguishing cells")
        weigh_cells = False

    if obs_keys is None:
        obs_keys = adata_input.uns["cell_image_props"].tolist()

    # Other specified network connectivities.
    if issparse(conn_csr_matrix):
        logger.info("Using provided connectivities")
        adata_input.obsp[f"{conn_key}_connectivities"] = conn_csr_matrix
        logger.info(f"{conn_key} added to adata.obsp")
        COMPUTE_SNN = False

    # yapf: disable
    elif isinstance(conn_csr_matrix, str) and conn_csr_matrix.lower() == "force":
        logger.info("Forcing to recalculate the connectivities.")
        COMPUTE_SNN = True
    else:
        CONN_EXISTS_IN_ADATA = (
            f"{conn_key}_connectivities" in adata_input.obsp
            and conn_key in adata_input.uns
            and adata_input.uns[conn_key]["neighbor_num"] == {"snn_connectivities": n_neighbors}
            and adata_input.uns[conn_key]["geom_morph_ratio"] == geom_morph_ratio
            and adata_input.uns[conn_key]["geom_constraint"] == geom_constraint
            and np.array_equal(adata_input.uns[conn_key]["features"], obs_keys)
            ) and (
                    (adata_input.uns[conn_key]["theta"] == 0 and adjust_cell_network_by_transcriptome_scale == 0)
                    or (
                        adata_input.uns[conn_key]["theta"] == adjust_cell_network_by_transcriptome_scale
                        and adata_input.uns[conn_key]["transcriptome"] == obsm_key
                        )
                    )

        # check whether desired SNN exists in adata
        if CONN_EXISTS_IN_ADATA:
            logger.info("SNN already in adata! Skip constructing SNN")
            COMPUTE_SNN = False
        else:
            COMPUTE_SNN = True

    # yapf: enable
    if COMPUTE_SNN:
        add_snn_to_adata(
            adata_input,
            obs_keys=obs_keys,
            neighbor_num=n_neighbors,
            geom_morph_ratio=geom_morph_ratio,
            geom_constraint=geom_constraint,
            theta=adjust_cell_network_by_transcriptome_scale,
            conn_key=conn_key,
            reduced_dimension_transcriptome_obsm_key=obsm_key,
        )
    # Done with SNN calculation.

    add_transition_matrix_to_adata(
        adata_input,
        conn_key=conn_key,
        diffusion_extent=smoothing_scale,
        weigh_cells=weigh_cells,
        preferential_flow=preferential_flow,
        node_features_obs_list=node_features_obs_list,
        inflation_percentage=inflation_percentage,
        balance_cell_quality=balance_cell_quality,
        bcq_IQR=bcq_IQR,
        snn_threshold=snn_threshold,
    )


def stabilize_expression(arr, ref, minmax):
    """Stabilize the predicted gene expression by quantile normalizing it in all cells.
    """

    n_cells, n_genes = arr.shape

    # Predicted gene expression and spot gene expression are normalized to the same range using robust normalization.
    arr_trans, ql, qu = robustnorm(arr, qr=minmax)
    ref_trans, ql_tar, qu_tar = robustnorm(ref, qr=minmax)

    for idx in range(n_genes):
        stacked_arr = np.vstack((arr_trans[:, idx], ref_trans[:, idx])).T
        arr_trans[:, idx] = quantile_normalize(stacked_arr)[:, 0]

    # rescale it back
    arr_qnormed = inverse_robustnorm(arr_trans, (ql_tar, qu_tar))

    return arr_qnormed


def markov_graph_diffusion_preserve_mean(
    S0,
    T_csr,
    n_iter=10,
    regulate_expression_mean=True,
    cellxspot=None,
    temp_dir=None,
    temp_files_prefix=None,
    show_progress=False
):
    n_obs, n_var = S0.shape

    it = tqdm(range(n_iter)) if show_progress else range(n_iter) 

    if n_obs <= MIN_NUM_CELLS:
        bottop = (0, 1)
    else:
        bottop = (0.5 * MIN_NUM_CELLS / n_obs, 1 - 0.5 * MIN_NUM_CELLS / n_obs)

    S_curr = S0

    for i in it:
        S_next = T_csr * S_curr
        #assert(S_next is not S_curr)
        S_next = stabilize_expression(S_next, S_curr, bottop)

        if regulate_expression_mean:
            if i == 0:
                S_ref = S_next
            if i + 1 == n_iter:
                logger.info("Rescaling to mean")
                #S_ref = S0
                S_next = rescale_expression_mean(S_next, S_ref, cellxspot)

        S_curr = S_next

        np.save(
            os.path.join(temp_dir, f"{temp_files_prefix}_{i+1}.npy"), S_curr
        )


def markov_graph_diffusion_paralell(
    S0, T_csr, n_iter, temp_dir, temp_files_prefix, regulate_expression_mean=True, cellxspot=None, n_jobs=8
):
    """ Perform Markov Chain Monte Carlo (MCMC) diffusion on input data with parallel implementation.

    Parameters
    ----------
    S0 : numpy.ndarray
        The input data array with dimensions (obs, vars) representing gene expression.
    T_csr : scipy.sparse.csr_matrix
        The transition matrix in compressed sparse row format.
    n_iter : int
        The number of iterations for the diffusion process.
    temp_dir : str
        The directory to store temporary files during the computation.
    temp_files_prefix : str
        Prefix for the temporary files to be generated.
    cellxspot : scipy.sparse.csr_matrix, optional
        The cell-spot mapping matrix in compressed sparse row format. Defaults to None.
    n_jobs : int, optional
        Number of parallel jobs to run. Defaults to 8.

    Returns
    -------
    None

    Notes
    -----
    If n_jobs is set to 1, the function invokes the diffusion process without multiprocessing.

    If n_jobs is greater than 1, the input data (S0) is divided into multiple pieces (vars-wise), and parallel
    diffusion processes are run using multiprocessing. After the diffusion processes are completed,
    the result from each process is combined and saved into temporary files. These files can then be
    used to obtain the combined diffusion result for each iteration.
    """

    if n_iter == 0:
        return

    if S0.shape[1] <= MANY_GENES or n_jobs == 1:
        # Do not invoke multiprocessing
        return markov_graph_diffusion_preserve_mean(
            S0,
            T_csr,
            n_iter=n_iter,
            regulate_expression_mean=regulate_expression_mean,
            cellxspot=cellxspot,
            temp_dir=temp_dir,
            temp_files_prefix=temp_files_prefix,
            show_progress=True
        )

    pieces = np.array_split(S0, n_jobs, axis=1)

    tasks = [
        (
            pieces[pile], T_csr, n_iter, regulate_expression_mean, cellxspot, temp_dir,
            f"{temp_files_prefix}_pile{pile+1}", False
        ) for pile in range(n_jobs)
    ]
    
    logger.info(f"Creating {n_jobs} piles of genes for parallel processing")

    with mp.Pool(n_jobs) as pool:
        pool.starmap(
            markov_graph_diffusion_preserve_mean,
            tasks,
        )

    for i in range(n_iter):
        chain_combined_iter = np.hstack(
            [
                np.load(
                    os.path.join(
                        temp_dir, f"{temp_files_prefix}_pile{pile+1}_{i+1}.npy"
                    )
                ) for pile in range(n_jobs)
            ]
        )
        np.save(
            os.path.join(temp_dir, f"{temp_files_prefix}_{i+1}.npy"),
            chain_combined_iter
        )

        for pile in range(n_jobs):
            os.remove(
                os.path.join(
                    temp_dir, f"{temp_files_prefix}_pile{pile+1}_{i+1}.npy"
                )
            )


def rescale_expression_mean(x, x_ref, cellxspot, scale=0.8):
    """Rescale the predicted gene expression to the nearby spot expression mean.

    Parameters
    ----------
    x : numpy.ndarray
        The predicted gene expression by diffusion.
    x_ref : numpy.ndarray
        The reference gene expression (spot-level) to be used for rescaling.
    cellxspot : scipy.sparse.csr_matrix
        The cell-spot mapping matrix in compressed sparse row format.
    scale : float, optional
        The scale of the variance of the cells expression. Defaults to 0.8.

    Returns
    -------
    y : numpy.ndarray
        The rescaled gene expression.
    """

    # get spot mean values of current gene expression
    Xs = estimate_spot_from_cells(
        x, cellxspot, mapping_method="mean"
    ) 
    Xc = cellxspot * Xs  # np.array

    #y = scale * (x - Xc) + x_ref
    y = x - scale * (Xc - x_ref)

    return y


def recover_expression_variance(x, x_std, scale=1):
    """ Sample gene expression according to the variance of the nearby cells expression using the normal distribution.

    Parameters
    ----------
    x : numpy.ndarray
        The predicted gene expression by diffusion.
    x_std : numpy.ndarray
        The standard deviation of the nearby cells expression.
    scale : float, optional
        The scale of the variance of the cells expression. Defaults to 1.

    Returns
    -------
    y : numpy.ndarray
        The sampled gene expression.
    """
    # xi --> xi + ei (for each gene i)
    # ei ~ N(0, x_var[i])

    dx_rand = np.random.normal(0, 1, size=x.shape)
    y = x + x_std * dx_rand * scale

    return y


def recover_expression_variance_smooth(x, x_std, scale=1, smooth_steps=10):
    """ Sample gene expression according to the variance of the nearby cells expression using the normal distribution. The result is smoothed by averaging over a number of steps.

    Parameters
    ----------
    x : numpy.ndarray
        The predicted gene expression by diffusion.
    x_std : numpy.ndarray
        The standard deviation of the nearby cells expression.
    scale : float, optional
        The scale of the variance of the cells expression. Defaults to 1.
    smooth_steps : int, optional
        The number of steps to smooth the sampled gene expression. Defaults to 10.

    Returns
    -------
    y : numpy.ndarray
        The smoothed sampled gene expression.
    """

    smooth_steps = int(smooth_steps)

    y = np.zeros_like(x)
    for i in range(smooth_steps):
        y += recover_expression_variance(x, x_std, scale=scale)

    y = y / smooth_steps

    return y


def sample_gene_expression(expr, nn, cellxspot, sample_scale=1, sample_steps=1):
    """ Sample gene expression according to mean of fineST prediction and the variance of the nearby cells expression

    Parameters
    ----------
    expr : numpy.ndarray
        The predicted gene expression by diffusion.
    nn : scipy.sparse.csr_matrix
        The cell-cell adjacency matrix in compressed sparse row format.
    cellxspot : scipy.sparse.csr_matrix
        The cell-spot mapping matrix in compressed sparse row format.
    sample_scale : float, optional
        The scale of the sampled gene expression fluctuation. Defaults to 1.
    sample_steps : int, optional
        The number of steps to smooth the sampled gene expression. Defaults to 1.
    """

    x_mean, x_std = get_stats_in_neighboring_cells(expr, nn)
    expr = recover_expression_variance_smooth(
        x_mean, x_std, scale=sample_scale, smooth_steps=sample_steps
    )

    return expr


def rescale_predicted_gene_expression(
    mean_array=None,
    ref_array=None,
    regulate_expression_mean=True,
    stochastic_expression=True,
    sample_predicted_expression_fluctuation_scale=1,
    smooth_predicted_expression_steps=1,
    cellxspot=None,
    neighbour_graph=None,
):
    out_layer_pos = []
    out_layer_pos.append("resc") if regulate_expression_mean else None
    out_layer_pos.append(
        f"samp{smooth_predicted_expression_steps}"
    ) if stochastic_expression else None

    out_layer_pos = "_".join(out_layer_pos)

    expr_adj = mean_array.copy()
    if regulate_expression_mean:
        expr_adj = rescale_expression_mean(expr_adj, ref_array, cellxspot)


    if stochastic_expression:
        expr_adj = sample_gene_expression(
            expr_adj,
            neighbour_graph,
            cellxspot,
            sample_scale=sample_predicted_expression_fluctuation_scale,
            sample_steps=smooth_predicted_expression_steps
        )

    expr_adj[expr_adj < 0] = 0

    return out_layer_pos, expr_adj


def get_frames_to_write(len_chain, write_freq):
    frames = np.arange(len_chain)
    frames_to_write = frames[::write_freq] + write_freq - 1

    return frames_to_write[frames_to_write < len_chain]


def join_arrays(
    expression_arrays,
    array_shape,
    vae_genes_indices=None,
    other_genes_indices=None,
):
    """
    Note all the indices are based on the original adata.var (all genes in the input adata)
    In mixed mode, the chain_g_iter will overwrite the chain_v_iter for the genes not used for *reduced mode*.

    Currently only support two keys in the expression_arrays dictionary, "gene" and "vae".
    """
    joint_expression_array = np.zeros(array_shape)

    try:    
        chain_v_iter = expression_arrays["vae"]
        joint_expression_array[:, vae_genes_indices] = chain_v_iter
    except:
        pass

    try:
        chain_g_iter = expression_arrays["gene"]
        joint_expression_array[:, other_genes_indices] = chain_g_iter
    except:
        pass

    return joint_expression_array


def read_single_gene_expression_array(
    temp_dir,
    file_name,
    gen_module,
):
    arr = np.load(os.path.join(temp_dir, file_name))
    return gen_module.decode(arr)


def read_mix_gene_expression_arrays(
    temp_dir,
    file_names,
    gen_modules,
):
    expression_arrays = {}
    for mode, file_name in file_names.items():
        gene_module = gen_modules[mode]
        expression_arrays[mode] = read_single_gene_expression_array(temp_dir, file_name, gene_module)

    return expression_arrays


def prepare_input_array(adata, input_layer=None, gen_module=None):
    # Genes used to train the VAE model
    vae_genes_indices = np.where(adata.var.used_for_vae)[0]

    # Genes to predict in *reduce mode*
    reduced_genes_indices = np.where(adata.var.used_for_reduced)[0]

    # Genes to predict in *gene mode*
    other_genes_indices = np.where(
        np.logical_and(
            adata.var.used_for_reduced == False,
            adata.var.used_for_prediction == True
        )
    )[0]

    S0_dict = OrderedDict()
    if len(reduced_genes_indices) < 1:
        X_g = get_adata_layer_array(
            adata[:, other_genes_indices], layer_key=input_layer
        )
        S0_dict['gene'] = X_g
    elif len(other_genes_indices) < 1:
        X_v = get_adata_layer_array(
            adata[:, vae_genes_indices], layer_key=input_layer
        )
        S0_dict['vae'] = gen_module.encode(X_v)
    elif len(reduced_genes_indices) > 0 and len(other_genes_indices) > 0:
        X_g = get_adata_layer_array(
            adata[:, other_genes_indices], layer_key=input_layer
        )
        X_v = get_adata_layer_array(
            adata[:, vae_genes_indices], layer_key=input_layer
        )
        S0_dict.update({'gene': X_g, 'vae': gen_module.encode(X_v)})
    else:
        logger.error(
            "No genes for prediction. Setting is wrong before running!"
        )
        raise ValueError
    return S0_dict, vae_genes_indices, reduced_genes_indices, other_genes_indices


def decide_stochastic_sampling(
    adata, conn_key, stochastic_expression_neighbors_level
):
    nn = adata.obsp[f"{conn_key}_connectivities"]
    cellxspot = arr_to_csr(pd.get_dummies(adata.obs["spot_barcodes"]), dtype=np.int8)

    if stochastic_expression_neighbors_level == "spot":
        nn2 = get_2nd_spot_neighbours(nn, cellxspot)
        stochastic_expression = True
    elif stochastic_expression_neighbors_level == "cell":
        nn2 = get_2nd_cell_neighbours(nn)
        stochastic_expression = True
    else:
        nn2 = None
        stochastic_expression = False
    return nn2, cellxspot, stochastic_expression


def estimate_expression_markov_graph_diffusion(
    adata,
    conn_key="snn",
    n_iter=20,
    input_layer=None,
    is_rawCount=True,
    stochastic_expression_neighbors_level="spot",
    regulate_expression_mean=False,
    smooth_predicted_expression_steps=1,
    sample_predicted_expression_fluctuation_scale=1,
    n_jobs=8,
    out_prefix="y",
    write_freq=10,
    temp_dir="",
    save_dir="",
    gen_module=None
):
    """ Estimate gene expression using Markov graph diffusion.
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. It should have the following fields:
        - adata.obsp[f"{conn_key}_connectivities"]: cell-cell adjacency matrix in csr_matrix format.
        - adata.obsp[f"{conn_key}_transition_matrix"]: cell-cell transition matrix in csr_matrix format.
        - adata.obsm["X_pca"]: cell-level gene expression in numpy.ndarray format. Optional if `adata.var.used_for_vae` is specified.
        - adata.var.used_for_vae: boolean array indicating whether a gene is used for VAE training.
        - adata.var.used_for_reduced: boolean array indicating whether a gene is used for dimension-reduced mode.
    conn_key : str, optional
        The prefix of the key for storing the cell-cell adjacency matrix and transition matrix in adata.obsp. Defaults to "snn".
    n_iter : int, optional
        The number of iterations for the diffusion process. Defaults to 20.
    input_layer : str, optional
        The name of the layer in adata.layers to be used for the diffusion process. Defaults to None (will use adata.X).
    is_rawCount : bool, optional
        Whether the input data is raw count. Defaults to True.
    stochastic_expression_neighbors_level : str, optional
        Level of sampling the predicted gene expression using the variance of the nearby cells expression. Valid options are "spot", "cell", and
        None. Defaults to "spot". If set to None, the predicted gene expression will not be sampled.
    regulate_expression_mean : bool, optional
        Whether to rescale the predicted gene expression to the nearby spot expression mean. Defaults to False.
    smooth_predicted_expression_steps : int, optional
        The number of steps to smooth the sampled gene expression. Defaults to 1.
    sample_predicted_expression_fluctuation_scale : float, optional
        The scale of the sampled gene expression fluctuation. Defaults to 1.
    n_jobs : int, optional
        Number of parallel jobs to run. Defaults to 8.
    out_prefix : str, optional
        The prefix of the key for storing the predicted gene expression in adata.layers. Defaults to "y".
    write_freq : int, optional
        The frequency of writing the predicted gene expression to adata.layers. Defaults to 10.
    temp_dir : str, optional
        The directory to store temporary files during the computation. Defaults to None (will use the current working directory).
    gen_module : tensorflow module, optional
        The generative module used for encoding and decoding. Defaults to None (provide when needed).
    
    Returns
    -------
    adata_out : AnnData
        Annotated data matrix with the following layers added:
        - predicted gene expression without rescaling.
        - predicted gene expression rescaled to the nearby spot expression mean. (optionally)
        - predicted gene expression sampled using the variance of the nearby cells expression. (optionally)
        - predicted gene expression rescaled to the nearby spot expression mean and then sampled using the variance of the nearby cells expression. (optionally)

    """
    T_csr = adata.obsp[f"{conn_key}_transition_matrix"]

    logger.info("fineST estimation starts.")

    S0_dict, vae_genes_indices, reduced_genes_indices, other_genes_indices = prepare_input_array(
        adata, input_layer=input_layer, gen_module=gen_module
    )

    predict_IO_mode(
        S0_dict,
        T_csr,
        adata,
        conn_key=conn_key,
        vae_genes_indices=vae_genes_indices,
        other_genes_indices=other_genes_indices,
        stochastic_expression_neighbors_level=
        stochastic_expression_neighbors_level,
        sample_predicted_expression_fluctuation_scale=
        sample_predicted_expression_fluctuation_scale,
        smooth_predicted_expression_steps=smooth_predicted_expression_steps,
        regulate_expression_mean=regulate_expression_mean,
        n_iter=n_iter,
        gen_module=gen_module,
        write_freq=write_freq,
        is_rawCount=is_rawCount,
        temp_dir=temp_dir,
        save_dir=save_dir,
        out_prefix=out_prefix,
        n_jobs=n_jobs
    )

    logger.info("fineST estimation finished. \n\n")


def predict_IO_mode(
    S0_dict,
    T_csr,
    adata,
    conn_key=None,
    vae_genes_indices=None,
    other_genes_indices=None,
    stochastic_expression_neighbors_level="spot",
    sample_predicted_expression_fluctuation_scale=1,
    smooth_predicted_expression_steps=1,
    regulate_expression_mean=False,
    n_iter=10,
    gen_module=None,
    write_freq=10,
    is_rawCount=True,
    temp_dir=None,
    save_dir=None,
    out_prefix=None,
    n_jobs=8
):
    """
    Note
    ----
        In IO mode, the diffusion process is run in parallel using multiprocessing. 
        It is memory efficient but slower than RAM mode as it relies on disk I/O.
    """

    nn2, cellxspot, stochastic_expression = decide_stochastic_sampling(
        adata, conn_key, stochastic_expression_neighbors_level
    )
    stochastic_expression = False if smooth_predicted_expression_steps == 0 else stochastic_expression

    # Zero-indexed iterations for saving the chain data
    iters_to_write = get_frames_to_write(n_iter, write_freq)

    for mode, S0 in S0_dict.items():
        
        markov_graph_diffusion_paralell(
            S0,
            T_csr,
            n_iter,
            temp_dir,
            f"{out_prefix}_{mode}_chain",
            regulate_expression_mean=regulate_expression_mean,
            cellxspot=cellxspot,
            n_jobs=n_jobs
        )

    # post processing

    # PZhang note (10/17/2023), 
    # Turn off the regulate_expression_mean for now. It was done during the diffusion.
    # The predicted gene expression is already rescaled to the nearby spot expression mean.

    logger.info(f"Saving gene expression matrices.")
    regulate_expression_mean = False

    gen_modules = {
        mode: gen_module for mode in S0_dict.keys()
    }
    gen_modules.update({"gene": IdentityGenerator()})
    # The reference layer is the spot-level gene expression (which should be always stored in .X)
    ref_array = get_adata_layer_array(
        adata, layer_key=None
    )

    data_shape = ref_array.shape
    prediction_genes_indices = np.where(adata.var.used_for_prediction)[0]

    _n_neigh = 4 * median_num_cells_mapped_to_spots(adata.obs, spot_identifier='spot_barcodes')
    input_expression_smooth = OrderedDict()
    for mode, arr in S0_dict.items():
        arr = gen_modules[mode].decode(arr)
        input_expression_smooth[mode] = knn_smooth(adata.obsm['spatial'], arr, _n_neigh)
    ref_array = join_arrays(input_expression_smooth, data_shape, vae_genes_indices=vae_genes_indices, other_genes_indices=other_genes_indices)
    ref_array = ref_array[:, prediction_genes_indices]

    for _iter in iters_to_write:
        logger.debug(f"iteration {_iter+1}")
        file_names = {
            mode: f"{out_prefix}_{mode}_chain_{_iter+1}.npy" for mode in S0_dict.keys()
        }

        expression_arrays_dict = read_mix_gene_expression_arrays(
            temp_dir, file_names, gen_modules
        )

        joint_expression_array = join_arrays(
            expression_arrays_dict,
            data_shape,
            vae_genes_indices=vae_genes_indices,
            other_genes_indices=other_genes_indices
        )


        joint_expression_array = joint_expression_array[:, prediction_genes_indices]
        data_type = int if is_rawCount else DATA_MATRIX_SAVE_DTYPE
        save_npz(
            os.path.join(save_dir, f"{out_prefix}_{_iter+1}.npz"), arr_to_csr(joint_expression_array, dtype=data_type)
        )

        if not regulate_expression_mean and not stochastic_expression:
            continue

        out_layer_pos, out_array = rescale_predicted_gene_expression(
            mean_array=joint_expression_array,
            ref_array=ref_array,
            regulate_expression_mean=regulate_expression_mean,
            stochastic_expression=stochastic_expression,
            sample_predicted_expression_fluctuation_scale=
            sample_predicted_expression_fluctuation_scale,
            smooth_predicted_expression_steps=smooth_predicted_expression_steps,
            cellxspot=cellxspot,
            neighbour_graph=nn2
        )
        save_npz(
            os.path.join(
                save_dir, f"{out_prefix}_{_iter+1}_{out_layer_pos}.npz"
            ), arr_to_csr(out_array, dtype=data_type)
        )
