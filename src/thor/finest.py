import glob
import json
import os
import shutil
import warnings
import logging
import copy
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler

from thor.pp import WholeSlideImage, Spatial
from thor.markov_graph_diffusion import (
    estimate_expression_markov_graph_diffusion,
    markov_graph_diffusion_initialize
)
from thor.plotting.graph import plot_cell_graph
from thor.utils import (
    generate_cell_adata, get_adata_layer_array, get_spot_heterogeneity_cv,
    var_cos
)
from thor.VAE import VAE, IdentityGenerator, train_vae

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
sc.settings.verbosity = "error"


MANY_GENES = 200

default_run_params = dict(
    initialize=True,
    burn_in_steps=5,
    layer=None,
    is_rawCount=False,
    regulate_expression_mean=False,
    stochastic_expression_neighbors_level="spot",
    n_iter=20,
    conn_key="snn",
    write_freq=10,
    out_prefix="y",
    sample_predicted_expression_fluctuation_scale=1,
    smooth_predicted_expression_steps=0,
    save_chain=False,
    n_jobs=1,
)

default_graph_params = dict(
    n_neighbors=5,
    conn_key="snn",
    obs_keys=None,
    reduced_dimension_transcriptome_obsm_key="X_pca",
    reduced_dimension_transcriptome_obsm_dims=2,
    geom_morph_ratio=1,
    geom_constraint=0,
    adjust_cell_network_by_transcriptome_scale=0,
    snn_threshold=0.1,
    smoothing_scale=0.8,
    conn_csr_matrix=None,
    inflation_percentage=None,
    node_features_obs_list=["spot_heterogeneity"],
    preferential_flow=True,
    weigh_cells=True,
    balance_cell_quality=False,
    bcq_IQR=(0.15, 0.85),
)


class fineST:
    """Class for in silico cell gene expression inference

    Parameters
    ----------
    image_path : :py:class:`str`
        Path to the whole slide image which is aligned to the spatial transcriptomics.
    name : :py:class:`str`
        Name of the sample.
    spot_adata_path : :py:class:`str`, optional
        Path to the processed spatial transcriptomics data (e.g., from the Visium sequencing data) in the ``.h5ad`` format. 

        The expression array (``.X``) and spots coordinates (``.obsm["spatial"]``) are required. Expecting that ``.X`` is lognormalized.

        One of ``spot_adata_path`` or ``st_dir`` is needed. If ``spot_adata_path`` is provided, ``st_dir`` will be neglected.
    st_dir : :py:class:`str`, optional
        Directory to the SpaceRanger output directory, where the count matrix and spatial directory are located.
    cell_features_csv_path : :py:class:`str`, optional
        Path to the CSV file that stores the cell features. The first two columns are expected (exactly) to be the nuclei positions "x" and "y".
    cell_features_list : :py:class:`list` or :py:obj:`None`, optional
        List of features to be used for generating the cell-cell graph. 
            The first two are expected (exactly) to be the nuclei positions "x" and "y". 

            By default, if no external features are provided, those features ``["x", "y", "mean_gray", "std_gray", "entropy_img", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]`` are used.
    genes_path : :py:class:`str`, optional
        Path to the file that contains a headless one column of the genes to be included.
            The gene names or gene IDs should be consistent with the ``self.adata.var_names``.
            If :py:obj:`None`, the genes will be highly variable genes or set further by ``self.set_genes_for_prediction``.
    save_dir : :py:class:`str` or :py:obj:`None`, optional
        Path to the directory of saving fineST prediction results.
    recipe : :py:class:`str`, optional
        Specifies the mode for predicting the gene expression. Valid options are: ``("gene", "reduced", "mix")``.
    **kwargs : :py:class:`dict`, optional
        Keyword arguments for any additional attributes to be set for the class. This allows future loading of the saved json file to create a new instance of the class.
    """

    def __init__(
        self,
        image_path,
        name,
        spot_adata_path=None,
        st_dir=None,
        cell_features_list=None,
        cell_features_csv_path=None,
        genes_path=None,
        save_dir=None,
        recipe="gene",
        **kwargs
    ):
        self.name = name
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), f"fineST_{name}")

        save_dir = os.path.abspath(save_dir)
        image_path = os.path.abspath(image_path)

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.image_path = image_path

        assert (spot_adata_path is not None) or (
            st_dir is not None
        ), "Spot-level spatial transcriptomics data are required!"

        if spot_adata_path is not None:
            self.spot_adata_path = os.path.abspath(spot_adata_path)
        else:
            # This means st_dir is provided
            # In this case we process the transcriptome data from the SpaceRanger output directory and output raw counts
            # In practice, one is encouraged to provide the processed spot adata (e.g., from the Visium sequencing data) in the ``.h5ad`` format tailored to user's needs.
            # At the moment, we do not support this option.
            self.st_dir = os.path.abspath(st_dir)
            st = Spatial(self.name, self.st_dir, image_path=self.image_path, save_dir=self.save_dir)
            st.process_transcriptome()
            logger.error("Please provide `spot_adata_path`")
            return

        self.set_cell_features_csv_path(cell_features_csv_path)
        self.set_cell_features_list(cell_features_list)

        self.genes = []
        if genes_path is not None:
            self.load_genes(genes_path)

        self.recipe = recipe
        self.graph_params = default_graph_params
        self.run_params = default_run_params
        self.__dict__.update(kwargs)

    def prepare_input(self, mapping_margin=10, spot_identifier="spot_barcodes"):
        """ Prepare the input for the fineST estimation.

            First, generate the cell-wise adata from the cell features and spot adata. In this step, the segmented cells will be read from the
            ``self.cell_features_csv_path`` and the outliers from the segmentation will be removed according to the distance between a cell and
            its nearest neighbor. Second, the spot gene expression is mapped to aligned nearest cells. Lastly, the spot heterogeneity will be
            computed using the image features for future construction of the cell-cell graph and the transition matrix.

        Parameters
        ----------
        mapping_margin : :py:class:`int` or :py:class:`float`, optional
            Margin for mapping the spot gene expression to the cells. Default is 10, which will attempt to map cells which are within 10- spot radius of any spot (so almost all identified cells are mapped to nearest spots). 
            Decrease this number if you would like to eliminate isolated cells.

        """

        if not hasattr(self, "adata"):
            adata_sc_nearest_spot = generate_cell_adata(
                self.cell_features_csv_path,
                self.spot_adata_path,
                obs_features=self.cell_features_list,
                mapping_margin=mapping_margin
            )

            self.adata = adata_sc_nearest_spot

        # compute spot heterogeneity, excluding the first two columns which are the 2D positions of the cells
        obs_edited = get_spot_heterogeneity_cv(
            self.adata.obs,
            self.cell_features_list[2:],
            spot_identifier
            )

        self.adata.obs.loc[:, "spot_heterogeneity"] = obs_edited
        #self.write_adata(f"{self.name}_adata_cell_pre-run.h5ad", self.adata)
        #self.data_pre_path = os.path.join(self.save_dir, f"{self.name}_adata_cell_pre-run.h5ad")

    def vae_training(
            self, vae_genes_set=None, min_mean_expression=0.1, **kwargs
    ):
        """ Train a VAE model for the spot-level transcriptome data. 

        Parameters
        ----------
        vae_genes_set : :py:class:`set` or :py:obj:`None`, optional
            Set of genes to be used for VAE training. 
                If :py:obj:`None`, all the genes (``adata.var.used_for_prediction``, which are specified in the :meth:`prepare_input`) with mean expression > ``min_mean_expression`` will be used.
        min_mean_expression : :py:class:`float`, optional
            Minimum mean expression for the genes to be used for VAE training.
        kwargs : :py:class:`dict`
            Keyword arguments for the :func:`thor.VAE.train_vae` function. 

        Returns
        -------
        :py:obj:`None`

        See Also
        --------
        :func:`thor.VAE.train_vae` : For detailed parameter descriptions and usage.

        """

        if vae_genes_set is None:
            vae_genes_set = set(
                self.adata.var_names[np.logical_and(
                    self.adata.var.used_for_prediction, self.adata.var.means
                    > min_mean_expression
                )]
            )

        assert vae_genes_set.issubset(
            set(self.adata.var_names[self.adata.var.used_for_prediction])
        )

        self.adata.var["used_for_vae"] = self.adata.var_names.isin(
            vae_genes_set
        )
        ad = self.adata[:, self.adata.var.used_for_vae]
        X = ad.X.toarray()
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)

        model_path = os.path.join(self.save_dir, "VAE_models")
        model_save_prefix = os.path.join(model_path, self.name)
        kwargs.update({"save_prefix": model_save_prefix})

        train_vae(X_norm, **kwargs)
        self.model_path = model_path

        with open(
            os.path.join(model_path, f"vae_genes_{self.name}.csv"), "w"
        ) as tfile:
            tfile.write("\n".join(vae_genes_set))

    def load_genes(self, genes_file_path):
        """Load the user-input genes to be used for prediction.

        Parameters
        ----------
        genes_file_path: :py:class:`str`
            Path to the csv file that contains the genes to be used for prediction. The genes should be in the first column of the csv file.
            Gene naming convention should match ``self.adata.var_names``.

        Returns
        -------
        :py:obj:`None`
            The genes are loaded into the list ``self.genes``.
        """

        genes_list = pd.read_csv(genes_file_path,
                                 header=None).values[:, 0].tolist()

        self.genes = genes_list

    def set_cell_features_csv_path(self, cell_features_csv_path=None):
        """Set the path to the CSV file containing cell features.

        Parameters
        ----------
        cell_features_csv_path : :py:class:`str` or :py:obj:`None`, optional
            Path to the CSV file containing cell features. If :py:obj:`None`, the cell features csv file will be obtained from the WSI, which includes nuclei segmentation and feature extraction.

        Returns
        -------
        :py:obj:`None`
            The file path is stored in ``self.cell_features_csv_path``.
        """
        if cell_features_csv_path is not None:
            self.cell_features_csv_path = os.path.abspath(cell_features_csv_path)
            return None

        wsi = WholeSlideImage(self.image_path, name=self.name, save_dir=self.save_dir)
        wsi.process()
        self.cell_features_csv_path = wsi.cell_features_csv_path

    def set_cell_features_list(self, cell_features_list=None):
        """Set the list of cell features to be used for graph construction.

        Parameters
        ----------
        cell_features_list : :py:class:`list` or :py:obj:`None`, optional
            List of features to be used for generating the cell-cell graph. If :py:obj:`None`, default features will be used:
            ["x", "y", "mean_gray", "std_gray", "entropy_img", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]

        Returns
        -------
        :py:obj:`None`
            The feature names are stored in the list ``self.cell_features_list``.
        """

        assert self.cell_features_csv_path is not None, "Please provide the cell features csv file path."
        
        cell_features_df = pd.read_csv(self.cell_features_csv_path, index_col=0)
        feature_names = cell_features_df.columns

        if cell_features_list is not None:
            self.cell_features_list = list(
                feature_names[feature_names.isin(cell_features_list)]
            )
        else:
            self.cell_features_list = list(feature_names)

    def set_params(self, **kwargs):
        """Set the parameters for the fineST estimation.

        This method allows you to configure both graph construction and Markov diffusion parameters.
        All parameters are optional and will update the default values.

        .. rst-class:: parameter-group
        
        **Graph Construction Parameters**
        =================================

        n_neighbors : :py:class:`int`, optional
            Number of neighbors for cell-cell graph construction. Default is 5.
        obs_keys : :py:class:`list` or :py:obj:`None`, optional
            List of observation keys to use for graph construction. If :py:obj:`None`, uses default features.
        reduced_dimension_transcriptome_obsm_key : :py:class:`str`, optional
            Key in ``obsm`` for reduced dimension representation. Default is "X_pca".
        reduced_dimension_transcriptome_obsm_dims : :py:class:`int`, optional
            Number of dimensions to use from the reduced representation. Default is 2.
        geom_morph_ratio : :py:class:`float`, optional
            Ratio between geometric and morphological distances. Default is 1.
        geom_constraint : :py:class:`float`, optional
            Constraint on geometric distances. Default is 0.
        snn_threshold : :py:class:`float`, optional
            Threshold for shared nearest neighbor graph. Default is 0.1.
        node_features_obs_list : :py:class:`list`, optional
            List of node features to use. Default is ["spot_heterogeneity"].
        balance_cell_quality : :py:class:`bool`, optional
            Whether to balance cell quality. Default is :py:obj:`False`.
        bcq_IQR : :py:class:`tuple`, optional
            Interquartile range for balancing cell quality. Default is (0.15, 0.85).

        .. rst-class:: parameter-group
        
        **Transition Matrix Parameters**
        ================================

        preferential_flow : :py:class:`bool`, optional
            Whether to use preferential flow in transition matrix. Default is :py:obj:`True`.
        weigh_cells : :py:class:`bool`, optional
            Whether to weigh cells (nodes) by node features. The idea is to give more weight to the cells 
            with higher quality (e.g. lower heterogeneity with surrounding cells). Default is :py:obj:`True`.
        smoothing_scale : :py:class:`float`, optional
            Scale for smoothing when constructing the transition matrix. The lower the scale, the more 
            self-transition (i.e. the more likely to stay in the same state). Default is 0.8.
            
            The diffusion transition matrix is defined as:
            
            .. math::
            
               T = I - \\lambda \\cdot K
            
            where :math:`I` is the identity matrix, :math:`K` is the connectivity matrix,
            and :math:`\\lambda` is the smoothing scale.
            
        inflation_percentage : :py:class:`float` or :py:obj:`None`, optional
            Percentage for reverse diffusion scale relative to the forward diffusion scale ``smoothing_scale``.
            Default is :py:obj:`None` (no reverse diffusion). When enabled, the reverse diffusion will be performed right after the forward diffusion every iteration.
            
            The reverse diffusion transition matrix is defined as:
            
            .. math::
            
               T_{rev} = I - \\mu \\cdot K
            
            where :math:`I` is the identity matrix, :math:`K` is the connectivity matrix,
            and :math:`\\mu` is the reverse diffusion scale.
            :math:`\\mu = -\\lambda \\cdot (1 + \\text{inflation_percentage} \\cdot 100)`
            
        conn_csr_matrix : :class:`scipy.sparse.csr_matrix` or :py:obj:`None`, optional
            Pre-computed connectivity matrix. Default is :py:obj:`None`. 
            
            * If provided, the connectivity matrix will be used and the other parameters will be ignored. 
            * If :py:obj:`None`, the connectivity matrix will be computed from the cell-cell graph if it does not exist in ``adata.obsp``; else the existing connectivity matrix will be used.
            * If "force", the connectivity matrix will be computed regardless of whether it already exists.
        
        .. rst-class:: parameter-group
        
        **Markov Diffusion Parameters**
        ===============================

        initialize : :py:class:`bool`, optional
            Whether to initialize graph and transition matrix. Default is :py:obj:`True`.
        burn_in_steps : :py:class:`int`, optional
            Number of steps for burn-in period. Default is 5.
        layer : :py:class:`str` or :py:obj:`None`, optional
            Layer in AnnData to use for gene expression. Default is :py:obj:`None` (use ``.X``).
        is_rawCount : :py:class:`bool`, optional
            Whether the input data is raw counts. Default is :py:obj:`False`.
        regulate_expression_mean : :py:class:`bool`, optional
            Whether to regulate expression mean. Default is :py:obj:`False`.
        stochastic_expression_neighbors_level : :py:class:`str`, optional
            Level for stochastic expression neighbors ("spot" or "cell"). Default is "spot".
        n_iter : :py:class:`int`, optional
            Number of iterations for Markov diffusion. Default is 20.
        conn_key : :py:class:`str`, optional
            Key for connectivity matrix in ``.obsp``. Default is "snn".
        write_freq : :py:class:`int`, optional
            Frequency of writing results to disk. Default is 10.
        out_prefix : :py:class:`str`, optional
            Prefix for output files. Default is "y".
        sample_predicted_expression_fluctuation_scale : :py:class:`float`, optional
            Scale for fluctuation in predicted expression. Default is 1.
        smooth_predicted_expression_steps : :py:class:`int`, optional
            Number of steps for smoothing predicted expression. Default is 0.
        save_chain : :py:class:`bool`, optional
            Whether to save the MCMC chain. Default is :py:obj:`False`.
        n_jobs : :py:class:`int`, optional
            Number of parallel jobs to run. Default is 1.
        adjust_cell_network_by_transcriptome_scale : :py:class:`float`, optional
            Scale for adjusting cell network by transcriptome. Default is 0.

        See Also
        --------
        :meth:`predict_gene_expression` : For using these parameters in prediction.
        :func:`thor.markov_graph_diffusion.markov_graph_diffusion_initialize` : For graph construction details.
        :func:`thor.markov_graph_diffusion.estimate_expression_markov_graph_diffusion` : For Markov diffusion details.
        """
        for param in self.run_params:
            if param in kwargs:
                self.run_params.update({param: kwargs[param]})

        for param in self.graph_params:
            if param in kwargs:
                self.graph_params.update({param: kwargs[param]})

    def sanity_check(self):
        """Perform sanity checks on the input data and parameters.

        This function verifies that all necessary attributes and parameters are set correctly before running the fineST estimation.

        Returns
        -------
        :py:class:`bool`
            Returns :py:obj:`True` if all checks pass, otherwise :py:obj:`False`.

        """

        if self.recipe in ["gene"]:
            required_attrs_for_prediction = ["adata"]
        else:
            required_attrs_for_prediction = ["adata", "model_path", "generate"]

        for attr in required_attrs_for_prediction:
            if not hasattr(self, attr):
                logger.error(
                    f"Need to set attribute {attr} before running the prediction"
                )
                return False

        return True

    def predict_gene_expression(self, **kwargs):
        """Predict gene expression using Markov graph diffusion.

        This method performs the following steps:
        1. Updates parameters using :meth:`set_params`
        2. Prepares the recipe based on the selected mode
        3. Optionally performs burn-in if transcriptome effects are included
        4. Initializes the cell graph and transition matrix
        5. Runs the Markov graph diffusion process
        6. Saves results and cleans up temporary files

        Parameters
        ----------
        **kwargs : :py:class:`dict`
            Same parameters as accepted by :meth:`set_params`. 

        Returns
        -------
        :class:`anndata.AnnData`
            Anndata object with predicted gene expression for cells.

        See Also
        --------
        :meth:`set_params` : For detailed parameter descriptions.
        :func:`thor.markov_graph_diffusion.estimate_expression_markov_graph_diffusion` : For the underlying implementation.
        """

        self.set_params(**kwargs)
        
        self.prepare_recipe()

        #print(self.run_params)
        #print(self.graph_params)

        temp_dir = os.path.join(self.save_dir, "TEMP")
        os.makedirs(temp_dir, exist_ok=True)

        # sanity_check
        if not self.sanity_check():
            return self.adata

        # burn-in if you would like to include the effect of the input transcriptome in the cell-cell graph construction
        if self.graph_params["adjust_cell_network_by_transcriptome_scale"] > 0:
            self._burn_in(n_iter=self.run_params["burn_in_steps"], n_pcs=self.graph_params["reduced_dimension_transcriptome_obsm_dims"])

        # initialization: constructing cell graph and the transition matrix
        if self.run_params["initialize"]:
            markov_graph_diffusion_initialize(self.adata, **self.graph_params)

        self.data_pre_path = os.path.join(self.save_dir,
                                          f"{self.name}_adata_cell_input.h5ad")
        self.run_params["regulate_expression_mean"] = self.run_params["regulate_expression_mean"] and self.recipe == "reduced"
        self.write_adata(f"{self.name}_adata_cell_input.h5ad", self.adata)
        self.write_params()
        self.save()

        # run the finest estimation
        estimate_expression_markov_graph_diffusion(
            self.adata,
            conn_key=self.run_params["conn_key"],
            n_iter=self.run_params["n_iter"],
            input_layer=self.run_params["layer"],
            is_rawCount=self.run_params["is_rawCount"],
            stochastic_expression_neighbors_level=self.
            run_params["stochastic_expression_neighbors_level"],
            regulate_expression_mean=self.run_params["regulate_expression_mean"],
            smooth_predicted_expression_steps=self.
            run_params["smooth_predicted_expression_steps"],
            sample_predicted_expression_fluctuation_scale=self.
            run_params["sample_predicted_expression_fluctuation_scale"],
            n_jobs=self.run_params["n_jobs"],
            out_prefix=self.run_params["out_prefix"],
            write_freq=self.run_params["write_freq"],
            temp_dir=temp_dir,
            save_dir=self.save_dir,
            gen_module=self.generate,
        )

        # Clean up
        if not self.run_params["save_chain"]:
            try:
                shutil.rmtree(temp_dir)
            except OSError as e:
                logger.error(e.strerror)

    def _burn_in(self, n_iter=5, genes_included="highly_variable", n_pcs=2):
        """ Burn-in the Markov graph diffusion. To include the effect of the input transcriptome in the cell-cell graph construction, we run :func:`estimate_expression_markov_graph_diffusion` function for the
        finest estimation using only histology features with vanilla parameters for ``n_iter`` steps (default: 5). In the burnin stage, ``adjust_cell_network_by_transcriptome_scale`` is set to 0. 

        Parameters
        ----------
        n_iter : :py:class:`int`, optional
            Number of iterations for the Markov graph diffusion in burn-in stage. Default is 5. 
        genes_included : :py:class:`str` or :py:obj:`None`, optional
            Genes to be used for PCA of the transcriptome. 
                If :py:obj:`None`, use the genes for prediction in the :class:`thor.fineST` object; 
                If "highly_variable" (default), the highly variable genes (recomputed) will be used; 
                If "all", all the genes will be used. 
        n_pcs : :py:class:`int`, optional
            Number of PCs for PCA of the transcriptome. Default is 2.

        Returns
        -------
        :py:obj:`None`
            Update ``self.adata.obsm["X_pca"]``.
        """

        if n_iter < 1:
            return

        # disable the logging for the burn-in stage at the moment for cleaner output
        logging.disable(logging.INFO)
        logger.info("Burn-in the Markov graph diffusion first.")

        os.environ["TQDM_DISABLE"] = '1'

        # Create a copy of the original object
        burnin = self.copy()
        burnin.adata.X = get_adata_layer_array(burnin.adata, layer_key=burnin.run_params["layer"])

        temp_dir = os.path.join(burnin.save_dir, "BURNIN_TEMP")
        os.makedirs(temp_dir, exist_ok=True)

        # initialization: constructing cell graph and the transition matrix
        burnin.graph_params["adjust_cell_network_by_transcriptome_scale"] = 0
        if burnin.run_params["initialize"]:
            markov_graph_diffusion_initialize(burnin.adata, **burnin.graph_params)

        # # set genes for prediction
        # if genes_included == "highly_variable":
        #     # recompute the highly variable genes based on the cell-wise gene expression
        #     sc.pp.highly_variable_genes(burnin.adata, inplace=True)
        # burnin.genes = genes_included
        burnin.set_genes_for_prediction(genes_selection_key=genes_included)

        burnin.recipe = "gene"
        burnin.prepare_recipe()
        
        # run the markov graph diffusion
        estimate_expression_markov_graph_diffusion(
            burnin.adata,
            conn_key=burnin.run_params["conn_key"],
            n_iter=n_iter,
            input_layer=burnin.run_params["layer"],
            is_rawCount=burnin.run_params["is_rawCount"],
            stochastic_expression_neighbors_level=None,
            regulate_expression_mean=False,
            smooth_predicted_expression_steps=0,
            n_jobs=burnin.run_params["n_jobs"],
            out_prefix="burnin",
            write_freq=n_iter,
            temp_dir=temp_dir,
            save_dir=burnin.save_dir,
            gen_module=burnin.generate,
        )

        ad_burnin = burnin.load_result(f"burnin_{n_iter}.npz")
        sc.tl.pca(ad_burnin, n_comps=n_pcs)
        self.adata.obsm["X_pca"] = ad_burnin.obsm["X_pca"]

        # Clean up
        del ad_burnin
        del burnin
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            logger.error(e.strerror)

        os.environ["TQDM_DISABLE"] = '0'
        logging.disable(logging.NOTSET)

    def write_adata(self, file_name, ad):
        """Write an AnnData object to disk in the results directory.
        
        Parameters
        ----------
        file_name : :py:class:`str`
            Relative path to the file to write. The file will be saved in ``self.save_dir``.
        ad : :class:`anndata.AnnData`
            Cell-wise gene expression to save.
        """

        cell_adata_out_path = os.path.join(self.save_dir, file_name)
        ad.write_h5ad(cell_adata_out_path)

    def write_params(self, exclude=["conn_csr_matrix"]):
        """Write the current parameters to a JSON file.

        Parameters
        ----------
        exclude : :py:class:`list` of :py:class:`str`, optional
            List of parameter names to exclude from writing. Default is ``["conn_csr_matrix"]``.

        Returns
        -------
        :py:obj:`None`
            The parameters are written to a JSON file in ``self.save_dir``.
        """
        with open(
            os.path.join(self.save_dir, f"{self.name}_run_params.json"), "w"
        ) as fp:
            json.dump(self.run_params, fp, indent=4)

        graph_params = self.graph_params.copy()
        if graph_params.get("conn_csr_matrix") is not None:
            del graph_params["conn_csr_matrix"]
        with open(
            os.path.join(self.save_dir, f"{self.name}_graph_params.json"), "w"
        ) as fp:
            json.dump(graph_params, fp, indent=4)

    def load_params(self, json_path):
        """Load parameters from a JSON file. The loaded parameters will be merged with the current parameters in ``self.run_params`` and ``self.graph_params``.

        Parameters
        ----------
        json_path : :py:class:`str`
            Path to the JSON file containing the parameters.
        """
        with open(json_path, "r") as fp:
            params = json.load(fp)
        self.set_params(**params)

    def load_result(self, file_name, layer_name=None):
        """Load the predicted gene expression data and create an :class:`anndata.AnnData` object.

        Parameters
        ----------
        file_name : :py:class:`str`
            Relative path to the file to load. The file should be in ``self.save_dir``.
        layer_name : :py:class:`str` or :py:obj:`None`, optional
            Name of the layer to save the loaded gene expression. If :py:obj:`None`, save the gene expression to ``adata.X``.

        Returns
        -------
        :class:`anndata.AnnData`
            The loaded :class:`anndata.AnnData` object.
        """
        if not hasattr(self, "adata"):
            self.adata = sc.read_h5ad(self.data_pre_path)
        layer_file = os.path.join(self.save_dir, file_name)
        layer_array = load_npz(layer_file)
        ad = self.adata[:, self.adata.var.used_for_prediction].copy()
        del ad.layers

        if layer_name is None:
            ad.X = layer_array
        else:
            ad.layers[layer_name] = layer_array
        return ad

    def prepare_recipe(self):
        """Prepare the recipe for gene expression prediction.

        This function sets up the appropriate genes and parameters based on the selected recipe. Supported recipes are:
        - "gene": use all the user-provided genes for prediction. The user-provided genes should be in the ``self.adata.var.used_for_prediction``.
        - "reduced": use the VAE genes for prediction. The VAE genes should be in the ``self.adata.var.used_for_vae`` and used for prediction, ignoring ``self.genes``.
        - "mix": use both the VAE genes and the rest of the user-provided genes for prediction. The VAE genes should be in the ``self.adata.var.used_for_vae``.

        Returns
        -------
        :py:obj:`None`
            The recipe-specific settings are applied to set ``self.adata.var`` columns for gene selection.
        """

        assert self.recipe in ("gene", "reduced", "mix"), "Please specify one of the implemented recipes: `mix`, `gene`, or `reduced`"
        
        logger.info(f"Using mode {self.recipe}")
        if self.recipe == "mix":
            try:
                self.get_reduced_genes()
            except:
                self.adata.var["used_for_reduced"] = False

            if len(self.adata.var[self.adata.var.used_for_reduced]) == 0:
                logger.warning(
                        "Failed to get reduced genes. Using all genes for prediction."
                        )
                self.adata.var["used_for_prediction"] = True
                self.recipe = "gene"
        
        if self.recipe == "reduced":
            logger.info("Using purely all the VAE genes for prediction. `self.genes` is ignored.")
            self.genes = []
            self.adata.var["used_for_reduced"] = self.adata.var["used_for_vae"]
            self.adata.var["used_for_prediction"] = self.adata.var["used_for_reduced"]
        
        if self.recipe == "gene":
            self.adata.var["used_for_reduced"] = False
            self.adata.var["used_for_vae"] = False
            self.generate = IdentityGenerator()

    def visualize_cell_network(self, **kwargs):
        """Visualize the cell-cell network.

        This function internally calls :func:`thor.plotting.graph.plot_cell_graph` to create the visualization.

        Parameters
        ----------
        **kwargs : :py:class:`dict`
            Additional parameters for the visualization. See :func:`thor.plotting.graph.plot_cell_graph`
            for available options.

        Returns
        -------
        :py:obj:`None`
            The network visualization is displayed.

        See Also
        --------
        :func:`thor.plotting.graph.plot_cell_graph` : For detailed parameter descriptions and usage.
        """

        return plot_cell_graph(self.adata, **kwargs)

    def set_genes_for_prediction(self, genes_selection_key="highly_variable"):
        """Set genes to be used for prediction.

        Parameters
        ----------
        genes_selection_key : :py:class:`str`, optional
            Key for gene selection in ``self.adata.var``. Default: "highly_variable"
            
            Valid options:
            
            - "highly_variable": Selects highly variable genes
            - "all": Selects all genes (not recommended)
            - :py:obj:`None`: Uses genes specified in ``self.genes``
            - Any key in ``self.adata.var``: Uses that key for selection

        Returns
        -------
        :py:obj:`None`
            The selected genes are marked in ``self.adata.var.used_for_prediction``.
        """

        ad = self.adata
        if genes_selection_key == "all":
            logger.warning(
                "Using all the genes. This may not be optimal and can be slow."
            )
            ad.var["used_for_prediction"] = True
            return

        if genes_selection_key is None:
            assert len(self.genes) > 0, "No genes provided. Please set `self.genes` or provide a valid genes_selection_key."
            ad.var["used_for_prediction"] = ad.var.index.isin(self.genes)
            return
        
        # If the genes_selection_key is not None, we will use the genes in adata.var[genes_selection_key] + self.genes for prediction.
        assert genes_selection_key in ad.var, f"{genes_selection_key} is not a valid key in adata.var."

        selected_list = list(ad.var_names[ad.var[genes_selection_key]])
        combined_list = list(set(selected_list + self.genes))
        ad.var["used_for_prediction"] = ad.var.index.isin(combined_list)

    def get_reduced_genes(self, keep=0.9, min_mean_expression=0.5):
        """Get a reduced set of genes which were used to train the VAE model. This is because the genes used for VAE training may not be reconstructed faithfully to the same extent. Therefore, we will use the genes with high reconstruction
        quality (measured by cosine similarity with the input gene expression). One should be aware that the genes used for VAE training (``self.adata.var.used_for_vae``) are not the same as the genes used for thor prediction in reduced mode (``self.adata.var.used_for_reduced``; subset).

        Parameters
        ----------
        keep : :py:class:`float`, optional
            Fraction of genes to keep based on their importance in the VAE model for thor prediction in reduced mode. The genes are ranked according to the VAE reconstruction
            quality (measured by cosine similarity with the input gene expression). Default is 0.9.
        min_mean_expression : :py:class:`float`, optional
            Minimum mean expression for genes to be considered. Note the expression values are log-transformed normalized expression. Default is 0.5.

        Returns
        -------
        :py:obj:`None`
            The selected genes are marked in ``self.adata.var.used_for_reduced``.
        """

        ad_spot = sc.read_h5ad(self.spot_adata_path)

        assert "used_for_vae" in self.adata.var, "Please set up `used_for_vae` column in adata.var by running either `self.vae_training` or `self.load_vae_model`"

        genes = self.adata.var_names[self.adata.var.used_for_vae]

        assert len(
            genes
        ) > 1, "No gene was used for VAE training. There is something wrong with the VAE model."

        X = get_adata_layer_array(ad_spot[:, genes])
        z = self.generate.encode(X)
        decoded = self.generate.decode(z)
        cos_genes = 1 - var_cos(decoded, X)
        mean_exp_genes = X.mean(axis=0)
        index_low_to_high = np.array(np.argsort(cos_genes))
        rank = np.argsort(index_low_to_high) + 1
        ad = self.adata[:, genes]
        ad.var["reconstr_rank"] = rank.tolist()
        ad.var["used_for_reduced"] = np.logical_and(
            ad.var["reconstr_rank"] < keep * ad.shape[1], mean_exp_genes
            >= min_mean_expression
        )
        reduced_genes = list(ad.var_names[ad.var.used_for_reduced])

        if "used_for_prediction" in self.adata.var:
            self.adata.var["used_for_reduced"] = np.logical_and(
                self.adata.var_names.isin(reduced_genes),
                self.adata.var.used_for_prediction
            )
        else:
            self.adata.var["used_for_reduced"] = self.adata.var_names.isin(
                reduced_genes
            )

    def load_generate_model(self):
        if self.model_path is None:
            self.generate = IdentityGenerator()
        else:
            self.load_vae_model(self.model_path)

    def load_vae_model(self, model_path=None):
        """Load a pre-trained VAE model.

        Parameters
        ----------
        model_path : :py:class:`str` or :py:obj:`None`, optional
            Path to the directory containing the VAE model. The model should be saved in the `encoder` and `decoder` subdirectories, with the filenames `{self.name}_VAE_encoder.h5` and `{self.name}_VAE_decoder.h5`.

        Returns
        -------
        :py:obj:`None`
            The VAE model is loaded into the instance as `self.generate` and the model path is stored in `self.model_path`.
        """

        if model_path is None:
            logger.error("Please provide a correct model path.")
            return

        encoder_model_path = os.path.join(model_path, "encoder")
        decoder_model_path = os.path.join(model_path, "decoder")

        vae = VAE()
        vae.load_models(
            encoder_model_path=encoder_model_path,
            decoder_model_path=decoder_model_path,
        )
        self.generate = vae
        self.model_path = model_path

        # load the associated genes used for vae training.
        # single column headless genes
        vae_genes_file_path = os.path.join(
            self.model_path, f"vae_genes_{self.name}.csv"
        )
        df = pd.read_csv(vae_genes_file_path, header=None, index_col=0)
        vae_genes_set = set(df.index)
        self.adata.var["used_for_vae"] = self.adata.var.index.isin(
            vae_genes_set
        )
#        self.adata.obsm["X_vae"] = self.generate.encode(self.adata[:, self.adata.var["used_for_vae"]].X.toarray())

    def save(self, exclude=["generate", "adata", "conn_csr_matrix"]):
        """Save the current state of the instance. The saved JSON file can be used to create a new instance of the class.

        Parameters
        ----------
        exclude : :py:class:`list` of :py:class:`str`, optional
            List of attributes to exclude from saving. Default is ["generate", "adata", "conn_csr_matrix"].

        Returns
        -------
        :py:obj:`None`
            The instance state is saved to a JSON file.
        """
        
        attrs = self.__dict__.copy()
        for attr in exclude:
            if attr in attrs:
                del attrs[attr]
        with open(os.path.join(self.save_dir, f"{self.name}_fineST.json"), "w") as fp:
            json.dump(attrs, fp, indent=4)


    def __deepcopy__(self, memo):
        # Create a new instance of the class
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance

        # Copy all attributes except 'generate'
        for k, v in self.__dict__.items():
            if k != 'generate':
                setattr(new_instance, k, copy.deepcopy(v, memo))
            else:
                new_instance.load_generate_model(self.model_path)

        return new_instance

    def copy(self):
        """Create a deep copy of the instance.

        Returns
        -------
        :class:`fineST`
            A new instance that is a deep copy of the current instance.
        """
        return copy.deepcopy(self)

    
    @classmethod
    def load(cls, json_path):
        """Load a fineST instance from a JSON file.

        Parameters
        ----------
        json_path : :py:class:`str`
            Path to the JSON file containing the saved instance.

        Returns
        -------
        :class:`fineST`
            A new instance loaded from the JSON file.
        """
        with open(json_path, "r") as fp:
            obj = json.load(fp)

        obj = cls(**obj)

        if hasattr(obj, "data_pre_path"):
            obj.adata = sc.read_h5ad(obj.data_pre_path)

        if hasattr(obj, "model_path"):
            obj.load_generate_model(obj.model_path)

        return obj

