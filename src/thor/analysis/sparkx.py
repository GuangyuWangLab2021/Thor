import os
import pandas as pd
import scanpy as sc
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler


class SPARKX:
    """Class for running `SPARK-X <https://doi.org/10.1186/s13059-021-02404-0>`_.

    Parameters
    ----------
    rscript_path: :py:class:`str`, default: "R/run_SPARKX.R"
        Path to the R script for running SPARKX.

    """

    def __init__(
        self,
        rscript_path="R/run_SPARKX.R",
        **kwargs,
    ) -> None:
        self.sparkscript = rscript_path

    def RUN_SPARKX_R(self, adata_path=None, layer=None, out_path=None):
        """Run SPARK-X with provided R script.
        
        Parameters
        ----------
        adata_path: :py:class:`str`
            Path to the AnnData object.
        layer: :py:class:`str`, default: :py:obj:`None`
            Layer of the AnnData object to use.
        out_path: :py:class:`str`, default: :py:obj:`None`
            Path to the output directory.
        """
        
        if layer is None:
            os.system(
                f"Rscript {self.sparkscript} -f {adata_path} -s {out_path}"
            )
            self.out_directory = os.path.join(out_path, "raw")
        else:
            os.system(
                f"Rscript {self.sparkscript} -f {adata_path} -p {layer} -s {out_path}"
            )
            self.out_directory = os.path.join(out_path, layer)
        self.adata_path = adata_path
        self.layer = layer

    def load_result(self):
        """Load the result of SPARK-X.
        
        """
        residual_filename = "res_matrix.csv"
        self.residual = pd.read_csv(
            os.path.join(self.out_directory, residual_filename),
            index_col=0,
            engine="c",
            na_filter=False,
            low_memory=False
        )

        svg_list = list(self.residual.index)

        if hasattr(self, "adata_path"):
            ad = sc.read_h5ad(self.adata_path)
            ad.var["spatially_variable"] = ad.var.index.isin(svg_list)
            ad.write_h5ad(self.adata_path)
            return ad

    def load_gene_modules(self, pattern_prefix="SP"):
        """Load the gene modules of SPARK-X.

        Parameters
        ----------
        pattern_prefix: :py:class:`str`, default: "SP"
            Prefix of the gene modules.
        """

        assert hasattr(self, "labels"), "Run clustering first!"
        pattern = pd.DataFrame(
            self.labels, index=self.residual.index, columns=["cluster"]
        )
        if ~hasattr(self, "adata"):
            self.adata = sc.read_h5ad(self.adata_path)
        self.adata = self.compute_pattern_mean(
            self.adata, self.residual, pattern, pattern_prefix
        )

    def hierarchy_clustering(self, **hc_kwargs):
        """Run hierarchical clustering with sklearn's AgglomerativeClustering on the residual matrix.

        Parameters
        ----------
        hc_kwargs: :py:class:`dict`
            Keyword arguments for AgglomerativeClustering.

        Returns
        -------
        labels: :py:class:`numpy.ndarray` (n_cells,)
            Cluster labels.
        """
        hierarchical_cluster = AgglomerativeClustering(**hc_kwargs)
        labels = hierarchical_cluster.fit_predict(self.residual)
        self.labels = labels

    def kmeans_clustering(self, n_patterns, **kmeans_kwargs):
        """Run k-means clustering with sklearn's KMeans on the residual matrix.

        Parameters
        ----------
        n_patterns: :py:class:`int`
            Number of clusters.
        kmeans_kwargs: :py:class:`dict`
            Keyword arguments for KMeans.

        Returns
        -------
        labels: :py:class:`numpy.ndarray` (n_cells,)
            Cluster labels.
        """
        
        kmeans_kwargs.update({"n_clusters": n_patterns})
        kmeans = KMeans(**kmeans_kwargs)
        labels = kmeans.fit_predict(self.residual)
        self.labels = labels

    @staticmethod
    def compute_pattern_mean(adata, data, pattern, obskey_prefix):
        """Compute the mean expression of each gene module.

        Parameters
        ----------
        adata: :py:class:`anndata.AnnData`
            AnnData object.
        data: :py:class:`pandas.DataFrame` (n_sig_genes x n_cells)
            Residual matrix of SPARK-X.
        pattern: :py:class:`pandas.DataFrame` (n_sig_genes x 1), column is cluster, index is gene
        obskey_prefix: :py:class:`str`
            Prefix of the observation key.

        Returns
        -------
        adata: :py:class:`anndata.AnnData`
            AnnData object with the computed pattern mean.
        """
        df = pd.DataFrame(
            {
                c: data.loc[pattern["cluster"] == c].mean()
                for c in pattern["cluster"].drop_duplicates()
            }
        )
        df = pd.DataFrame(
            MinMaxScaler().fit_transform(df.T).T,
            columns=df.columns,
            index=df.index
        )
        for p in df.columns:
            adata.obs[f"{obskey_prefix}{p}"] = df[p]
        return adata
