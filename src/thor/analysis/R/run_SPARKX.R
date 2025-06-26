# install required packages
required_packages <- c("optparse", "amap", "anndata", "devtools", "dynutils")
to_install <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(to_install)) install.packages(to_install, repos = "http://cran.us.r-project.org")

# install SPARK if not yet installed
if(!("SPARK" %in% installed.packages()[,"Package"]))
  devtools::install_github('xzhoulab/SPARK')

library(amap)
LMReg <- function(ct, T) {
    return(lm(ct ~ T)$residuals)
}


library(dynutils)
#library(Matrix)

runSparkx <- function(path, para, sample_name){
    library(anndata)
    library(SPARK)
    ad = read_h5ad(path)

    if(is.null(para)) {
        counts = ad$raw$X 
        rownames(counts) <- ad$raw$obs_names
        colnames(counts) <- ad$raw$var_names
        para = "raw"
    } else if (toupper(para) == "X") {
        counts = ad$X
    } else {
        counts = ad$layers[para]
    }

    if(is_sparse(counts)) {
        counts <- as.matrix(counts)
    }

    out_dir = file.path(sample_name, para)
    dir.create(out_dir, showWarnings = TRUE, recursive = TRUE, mode = "0777")

    counts_sc = t(counts)
    location_sc <- cbind.data.frame(x = ad$obsm$spatial[,1], y = ad$obsm$spatial[,2])
    lib_size <- apply(counts_sc, 2, sum)

    sparkx <- sparkx(counts_sc, location_sc, numCores=1, option="mixture")
    saveRDS(sparkx, file = file.path(out_dir, 'sparkx.rds'))
    write.csv(sparkx$res_mtest, file.path(out_dir, 'qval.csv'))

    vst_count_sc <- counts_sc
    sig_vst_count_sc <- vst_count_sc[which(sparkx$res_mtest$adjustedPval < 0.05),]
    sig_vst_res_sc <- t(apply(sig_vst_count_sc, 1, LMReg, T = log(lib_size)))

    write.csv(sig_vst_res_sc, file.path(out_dir, 'res_matrix.csv'))
    return(sparkx)
}


suppressMessages(library("optparse"))
options(warn=-1)

option_list = list(
                   make_option(c("-f", "--inputFile"), type="character", help="Input anndata file", metavar="character"),
                   make_option(c("-p", "--inputParameter"), type="character", help="Input parameter of gene expression", metavar="character"),
                   make_option(c("-s", "--sampleName"), type="character", help="output directory", metavar="character")
                   );
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser)

inputFile = opt$inputFile
inputParameter = opt$inputParameter
sample_name = opt$sampleName

sparkx <- runSparkx(
                    path = inputFile,
                    para = inputParameter,
                    sample_name = sample_name
)
