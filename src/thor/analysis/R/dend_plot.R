#
#library(dendextend)

suppressPackageStartupMessages(library(dendextend))

num = 6


hc = readRDS('SPARKX_cell/y_smooth_20/hc.rds')
dend <- as.dendrogram(hc)


GENE <- c("Lypla1","Cpa6", "Prex2", "Kcnb2", "Rdh10", "Mcm3", "Rims1", "Bend6")
Pattern <- c("P8", "P2", "P6", "P1", "P5", "P4", "P3", "P7")

LAB = rep("", nobs(dend))
LAB[hc$order[match(GENE, hc$labels)]] = Pattern


dend <- dend %>%
  color_branches(k = num) %>%
  set("branches_lwd", c(2,1,2)) %>%
  set("branches_lty", c(1,2,1)) %>%
  set("labels", LAB) %>%
  color_labels(dend, k = num)

plot(dend, horiz = F)


