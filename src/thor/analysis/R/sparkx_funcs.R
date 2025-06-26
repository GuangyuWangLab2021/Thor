library(ggplot2)
library(RColorBrewer)
library(gridExtra)
library(grid)
library(reshape2)

## anscombe variance stabilizing transformation: NB
var_stabilize <- function(x, sv = 1) {
    varx = apply(x, 1, var)
    meanx = apply(x, 1, mean)
    phi = coef(nls(varx ~ meanx + phi * meanx^2, start = list(phi = sv)))
    return(log(x + 1/(2 * phi)))
}# end func


relative_func <- function(expres) {
    maxd = max(expres) - min(expres)
    rexpr = (expres - min(expres))/maxd
    return(rexpr)
}# end func

##-----------------------------------------------------------------
## for data frame
pattern_plot2 <- function(pltdat, igene, xy = T, main = F, titlesize = 2, 
    pointsize = 0.1, xpand = 0, ypand = 1, title = NULL) {
    if (!xy) {
        xy <- matrix(as.numeric(do.call(rbind, strsplit(as.character(pltdat[, 
            1]), split = "x"))), ncol = 2)
        rownames(xy) <- as.character(pltdat[, 1])
        colnames(xy) <- c("x", "y")
        pd <- cbind.data.frame(xy, pltdat[, 2:ncol(pltdat)])
    } else {
        pd <- pltdat
    }
    
    # pal <- colorRampPalette(c('seagreen1','orangered')) pal <-
    # colorRampPalette(c('#00274c','#ffcb05')) pal <-
    # colorRampPalette(c('deepskyblue','goldenrod1')) pal <-
    # colorRampPalette(c('deepskyblue','deeppink'))
    pal <- colorRampPalette(c("mediumseagreen", "lightyellow2", "deeppink"))
    gpt <- ggplot(pd, aes(x = x, y = y, color = pd[, igene + 2])) + geom_point(size = pointsize) + 
        # scale_color_gradientn(colours=pal(5))+
    scale_color_gradientn(colours = pal(5)) + scale_x_discrete(expand = c(xpand, 
        ypand)) + scale_y_discrete(expand = c(xpand, ypand)) + coord_equal() + 
        # labs(title = colnames(pd)[igene+2], x = NULL, y = NULL)+
    theme_bw()
    if (main) {
        if (is.null(title)) {
            title = colnames(pd)[igene + 2]
        }
        out = gpt + labs(title = title, x = NULL, y = NULL) + theme(legend.position = "none", 
            plot.title = element_text(hjust = 0.5, size = rel(titlesize)))
    } else {
        out = gpt + labs(title = NULL, x = NULL, y = NULL) + theme(legend.position = "none")
    }
    return(out)
}# end func



##-----------------------------------------------------------------
## no requirement on equal coordinates and provide additional space on
## xy
pattern_plot3 <- function(pltdat, igene, xy = T, main = F, titlesize = 2, 
    pointsize = 0.1, min.pand = 0.99, max.pand = 1.01, title = NULL) {
    if (!xy) {
        xy <- matrix(as.numeric(do.call(rbind, strsplit(as.character(pltdat[, 
            1]), split = "x"))), ncol = 2)
        rownames(xy) <- as.character(pltdat[, 1])
        colnames(xy) <- c("x", "y")
        pd <- cbind.data.frame(xy, pltdat[, 2:ncol(pltdat)])
    } else {
        pd <- pltdat
    }
    
    pal <- colorRampPalette(c("mediumseagreen", "lightyellow2", "deeppink"))
    gpt <- ggplot(pd, aes(x = x, y = y, color = pd[, igene + 2])) + geom_point(size = pointsize) + 
        scale_color_gradientn(colours = pal(5)) + scale_x_discrete(expand = c(0, 
        1)) + scale_y_discrete(expand = c(0, 1)) + expand_limits(x = c(min(pd$x) * 
        min.pand, max(pd$x) * max.pand), y = c(min(pd$y) * min.pand, max(pd$y) * 
        max.pand)) + theme_bw()
    if (main) {
        if (is.null(title)) {
            title = colnames(pd)[igene + 2]
        }
        out = gpt + labs(title = title, x = NULL, y = NULL) + theme(legend.position = "none", 
            plot.title = element_text(hjust = 0.5, size = rel(titlesize)))
    } else {
        out = gpt + labs(title = NULL, x = NULL, y = NULL) + theme(legend.position = "none")
    }
    return(out)
}# end func


