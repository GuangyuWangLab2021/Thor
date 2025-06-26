import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import textalloc as ta


def fringe(
    data=None,
    genes=None,
    cmap='viridis',
    lw=5,
    annotate=False,
    annot_bbox=None,
    text_offset_x=0.08,
    text_offset_y=0,
    text_size=8,
    ax=None
):
    """Plot log2foldchange of gene expression against distance from the baseline.

    Parameters
    ----------
    data : Pandas DataFrame
        Rows are windows/bins. Columns are `distance` and all the genes (Default value = None)
    genes : List
        List of genes for plotting (Default value = None)
    cmap : Str
        Colormap name in matplotlib for plotting the genes. (Default value = 'viridis')
    lw : Numeric
        line width. (Default value = 5)
    annotate : Bool
        Whether to annotate the lines with the gene names. (Default value = False)
    text_offset_x : Numeric
        Offset applied to the x position of the annotation texts. (Default value = 0.08)
    text_offset_y : Numeric
        Offset applied to the y position of the annotation texts. (Default value = 0)
    text_size : Numeric
        Size of the annotation texts. (Default value = 8)
    ax : Matplotlib axis passed on for plotting
         (Default value = None)

    Returns
    -------
    ax : Matplotlib axis

    
    """
    edge_val = {g: abs(data[g].iloc[-1]) for g in genes}

    vmax = max(edge_val.values())
    #vmin = min(edge_val.values())
    vmin = 0

    try:
        cmap = mpl.colormaps[cmap]
    except:
        pass

    i0 = len(ax.lines)
    for g in genes:
        c = cmap((edge_val[g] - vmin) / vmax)
        ax = sns.lineplot(data, x=data.index, y=g, color=c, ax=ax, lw=lw)

    # unused
    if False:
         for i, g in enumerate(genes):
            line = ax.lines[i + i0]
            y = line.get_ydata()[-1] * (1 + text_offset_y * ((i + 1) % 4))
            x = 1 + text_offset_x * (i % 4)
            ax.annotate(
                g,
                weight='bold',
                xy=(x, y),
                xytext=(6, 0),
                color=line.get_color(),
                xycoords=ax.get_yaxis_transform(),
                textcoords="offset points",
                size=text_size,
                va="center"
            )
        # fig = ax.get_figure()
        # x_ = [0]*len(genes)
        # y_ = [edge_val[g] for g in genes]
        # ta.allocate_text(fig, ax, x_, y_, genes, x_scatter=x_, y_scatter=y_, textsize=10)
    if annotate:
        colors = [mpl.colors.rgb2hex(ax.lines[i + i0].get_color(), keep_alpha=True) for i in range(len(genes))]
        annotate_with_table(ax, genes, colors, bbox=annot_bbox)


    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(u"Distance from the edge (\u03bcm)")
    ax.set_ylabel("log2(fold change)")
    return ax


def annotate_with_table(ax, texts, colors, bbox=[1, 0, 0.3, 0.5]):
    # Create a table for all data points with colored text
    def split_lists(lst, display_rows=10):
        size = len(lst)
        left_over = size % display_rows
        if left_over > 0:
            fill = display_rows - left_over 
            lst += [''] * fill

        stops = np.linspace(0, len(lst), display_rows+1).astype(int)
        lst_2d = []
        for i in range(display_rows):
            lst_2d.append(lst[stops[i]:stops[i+1]])
        return lst_2d

    data_table = split_lists(texts)
    colors = split_lists(colors)
    table = plt.table(cellText=data_table, cellLoc='center', bbox=bbox)
    table.auto_set_font_size(True)
    #table.set_fontsize(10)

    # Hide table borders
    for key, cell in table.get_celld().items():
        cell.visible_edges = ""
        c = colors[key[0]][key[1]]
        if c != '':
            cell.get_text().set_color(c)


def deg(
    data=None,
    genes=None,
    baseline_from_edge=[0,0],
    cmaps=['Oranges', 'Blues'],
    lw=5,
    annotate=False,
    text_offset_x=0.08,
    text_offset_y=0,
    text_size=8,
    **subplots_kwds
):
    """
    Plot log2foldchange of gene expression against distance from the baseline.

    Parameters
    ----------
    data : pandas DataFrame
        A DataFrame where rows represent windows/bins and columns include 'distance' and gene expression data.
    genes : tuple or list of two lists
        Lists of up-regulated and down-regulated genes to plot.
    cmaps : tuple or list, optional
        Colormap names in matplotlib for plotting the genes. Default is ['Oranges', 'Blues'].
    lw : numeric, optional
        Line width for the plot. Default is 5.
    annotate : bool, optional
        Whether to annotate the lines with the gene names. Default is False.
    text_offset_x : numeric, optional
        Offset applied to the x position of the annotation texts. Default is 0.08.
    text_offset_y : numeric, optional
        Offset applied to the y position of the annotation texts. Default is 0.
    text_size : numeric, optional
        Size of the annotation texts. Default is 8.
    **subplots_kwds : dict, optional
        Keyword arguments for the matplotlib.pyplot subplots function.

    Notes
    -----
    This function plots the log2foldchange of gene expression against the distance from the baseline for up-regulated and down-regulated genes separately.

    Returns
    -------
    None
        This function does not return any value. It generates the plot directly.

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt

    >>> # Assuming data is a pandas DataFrame with columns 'distance', 'gene1', 'gene2', etc.
    >>> data = pd.DataFrame(...)
    >>> up_genes = ['gene1', 'gene3', 'gene5']
    >>> down_genes = ['gene2', 'gene4', 'gene6']
    >>> genes = (up_genes, down_genes)

    >>> # Plot the log2foldchange of gene expression against distance using default settings.
    >>> fringe(data=data, genes=genes)

    >>> # Plot the log2foldchange of gene expression against distance with custom settings.
    >>> fringe(data=data, genes=genes, cmaps=['Reds', 'Greens'], lw=2, annotate=True, text_offset_x=0.1)

    """

    g_up, g_down = genes
    cmap_up, cmap_down = cmaps

    fig, ax = plt.subplots(**subplots_kwds)
    ax = fringe(
        data=data,
        genes=g_up,
        cmap=cmap_up,
        lw=lw,
        annotate=annotate,
        annot_bbox=[1, 0.55, 0.3, 0.45],
        text_offset_x=text_offset_x,
        text_offset_y=text_offset_y,
        text_size=text_size,
        ax=ax
    )
    fringe(
        data=data,
        genes=g_down,
        cmap=cmap_down,
        lw=lw,
        annotate=annotate,
        annot_bbox=[1, 0, 0.3, 0.45],
        text_offset_x=text_offset_x,
        text_offset_y=text_offset_y,
        text_size=text_size,
        ax=ax
    )

    ax.axvline(x=baseline_from_edge[0], linestyle='--', color='k')
    ax.axvline(x=baseline_from_edge[1], linestyle='--', color='k')

    return ax
