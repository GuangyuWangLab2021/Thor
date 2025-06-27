## The code does not really work as expected. The spot size is not scaled correctly when changing figure size.

import logging

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)

from thor.utils import get_spot_diameter_in_pixels, get_scalefactors


def spot_over(
    ad,
    ad_spot,
    spot_scale=1,
    figsize=None,
    spot_cmap='viridis',
    spot_color_obs_key=None,
    spot_facecolor=None,
    spot_alpha=0.5,
    spot_linewidth=0.5,
    **sc_kwargs
):
    """Plot spatial expression data with spots on top.

    Parameters
    ----------
    ad : :class:`anndata.AnnData`
        Annotated data matrix at the cell level.
    ad_spot : :class:`anndata.AnnData`
        Annotated data matrix at the spot level.
    spot_scale : :py:class:`float`, optional (default: 1)
        Scale factor for spot size.
    figsize : :py:class:`tuple`, optional (default: :py:obj:`None`)
        Figure size in full resolution.
    spot_cmap : :py:class:`str`, optional (default: 'viridis')
        Colormap for spot colors.
    spot_color_obs_key : :py:class:`str`, optional (default: :py:obj:`None`)
        Key for observation in `ad_spot` that contains spot colors.
    spot_facecolor : :py:class:`str`, optional (default: :py:obj:`None`)
        Face color for spots.
    spot_alpha : :py:class:`float`, optional (default: 0.5)
        Alpha value for spots.
    spot_linewidth : :py:class:`float`, optional (default: 0.5)
        Line width for spots.
    **sc_kwargs : keyword arguments
        Keyword arguments for :func:`~scanpy.pl.spatial`.

    Returns
    -------
    Depending on scanpy settings returns axes or list of axes.
    """

    fig, ax = plt.subplots(figsize=figsize)

    if 'ax' in sc_kwargs:
        sc_kwargs.update({'ax': ax})

    basis = sc_kwargs.get('basis', 'spatial')
    library_id = sc_kwargs.get('library_id', None)
    img_key = sc_kwargs.get('img_key', 'hires')

    d = get_spot_diameter_in_pixels(ad_spot, library_id=library_id)
    r = 0.5 * d * spot_scale

    scalef = get_scalefactors(ad_spot, library_id=library_id)[f"tissue_{img_key}_scalef"]
    spotpos = ad_spot.obsm[basis] * scalef

    try:
        z = ad_spot.obs[spot_color_obs_key]
    except KeyError:
        z = np.ones(ad_spot.shape[0])
    norm = colors.Normalize(z.min(), z.max())
    cmap = mpl.colormaps[spot_cmap]
    ec = cmap(norm(z))

    ax.scatter(
        x=spotpos[:, 0],
        y=spotpos[:, 1],
        facecolor=spot_facecolor,
        edgecolors=ec,
        s=r,
        zorder=100,
        lw=spot_linewidth,
        alpha=spot_alpha
    )

    return sc.pl.spatial(ad, ax=ax, **sc_kwargs)


def spot_over_fig(
    fig,
    ad_spot,
    spot_scale=1,
    spot_cmap='viridis',
    spot_color_obs_key=None,
    spot_facecolor=None,
    spot_alpha=0.5,
    spot_linewidth=0.5,
    offset_x=0,
    offset_y=0,
    **kwargs
):
    """Plot spatial expression data with spots on top.

    Parameters
    ----------
    ad_spot : :class:`anndata.AnnData`
        Annotated data matrix at the spot level.
    spot_scale : :py:class:`float`, optional (default: 1)
        Scale factor for spot size.
    spot_cmap : :py:class:`str`, optional (default: 'viridis')
        Colormap for spot colors.
    spot_color_obs_key : :py:class:`str`, optional (default: :py:obj:`None`)
        Key for observation in `ad_spot` that contains spot colors.
    spot_facecolor : :py:class:`str`, optional (default: :py:obj:`None`)
        Face color for spots.
    spot_alpha : :py:class:`float`, optional (default: 0.5)
        Alpha value for spots.
    spot_linewidth : :py:class:`float`, optional (default: 0.5)
        Line width for spots.
    **kwargs : keyword arguments
        Keyword arguments. 

    """

    ax = fig.gca()

    basis = kwargs.get('basis', 'spatial')
    library_id = kwargs.get('library_id', None)
    img_key = kwargs.get('img_key', 'hires')

    d = get_spot_diameter_in_pixels(ad_spot, library_id=library_id)
    s = d * spot_scale

    try:
        scalef = get_scalefactors(ad_spot, library_id=library_id)[f"tissue_{img_key}_scalef"]
    except KeyError:
        scalef = 1
    spotpos = ad_spot.obsm[basis] - np.array([offset_x, offset_y])
    spotpos = spotpos * scalef

    try:
        z = ad_spot.obs[spot_color_obs_key]
    except KeyError:
        z = np.ones(ad_spot.shape[0])

    norm = colors.Normalize(z.min(), z.max())
    cmap = mpl.colormaps[spot_cmap]
    ec = cmap(norm(z))

    scatter(
        spotpos[:, 0],
        spotpos[:, 1],
        ax,
        size=s,
        facecolor=spot_facecolor,
        edgecolors=ec,
        zorder=100,
        lw=spot_linewidth,
        alpha=spot_alpha
    )

    return fig, ax



# source: https://stackoverflow.com/a/48174228/4124317
class scatter():
    def __init__(self,x,y,ax,size=1,**kwargs):
        self.n = len(x)
        self.ax = ax
        self.ax.figure.canvas.draw()
        self.size_data=size
        self.size = size
        self.sc = ax.scatter(x,y,s=self.size,**kwargs)
        self._resize()
        self.cid = ax.figure.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self,event=None):
        ppd=72./self.ax.figure.dpi
        trans = self.ax.transData.transform
        s =  ((trans((1,self.size_data))-trans((0,0)))*ppd)[1]
        if s != self.size:
            self.sc.set_sizes(s**2*np.ones(self.n))
            self.size = s
            self._redraw_later()
    
    def _redraw_later(self):
        self.timer = self.ax.figure.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.ax.figure.canvas.draw_idle())
        self.timer.start()
