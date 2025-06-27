import numpy as np
from matplotlib import pyplot as plt
from thor.utils import get_ROI_tuple_from_polygon

def annotate_ROI(im, ROI_polygon=None, baseline_polygon=None, lw=3):
    """Annotate the ROI and baseline on the image.

    Parameters
    ----------
    im : :class:`numpy.ndarray`
        The image to be annotated.
    ROI_polygon : :class:`shapely.geometry.Polygon`
        The ROI polygon.
    baseline_polygon : :class:`shapely.geometry.Polygon`
        The baseline polygon.
    lw : :py:class:`float`
        The line width of the annotation.

    """
    ROI_ex = np.array(ROI_polygon.exterior.coords.xy).T

    colors = ['deepskyblue', 'tomato']
    ROI_tuple = get_ROI_tuple_from_polygon(ROI_ex)
    l, b, w, h = ROI_tuple

    plt.imshow(im[b:b + h + 1, l:l + w + 1])
    plt.plot(
        ROI_ex[:, 0] - l,
        ROI_ex[:, 1] - b,
        c=colors[1],
        linestyle='--',
        dashes=(3, 1),
        linewidth=lw
    )

    if baseline_polygon is not None:
        baseline_ex = np.array(baseline_polygon.exterior.coords.xy).T
        plt.plot(
            baseline_ex[:, 0] - l,
            baseline_ex[:, 1] - b,
            c=colors[0],
            linestyle='--',
            dashes=(3, 1),
            linewidth=lw
        )
    plt.axis('off')
    plt.show()
