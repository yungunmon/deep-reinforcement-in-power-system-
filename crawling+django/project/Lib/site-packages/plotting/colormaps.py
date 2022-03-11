"""
Plotting of 3D arrays in 2D plots
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.ma as ma
import seaborn as sns
from plotting.base import plotting_base


def heatmap_plot(data, **kwargs):
    """
    Heat map plot using seaborn heatmap

    Parameters
    ----------
    data : ndarray | pandas.DataFrame
        ndarray of heatmap values or
        pandas DataFrame of heat map values with tick labels as index
        and column labels
    kwargs : dict
        kwargs for seaborn.heatmap and plotting_base

    See Also
    --------
    seaborn.heatmap : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, data, **kwargs):
        sns.heatmap(data, ax=axis, **kwargs)

    plotting_base(plot_func, data, legend=None, **kwargs)


def add_colorbar(axis, cf, ticks, size, padding,
                 location='right', label=None, lines=None, fontsize=14):
    """
    Add a colorbar legend to given axis

    Parameters
    ----------
    axis : matplotlib.axis
        Axis objet to add colorbar to
    cf : matplotlib.cm.ScalarMappable
        Contour set or colormap mappable to use for colors
    ticks : list
        list of tick values
    size : tuple
        colorbar size
    padding : float
        how much to pad around the colorbar
    location : str, optional
        Location of colorbar, by default 'right'
    label : str, optional
        Label for colorbar, by default None
    lines : list, optional
        list of lines to add, by default None
    fontsize : int, optional
        fontsize for label
        tick size = fontsize -2
        by default 14
    """
    divider = make_axes_locatable(axis)

    caxis = divider.append_axes(location, size=size,
                                pad=padding)

    if location in ['top', 'bottom']:
        orientation = 'horizontal'
    else:
        orientation = 'vertical'

    cbar = plt.colorbar(cf, ticks=ticks, cax=caxis,
                        orientation=orientation,
                        ticklocation=location)

    cbar.ax.tick_params(labelsize=fontsize - 2)

    if label is not None:
        cbar.set_label(label, size=fontsize)

    if lines is not None:
        cbar.add_lines(lines)


def contour_plot(data, **kwargs):
    """
    Create a contoured colormap from data shape = (n, 3)

    Parameters
    ----------
    data : ndarray
        n X 3 array of data to plot of form (x, y, c)
    figsize : tuple, optional
        Figure size, by default (8, 6)
    fontsize : int, optional
        Labels font size, by default 14
    zlim : float, optional
        z / c limit, by default None
    major_spacing : float, optional
        space between major contours, by default None
    minor_spacing : float, optional
        space between minor contours, by default None
    contour_width : int, optional
        contour line width, by default 1
    contour_color : str, optional
        contour line color, by default 'k'
    opacity : float, optional
        opacity of colormap, by default 1.
    colorbar : bool, optional
        Display color bar, by default True
    colorbar_location : str, optional
        Location of colorbar, by default 'right'
    colorbar_label : str, optional
        Colorbar label, by default None
    colorbar_lines : bool, optional
        Plot lines on colorbar, by default True
    colorbar_ticks : int, optional
       Number of colorbar ticks, by default None
    colormap : str, optional
        colormap style, by default 'jet'
    kwargs : dict
        kwargs for plotting_base

    See Also
    --------
    matplotlib.pyplot.contour : plotting function
    matplotlib.pyplot.countourf : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, data, figsize=(8, 6), fontsize=14, zlim=None,
                  major_spacing=None, minor_spacing=None, contour_width=1,
                  contour_color='k', opacity=1., colorbar=True,
                  colorbar_location='right', colorbar_label=None,
                  colorbar_lines=True, colorbar_ticks=None, colormap='jet'):

        assert len(data) == 3, 'Data must be of shape (x, y, c)'

        x, y, z = data
        z_m = ma.masked_invalid(z)

        a_ratio = z.shape
        a_ratio = a_ratio[1] / a_ratio[0]

        if isinstance(figsize, (int, float)):
            figsize = [figsize * a_ratio, figsize]
        else:
            figsize = max(figsize)
            figsize = [figsize * a_ratio, figsize]

        if zlim is None:
            zmin, zmax = np.nanmin(z), np.nanmax(z)
        else:
            zmin, zmax = zlim

        if major_spacing is None:
            major_spacing = (zmax - zmin) / 10
        if minor_spacing is None:
            minor_spacing = major_spacing / 10

        cl_levels = np.arange(zmin, zmax + major_spacing, major_spacing)
        cf_levels = np.arange(zmin, zmax + minor_spacing, minor_spacing)

        if colorbar_ticks is None:
            l_levels = cl_levels[::2]
        else:
            l_levels = (zmax - zmin) / colorbar_ticks
            l_levels = np.arange(zmin, zmax + l_levels, l_levels)

        orientation = 'vertical'
        if colorbar_location in ['top', 'bottom']:
            orientation = 'horizontal'

        cf = plt.contourf(x, y, z_m, alpha=opacity, levels=cf_levels,
                          extend='both', antialiased=True)

        if contour_color is not None:
            cl = plt.contour(cf, levels=cl_levels, colors=(contour_color,),
                             linewidths=(contour_width,))

        if colormap is not None:
            cf.set_cmap(colormap)

        if colorbar:
            cbar_padding = 0.1
            if colorbar_location in ['top', 'bottom']:
                figsize[1] += figsize[1] / 10
                cbar_size = figsize[0] / 20
            else:
                figsize[0] += figsize[0] / 10
                cbar_size = figsize[1] / 20

            divider = make_axes_locatable(axis)

            caxis = divider.append_axes(colorbar_location, size=cbar_size,
                                        pad=cbar_padding)

            cbar = plt.colorbar(cf, ticks=l_levels, cax=caxis,
                                orientation=orientation,
                                ticklocation=colorbar_location)

            cbar.ax.tick_params(labelsize=fontsize - 2)

            if colorbar_label is not None:
                cbar.set_label(colorbar_label, size=fontsize)

            if colorbar_lines is not None:
                if contour_color is not None:
                    cbar.add_lines(cl)

    plotting_base(plot_func, data, **kwargs)


def colorbar(zlim, ticks=None, lines=None, line_color='k', linewidth=1,
             colormap='jet', extend='neither', ticklocation='right',
             fontsize_other=18, label=None, fontsize_label=21, figsize=6,
             dpi=100, showfig=True, filename=None):

    """
    Create colorbar

    Parameters
    ----------
    zlim : tuple
        List or tuple indicating zmin and zmax.
    tick : int
        Number of ticks to label.
    lines : int
        Number of lines to draw on colorbar.
    line_color : str
        Color of lines drawn on colorbar.
    linewidth : int
        Line width for each line drawn on colorbar.
    colormap : str
        Color scheme for colorbar.
    extend : str
        Direction to extend colors beyond zmin and zmax.
    ticklocation : str
        Orientation of colorbar and location of tick marks.
    fontsize_other : int
        Font size of tick numbers.
    label : str
        Label for colorbar
    fontsize_label : int
        Font size of label.
    figsize : tuple
        Width and height of figure
    dpi : int
        DPI resolution of figure.
    showfig : bool
        Whether to show figure.
    filename : str
        Name of file/path to save the figure to.
    """

    a_ratio = 20

    if isinstance(figsize, (list, tuple)):
        figsize = max(figsize)

    if ticklocation in ['right', 'left']:
        figsize = (figsize / a_ratio, figsize)
        orientation = 'vertical'
    else:
        figsize = (figsize, figsize / a_ratio)
        orientation = 'horizontal'

    if ticks is not None:
        ticks = (zlim[1] - zlim[0]) / ticks
        ticks = np.arange(zlim[0], zlim[1] + ticks, ticks)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    axis = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    norm = mpl.colors.Normalize(vmin=zlim[0], vmax=zlim[1])

    cb = mpl.colorbar.ColorbarBase(axis, cmap=colormap, norm=norm,
                                   orientation=orientation, extend=extend,
                                   ticks=ticks, ticklocation=ticklocation)
    cb.ax.tick_params(labelsize=fontsize_other)

    if label is not None:
        cb.set_label(label, size=fontsize_label)

    if lines is not None:
        lines = (zlim[1] - zlim[0]) / lines
        lines = np.arange(zlim[0], zlim[1] + lines, lines)
        cb.add_lines(lines, colors=(line_color,) * len(lines),
                     linewidths=(linewidth,) * len(lines))

    if filename is not None:
        plt.savefig(filename, dpi=dpi, transparent=True,
                    bbox_inches='tight')

    if showfig:
        plt.show()

    plt.close()
