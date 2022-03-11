"""
plotting wrapper on matplotlib and seaborn to provide a single functional
call a la mathematica
"""
from .points import (COLORS, LINESTYLES, MARKERS, riffle_lines, get_COLORS,
                     line_plot, error_plot, dual_plot, sns_hist_plot,
                     hist_plot, scatter_plot)
from .dataframes import (pivot_timeseries, pivot_df, box_plot, point_plot,
                         dist_plot, bar_plot, df_line_plot, df_error_plot,
                         stackedbar_plot, df_bar_plot, df_pie_plot)
from .colormaps import (colorbar, contour_plot, heatmap_plot)

import matplotlib as mpl
import seaborn as sns

sns.set_style("white")
sns.set_style("ticks")
sns.set_palette('colorblind')
mpl.rcParams['font.sans-serif'] = 'DejaVu Sans'
mpl.rcParams['pdf.fonttype'] = 42


def change_tick_style(style='classic'):
    """
    Change the matplotlib style between classic and new

    Parameters
    ----------
    style : str
        style type to set up
    """
    if style == 'classic':
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        mpl.rcParams['xtick.top'] = True
        mpl.rcParams['ytick.right'] = True
    else:
        mpl.rcParams['xtick.direction'] = 'out'
        mpl.rcParams['ytick.direction'] = 'out'
        mpl.rcParams['xtick.top'] = False
        mpl.rcParams['ytick.right'] = False
