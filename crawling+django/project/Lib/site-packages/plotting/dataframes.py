"""
Plotting dataframe data with seaborn and pandas
"""
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plotting.base import plotting_base


def pivot_timeseries(df, var_name, timezone=None):
    """
    Pivot timeseries DataFrame and shift UTC by given timezone offset

    Parameters
    ----------
    df : pandas.DataFrame
        Timeseries DataFrame to be pivoted with year, month, hour columns
    var_name : str
        Name for new column describing data
    timezone : int, optional
        UTC offset to apply to DatetimeIndex, by default None

    Returns
    -------
    pandas.DataFrame
        Seaborn style long table with source, year, month, hour columns
    """
    sns_df = []
    for name, col in df.iteritems():
        col = col.to_frame()
        col.columns = [var_name]
        col['source'] = name
        col['year'] = col.index.year
        col['month'] = col.index.month
        col['hour'] = col.index.hour
        if timezone is not None:
            td = pd.to_timedelta('{:}h'.format(timezone))
            col['local_hour'] = (col.index + td).hour

        sns_df.append(col)

    return pd.concat(sns_df)


def pivot_df(df, var_name):
    """
    Pivot DataFrame converting columns to source and data to var_name

    Parameters
    ----------
    df : pandas.DataFrame
        Source DataFrame to convert to Seaborn long style
    var_name : str
        Column name to use for data in final DataFrame

    Returns
    -------
    pandas.DataFrame
        Seaborn long style DataFrame
    """
    sns_df = []
    for name, col in df.iteritems():
        col = col.to_frame()
        col.columns = [var_name]
        col['source'] = name
        sns_df.append(col)

    return pd.concat(sns_df)


def box_plot(df, **kwargs):
    """
    Box plot based on seaborns boxplot

    Parameters
    ----------
    df : pandas.DataFrame
        Seaborn compliant (long style) DataFrame
    kwargs : dict
        kwargs for seaborn.boxplot and plotting_base

    See Also
    --------
    seaborn.boxplot : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, df, **kwargs):
        meanprops = dict(marker='o', markeredgecolor='black',
                         markerfacecolor="None", markersize=5)
        sns.boxplot(data=df, ax=axis, meanprops=meanprops, **kwargs)

    plotting_base(plot_func, df, **kwargs)


def dist_plot(df, fit=False, **kwargs):
    """
    Distribution plot based on seaborn distplot

    Parameters
    ----------
    df : pandas.DataFrame
        Seaborn compliant (long style) DataFrame
    fit : bool
        Fit the distribution
    kwargs : dict
        kwargs for seaborn.distplot and plotting_base

    See Also
    --------
    seaborn.distplot : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, df, fit=None, **kwargs):
        palette = itertools.cycle(sns.color_palette())
        if isinstance(df, pd.DataFrame):
            for label, series in df. iteritems():
                if fit:
                    sns.distplot(series, kde=False,
                                 fit_kws={"color": next(palette)},
                                 label=label, ax=axis,
                                 **kwargs)
                else:
                    sns.distplot(series, label=label, ax=axis,
                                 **kwargs)
        else:
            if fit:
                sns.distplot(df, kde=False,
                             fit_kws={"color": next(palette)},
                             ax=axis, **kwargs)
            else:
                sns.distplot(df, ax=axis, **kwargs)

    plotting_base(plot_func, df, fit=fit, **kwargs)


def point_plot(df, **kwargs):
    """
    Point / line plot based on seaborn pointplot

    Parameters
    ----------
    df : pandas.DataFrame
        Seaborn compliant (long style) DataFrame
    kwargs : dict
        kwargs for seaborn.pointplot and plotting_base

    See Also
    --------
    seaborn.pointplot : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, df, **kwargs):
        sns.pointplot(data=df, ax=axis, **kwargs)

    plotting_base(plot_func, df, **kwargs)


def bar_plot(df, kind='bar', **kwargs):
    """
    Bar plot based on seaborn's catplot

    Parameters
    ----------
    df : pandas.DataFrame
        Seaborn compliant (long style) DataFrame
    kind : str
        kind of plot to use "count" or "bar"
    kwargs : dict
        kwargs for seaborn.barplot and plotting_base

    See Also
    --------
    seaborn.catplot : plotting function
    seaborn.barplot : plotting function
    seaborn.countplot : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, df, kind='bar', **kwargs):
        if kind == 'bar':
            sns.barplot(data=df, ax=axis, **kwargs)
        elif kind == 'count':
            sns.countplot(data=df, ax=axis, **kwargs)
        else:
            raise ValueError('kind must be "count" or "bar"')

    plotting_base(plot_func, df, kind=kind, **kwargs)


def df_bar_plot(df, **kwargs):
    """
    Bar plot based on pandas bar plot

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to plot
    kwargs : dict
        kwargs for pandas.DataFrame.plot

    See Also
    --------
    pandas.DataFrame.plot

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, df, **kwargs):
        df.plot(kind='bar', ax=axis, **kwargs)

    plotting_base(plot_func, df, **kwargs)


def stackedbar_plot(df, x, y, stack, **kwargs):
    """
    Bar plot based on seaborn's catplot

    Parameters
    ----------
    df : pandas.DataFrame
        Seaborn compliant (long style) DataFrame
    x : str
        Column to use for x-axis
    y : str
        Column to use for y-axis
    stack : str
        Column to stack
    order : list
        Stacking order
    kwargs : dict
        kwargs for pandas.DataFrame.plot and plotting_base

    See Also
    --------
    pandas.DataFrame.plot : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, df, x, y, stack, order=None, **kwargs):
        df = df.pivot(index=x, values=y, columns=stack)
        if order is not None:
            df = df[order]

        df.plot(kind='bar', stacked=True, ax=axis, **kwargs)

    plotting_base(plot_func, df, x, y, stack, **kwargs)


def df_scatter(df, colorbar=False, **kwargs):
    """
    scatter plot based on pandas.plot.scatter

    Parameters
    ----------
    df : pandas.DataFrame
        Seaborn compliant (long style) DataFrame
    colorbar : bool, optional
        Flag to add colorbar, by default False
    kwargs : dict
        kwargs for pandas.DataFrame.plot.scatter and plotting_base

    See Also
    --------
    pandas.DataFrame.plot.scatter : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, df, colorbar=False, **kwargs):
        c = df.plot.scatter(ax=axis, **kwargs)
        if colorbar:
            plt.colorbar(c)

    plotting_base(plot_func, df, colorbar=colorbar, **kwargs)


def df_line_plot(df, **kwargs):
    """
    point / line plot based on pandas plot

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of data to plot
    kwargs : dict
        kwargs for pandas.DataFrame.plot and plotting_base

    See Also
    --------
    pandas.DataFrame.plot : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, df, **kwargs):
        df.plot(ax=axis, **kwargs)

    plotting_base(plot_func, df, **kwargs)


def df_error_plot(df, error, **kwargs):
    """
    point / line plot with error bars based on pandas plot

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of data to plot
    error : pandas.DataFrame
        Error values for data values
    kwargs : dict
        kwargs for pandas.DataFrame.plot and plotting_base

    See Also
    --------
    pandas.DataFrame.plot : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, df, error, **kwargs):
        error.columns = df.columns
        error.index = df.index
        df.plot(yerr=error, ax=axis, **kwargs)

    plotting_base(plot_func, df, error, **kwargs)


def df_pie_plot(df, **kwargs):
    """
    Pie chart using pandas plot

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of data to plot
    kwargs : dict
        kwargs for pandas.DataFrame.plot and plotting_base

    See Also
    --------
    pandas.DataFrame.plot : plotting function

    plotting.base.plotting_base : plotting base
    """
    def plot_func(axis, df, **kwargs):
        df.plot.pie(ax=axis, **kwargs)

    plotting_base(plot_func, df, **kwargs)
