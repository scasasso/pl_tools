import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="whitegrid", color_codes=True)
from defines import *


def plot_hist_period(s, out_dir, tag='', group='1D', func=np.sum, **kwargs):
    what = s.name
    fname = 'hist_' + what + '_group%s%s.png' % (group, tag)
    s = s.groupby(pd.Grouper(freq=group)).agg(func)
    a = s.values

    fig, ax = plt.subplots()
    x_bins = kwargs.get('x_bins', None)
    if x_bins is None:
        x_min, x_max = min(a), max(a)
        x_bins = np.linspace(x_min, x_max, 100)

    ax.hist(a, alpha=0.7, histtype='step', bins=x_bins, color='blue', fill=True)

    if kwargs.get('add_desc', False) is True:

        desc = s.describe()
        plt.text(0.02, 0.95, 'count = {0}'.format(int(desc['count'])), transform=ax.transAxes)
        plt.text(0.02, 0.90, 'mean = {0:.2f}'.format(desc['mean']), transform=ax.transAxes)
        plt.text(0.02, 0.85, 'std = {0:.2f}'.format(desc['std']), transform=ax.transAxes)
        plt.text(0.02, 0.80, 'min = {0:.2f}'.format(desc['min']), transform=ax.transAxes)
        plt.text(0.02, 0.75, '25% = {0:.2f}'.format(desc['25%']), transform=ax.transAxes)
        plt.text(0.02, 0.70, 'median = {0:.2f}'.format(desc['50%']), transform=ax.transAxes)
        plt.text(0.02, 0.65, '75% = {0:.2f}'.format(desc['75%']), transform=ax.transAxes)
        plt.text(0.02, 0.60, 'max = {0:.2f}'.format(desc['max']), transform=ax.transAxes)

    if kwargs.get('xlab', None) is not None:
        plt.xlabel(kwargs['xlab'])
    else:
        plt.xlabel(what)

    plt.savefig(os.path.join(out_dir, fname))


def plot_hist_class(df, plot_col, cat_col, out_dir, tag='', **kwargs):

    fname = 'hist_%s_%s_classes%s.png' % (plot_col, cat_col, tag)

    fig, ax = plt.subplots()

    x_bins = kwargs.get('x_bins', None)
    if x_bins is None:
        x_min, x_max = df[cat_col].min(), df[cat_col].max()
        x_bins = np.linspace(x_min, x_max, 100)

    for i, (cat, df_cat) in enumerate(df.groupby(cat_col)):
        a = df_cat[plot_col].values
        cat_str = cat_col + ' = ' + str(cat)
        ax.hist(a, alpha=0.3, histtype='step', bins=x_bins, color=COLOR_LIST_MPL[i], fill=True, label=cat_str)

    # if kwargs.get('add_desc', False) is True:
    #     desc = s.describe()
    #     plt.text(0.02, 0.95, 'count = {0}'.format(int(desc['count'])), transform=ax.transAxes)
    #     plt.text(0.02, 0.90, 'mean = {0:.2f}'.format(desc['mean']), transform=ax.transAxes)
    #     plt.text(0.02, 0.85, 'std = {0:.2f}'.format(desc['std']), transform=ax.transAxes)
    #     plt.text(0.02, 0.80, 'min = {0:.2f}'.format(desc['min']), transform=ax.transAxes)
    #     plt.text(0.02, 0.75, '25% = {0:.2f}'.format(desc['25%']), transform=ax.transAxes)
    #     plt.text(0.02, 0.70, 'median = {0:.2f}'.format(desc['50%']), transform=ax.transAxes)
    #     plt.text(0.02, 0.65, '75% = {0:.2f}'.format(desc['75%']), transform=ax.transAxes)
    #     plt.text(0.02, 0.60, 'max = {0:.2f}'.format(desc['max']), transform=ax.transAxes)

    if kwargs.get('xlab', None) is not None:
        plt.xlabel(kwargs['xlab'])
    else:
        plt.xlabel(plot_col)

    if kwargs.get('logy', False) is True:
        ax.set_yscale('log')

    plt.legend(loc='best')
    plt.savefig(os.path.join(out_dir, fname))


def plot_sns_class(df, plot_col, cat_col, out_dir, tag='', group=None, func=np.sum, **kwargs):

    # Initialize the figure
    fig, ax = plt.subplots()

    # Plot options
    plot_type = kwargs.get('plot_type', 'boxplot')
    orient = kwargs.get('orient', 'h')

    # # Aggregation
    # if group is not None:
    #     df = df.groupby(pd.Grouper(freq=group)).agg({plot_col: func, cat_col: np.mean}).reset_index()
    #     tag = '_group' + group + tag

    # File name
    fname = '%s_%s_%s_classes%s.png' % (plot_type, plot_col, cat_col, tag)

    # Plot
    if orient == 'h':
        getattr(sns, plot_type)(y=cat_col, x=plot_col, data=df, orient='h')
    else:
        getattr(sns, plot_type)(x=cat_col, y=plot_col, data=df, orient='v')

    # Axis limits
    if kwargs.get('xlim', None) is not None:
        xlim = kwargs['xlim']
        ax.set_xlim(xlim[0], xlim[1])

    plt.savefig(os.path.join(out_dir, fname))
