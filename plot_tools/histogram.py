# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     histogram.py
#
# Project:  Predictive Layer for: pl_tools
# Author:   Stefano
# Date:     27 January 2018
#
# Scope:    The file contains implementation of histogram plotting
#
# Copyright (c) 2018, Predictive Layer Limited.  All Rights Reserved.
#
# The contents of this software are proprietary and confidential to the author.
# No part of this program may be photocopied,  reproduced, or translated into
# another programming language without prior written consent of the author.
#
#
#
################################################################################
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
# sns.set(style="whitegrid", color_codes=True)
from defines import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')


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
        plt.text(0.02, 0.90, 'mean = {0:.4f}'.format(desc['mean']), transform=ax.transAxes)
        plt.text(0.02, 0.85, 'std = {0:.4f}'.format(desc['std']), transform=ax.transAxes)
        plt.text(0.02, 0.80, 'min = {0:.4f}'.format(desc['min']), transform=ax.transAxes)
        plt.text(0.02, 0.75, '25% = {0:.4f}'.format(desc['25%']), transform=ax.transAxes)
        plt.text(0.02, 0.70, 'median = {0:.4f}'.format(desc['50%']), transform=ax.transAxes)
        plt.text(0.02, 0.65, '75% = {0:.4f}'.format(desc['75%']), transform=ax.transAxes)
        plt.text(0.02, 0.60, 'max = {0:.4f}'.format(desc['max']), transform=ax.transAxes)

    if kwargs.get('xlab', None) is not None:
        plt.xlabel(kwargs['xlab'])
    else:
        plt.xlabel(what)

    if kwargs.get('ylim', None):
        plt.ylim(kwargs['ylim'])

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

    if kwargs.get('xlab', None) is not None:
        plt.xlabel(kwargs['xlab'])
    else:
        plt.xlabel(plot_col)

    if kwargs.get('logy', False) is True:
        ax.set_yscale('log')

    plt.legend(loc='best')
    plt.savefig(os.path.join(out_dir, fname))


def plot_sns_class(df, plot_col, cat_col, out_dir, tag='', **kwargs):

    # Initialize the figure
    fig, ax = plt.subplots()

    # Plot options
    plot_type = kwargs.get('plot_type', 'boxplot')
    orient = kwargs.get('orient', 'h')

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


def plot_hist_period_roll(s, out_dir, tag='', group=10 * 24, func=np.sum, **kwargs):
    what = s.name
    fname = 'hist_' + what + '_group{0}{1}_roll.png'.format(group, tag)
    s = s.rolling(group, min_periods=1).apply(func)
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
        plt.text(0.02, 0.90, 'mean = {0:.4f}'.format(desc['mean']), transform=ax.transAxes)
        plt.text(0.02, 0.85, 'std = {0:.4f}'.format(desc['std']), transform=ax.transAxes)
        plt.text(0.02, 0.80, 'min = {0:.4f}'.format(desc['min']), transform=ax.transAxes)
        plt.text(0.02, 0.75, '25% = {0:.4f}'.format(desc['25%']), transform=ax.transAxes)
        plt.text(0.02, 0.70, 'median = {0:.4f}'.format(desc['50%']), transform=ax.transAxes)
        plt.text(0.02, 0.65, '75% = {0:.4f}'.format(desc['75%']), transform=ax.transAxes)
        plt.text(0.02, 0.60, 'max = {0:.4f}'.format(desc['max']), transform=ax.transAxes)

    if kwargs.get('xlab', None) is not None:
        plt.xlabel(kwargs['xlab'])
    else:
        plt.xlabel(what)

    if kwargs.get('ylim', None):
        plt.ylim(kwargs['ylim'])

    plt.savefig(os.path.join(out_dir, fname))


def plot_scan(df, what, out_dir, tag=''):

    # Retrieve information about the thresholds
    thrs = np.concatenate((df['thr'].unique(), df['thr_low'].unique()))
    thrs = sorted(list(set(thrs)))
    thr_min, thr_max = np.min(thrs), np.max(thrs)
    thr_step = np.min(np.array(thrs[1:]) - np.array(thrs[:-1]))
    n_bins = len(np.arange(thr_min, thr_max + 0.00001, thr_step))

    # Build x, y bins
    x, y = np.mgrid[slice(thr_min - thr_step/2., thr_max + thr_step/2. + 0.00001, thr_step),
                    slice(thr_min - thr_step/2., thr_max + thr_step/2. + 0.00001, thr_step)]

    z = -1.E+06 * np.ones((n_bins, n_bins))
    for irow, row in df.iterrows():
        i = int(round((row['thr'] - thr_min) / thr_step))
        j = int(round((row['thr_low'] - thr_min) / thr_step))
        z[i, j] = row[what]

    z_max = np.max(z)
    z_min = max(np.min(z[z > -1.]), 0.)

    # Initialize the figure
    fig, ax = plt.subplots()
    plt.pcolor(x, y, z, cmap='seismic', vmin=z_min, vmax=z_max)

    plt.title(df['name'][0])

    # # set the limits of the plot to the limits of the data
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    ax.set_xlabel('thr')
    ax.set_ylabel('thr_low')
    plt.colorbar()

    plt.savefig(os.path.join(out_dir, 'scan_%s%s.png' % (what, tag)))
