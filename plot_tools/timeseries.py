# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     timeseries.py
#
# Project:  Predictive Layer for: pl_tools
# Author:   Stefano
# Date:     27 January 2018
#
# Scope:    The file contains implementation of timeseries plotting
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
import numpy as np
import plotly.graph_objs as go
import plotly
from defines import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_ts_mpl(x, y, title=None, ylab=None, out_dir='.', color='blue', filename=None, annotate=None, ylim=None):

    # Output file
    if filename is None and title is None:
        raise ValueError('Do not know the name of the output file')
    elif filename is None:
        filename = title

    # All the things to plot
    if not isinstance(y, list):
        y = [y]
    if not isinstance(color, list):
        color = COLOR_LIST_MPL[:len(y)]
    if not isinstance(title, list):
        title = [title] * len(y)

    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ay, col, tit in zip(y, color, title):
        plt.plot(x, ay, label=tit, color=col)
    if ylab is not None:
        plt.ylabel(ylab)
    plt.legend(loc='upper left')
    fig.autofmt_xdate()
    if annotate is not None:
        if not isinstance(annotate, list):
            annotate = [annotate]
        for i, (txt, col) in enumerate(zip(annotate, color)):
            plt.text(0.02, 0.75 - 0.02 * max(0, len(annotate) - 1) - i * 0.05, txt, transform=ax.transAxes, color=col)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename + '.png'))


def plot_ts_ply(x, y, title, ylab=None, out_dir='.', color='blue', filename=None, annotate=None, ylim=None):

    # Output file
    if filename is None:
        filename = title

    if annotate is None:
        annotate = ''

    if ylab is None:
        ylab = ''

    # All the things to plot
    if not isinstance(y, list):
        y = [y]
    if not isinstance(color, list):
        color = COLOR_LIST_PLY[:len(y)]
    if not isinstance(title, list):
        title = [title] * len(y)

    layout = dict(
        title=annotate,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(),
            type='date'
        ),
        yaxis=dict(
            title=ylab,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            ),
        ),
    )

    if ylim is not None:
        layout['yaxis']['range'] = list(ylim)

    data = []
    for ay, atitle, acolor in zip(y, title, color):
        points = go.Scatter(x=x, y=ay, name=atitle, line=dict(color=acolor), opacity=0.8)
        data.append(points)

    plotly_fig = dict(data=data, layout=layout)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    plotly.offline.plot(plotly_fig, filename=os.path.join(out_dir, "{}.html".format(filename)), auto_open=False)
