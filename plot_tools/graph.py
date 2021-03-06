# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     graph.py
#
# Project:  Predictive Layer for: pl_tools
# Author:   Stefano
# Date:     27 January 2018
#
# Scope:    The file contains implementation of graph plotting
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
from defines import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_scan_1d(df, what, out_dir, tag=''):

    # Filename
    fname = 'scan1d_%s%s' % (what, tag)

    # Sort
    df = df.sort_values(by='thr')
    xs = df['thr'].values
    ys = df[what].values

    # Plot
    plt.close('all')
    plt.plot(xs, ys, label=what, color=COLOR_LIST_MPL[0])
    plt.ylabel(what)
    plt.xlabel('threshold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname + '.png'))


def plot_pnl_cat(df, what, cat, out_dir, tag='', estimator=np.sum):
    # Filename
    fname = 'pnl_cat_%s_%s%s' % (what, cat, tag)

    plt.close('all')

    # Groupby
    gpb = df.groupby(cat)[what].agg(estimator)
    gpb.plot(kind='bar', color=(gpb > 0.).map({True: 'g', False: 'r'}))
    plt.ylabel('PnL Euro')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname + '.png'))
