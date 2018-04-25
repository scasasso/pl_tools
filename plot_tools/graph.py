import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from defines import *


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
