# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     regression.py
#
# Project:  Predictive Layer for: pl_tools
# Author:   Stefano
# Date:     27 January 2018
#
# Scope:    The file contains regression evaluation scripts
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
import logging
from plot_tools.timeseries import plot_ts_mpl, plot_ts_ply
from plot_tools.histogram import plot_hist_period

logger = logging.getLogger(__file__)


REGRESSION_METRICS = ['RMSE', 'MAE', 'MAPE', 'R2']


class RegressionValidator(object):
    def __init__(self, target_coll, pred_coll, round=3):
        self.target_coll = target_coll
        if not isinstance(pred_coll, list):
            self.pred_colls = [pred_coll]
        else:
            self.pred_colls = pred_coll

        # Default initialization
        self.df_val = pd.DataFrame()

        # Rounding
        self.round = round

    def _fetch(self, dt_start, dt_end):
        # Target
        if isinstance(self.target_coll, pd.core.series.Series) or isinstance(self.target_coll, pd.core.frame.DataFrame):
            _df = pd.DataFrame(self.target_coll)
            df_target = _df.rename({_df.columns[0]: 'target'}, axis=1)
        else:
            df_target = self.target_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='target')

        df_preds = []
        for pred in self.pred_colls:
            # Predictions
            if isinstance(pred, pd.core.series.Series) or isinstance(pred, pd.core.frame.DataFrame):
                _df = pd.DataFrame(pred)
                df_preds.append(_df.rename({_df.columns[0]: 'pred_' + _df.columns[0]}, axis=1))
            else:
                df_preds.append(pred.get_data(dt_start=dt_start, dt_end=dt_end, rename='pred_' + pred.name))

        # Concatenate
        df = pd.concat([df_target] + df_preds, axis=1).round(self.round)

        return df.dropna(how='any')

    def produce_validation(self, dt_start, dt_end, skip_dates=None):
        # Default
        if skip_dates is None:
            skip_dates = []
        for i, d in enumerate(skip_dates):
            try:
                skip_dates[i] = pd.Timestamp(d)
            except Exception as e:
                msg = 'Problem in converting datetime %s, got following exception:\n%s' % (d, str(e))
                logger.error(msg)
                raise ValueError(msg)

        # Fetch the data
        self.df_val = self._fetch(dt_start, dt_end)

        # Eventually drop dates
        try:
            self.df_val = self.df_val.drop(skip_dates)
        except ValueError:
            logger.warning('You are trying to skip dates which are not in index')
            pass

        if len(self.pred_colls) > 1:
            # Add the blend
            self.df_val['pred_blend'] = self.df_val.filter(regex="pred_.*").mean(axis=1).round(self.round)

        # Add the regression metrics
        for col in self.df_val.columns:
            if col.startswith('pred') is not True:
                continue
            self._add_metrics(col)

    def _add_metrics(self, col):

        tag = col.replace('pred_', '')
        self.df_val['AE_' + tag] = np.abs(self.df_val['target'] - self.df_val[col]).round(self.round)
        self.df_val['APE_' + tag] = (self.df_val['AE_' + tag] / self.df_val['target']).round(4)
        self.df_val['MSE_' + tag] = (self.df_val['target'] - self.df_val[col]).pow(2).round(self.round)
        self.df_val['SSres_' + tag] = self.df_val['MSE_' + tag].cumsum().round(self.round)
        self.df_val['cum_mean'] = self.df_val['target'].expanding(min_periods=1).mean().round(self.round)
        self.df_val['SStot_' + tag] = (self.df_val['target'] - self.df_val['cum_mean']).pow(2).cumsum().round(self.round)

        self.df_val['MAE_' + tag] = self.df_val['AE_' + tag].expanding(min_periods=1).mean().round(self.round)
        self.df_val['RMSE_' + tag] = np.sqrt(self.df_val['MSE_' + tag].expanding(min_periods=1).mean()).round(self.round)
        self.df_val['R2_' + tag] = (1. - self.df_val['SSres_' + tag] / self.df_val['SStot_' + tag]).replace([np.inf, -np.inf], 0.).round(self.round)
        self.df_val['MAPE_' + tag] = self.df_val['APE_' + tag].expanding(min_periods=1).mean().round(4)

        self.df_val = self.df_val.drop(['SSres_' + tag, 'SStot_' + tag, 'cum_mean'], axis=1)

    def dump(self, out_dir):
        self.df_val.to_csv(os.path.join(out_dir, 'validation.csv'))


class RegressionPlotter(object):
    def __init__(self, df, out_dir, labels=None, plot_blend=True):

        # Data can come as Dataframe directly or as csv file
        if isinstance(df, pd.core.frame.DataFrame):
            self.df_val = df.copy()
        else:
            self.df_val = pd.read_csv(df, index_col=0, parse_dates=[0])

        # Sanity check
        if not ('target' in self.df_val.columns and len(self.df_val.filter(regex="pred_.*").columns) >= 1):
            msg = 'A Dataframe with a \"target\" columns with at least one \""pred_X\"" column is expected'
            logger.error(msg)
            raise KeyError(msg)

        # Re-order columns
        self.df_val.columns = ['target'] + [c for c in self.df_val.columns if c != 'target']

        # Output directory
        self.out_dir = out_dir

        # To blend or not to blend?
        self.n_models = len([c for c in self.df_val.columns if c.startswith('pred_') and not (c == 'pred_blend' and plot_blend is False)])
        if plot_blend is False or self.n_models <= 1:
            self.df_val = self.df_val.drop([c for c in self.df_val if c.endswith('_blend')], axis=1)

        # Labels
        if labels is None:
            self.labels = [col.replace('pred_', '') for col in self.df_val.columns if col.startswith('pred_') and not (c == 'pred_blend' and plot_blend is False)]
        else:
            self.labels = labels
        if len(self.labels) != self.n_models:
            msg = 'Something has gone wrong in creating labels: ' \
                  'found {0} and expected {1}:\n{2}'.format(len(self.labels), self.n_models, self.labels)
            logger.error(msg)
            raise ValueError(msg)

    def plot_ts_smooth(self, what, smooth=None, ylab=None, backend='plotly', **kwargs):

        # Datetimes
        xs = self.df_val.index

        fname = 'ts_' + what
        if smooth is not None:
            fname += '_smooth' + str(smooth)

        ys, titles = [], []
        if 'pred' in what:
            ys.append(self.df_val['target'])
            titles.append('target')

        cols_to_plot = [c for c in self.df_val.columns if c.startswith(what + '_')]
        if len(cols_to_plot) != len(self.labels):
            msg = 'There\'s a mismatch between the number of objects to plot ({0}) ' \
                  'and the number of labels ({1})'.format(len(cols_to_plot), len(self.labels))
            logger.error(msg)
            raise ValueError(msg)
        for col, label in zip(cols_to_plot, self.labels):
            if col.startswith(what) is not True or col == 'target':
                continue
            titles.append(label)
            if smooth is not None:
                ys.append(self.df_val[col].rolling(window=smooth, min_periods=1).mean().values)
            else:
                ys.append(self.df_val[col].values)

        if backend == 'matplotlib':
            plot_ts_mpl(xs, ys, title=titles, ylab=ylab,
                        out_dir=self.out_dir, filename=fname, **kwargs)
        else:
            plot_ts_ply(xs, ys, title=titles, ylab=ylab,
                        out_dir=self.out_dir, filename=fname, **kwargs)

    def plot_hist_inperiod(self, what, group='1D', func=np.mean, **kwargs):
        cols = [c for c in self.df_val.columns if c.startswith(what)]
        for col, label in zip(cols, self.labels):
            plot_hist_period(self.df_val[col], self.out_dir, tag='_' + label, group=group, func=func, **kwargs)
