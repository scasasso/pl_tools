# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     market_tendency.py
#
# Project:  Predictive Layer for: pl_tools
# Author:   Stefano
# Date:     27 January 2018
#
# Scope:    The file contains market_tendency evaluation scripts
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
import pytz
import os
import gc
import logging
import pandas as pd
import numpy as np
from plot_tools.timeseries import plot_ts_mpl
from plot_tools.histogram import plot_hist_period, plot_sns_class, plot_hist_class, plot_scan
from plot_tools.graph import plot_scan_1d
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__file__)

# Fir the scan
DEFAULT_THR_LIST_SCAN = np.round(np.arange(0.35, 0.65001, 0.01), 2)

# Metrics
METRICS = ['accuracy', 'accuracy_h', 'gain_cum', 'gain_per_pos']


class MarketTendencyValidator(object):
    def __init__(self, name, da_coll, pos_coll, score_coll, neg_coll=None, timezone=None, na_strategy='drop'):
        self.name = name
        self.da_coll = da_coll
        self.pos_coll = pos_coll
        self.score_coll = score_coll
        self.timezone = timezone
        if na_strategy not in ['drop', 'ffill']:
            raise ValueError('Unknown strategy for NaNs: %s' % str(na_strategy))
        self.na_strategy = na_strategy

        if neg_coll is None:
            self.neg_coll = self.pos_coll
        else:
            self.neg_coll = neg_coll

    def _fetch(self, dt_start, dt_end):
        # Positive price
        df_pos = self.pos_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='positive_price', tz=self.timezone, verbose=0)

        # Negative price
        df_neg = self.neg_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='negative_price', tz=self.timezone, verbose=0)

        # Day-ahead price
        df_da = self.da_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='dayahead_price', tz=self.timezone, verbose=0)
        df_da = df_da.reindex(index=df_pos.index, method='ffill')

        # Scores
        df_scores = self.score_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='prob', tz=self.timezone, verbose=0)

        # Concatenate
        df = pd.concat([df_da, df_pos, df_neg, df_scores], axis=1)

        # NaNs
        if self.na_strategy == 'ffill':
            df = df.fillna(method='ffill')
        elif self.na_strategy == 'drop':
            df = df.dropna(how='any')

        return df

    def produce_validation(self, dt_start, dt_end, thr, thr_low=None, agg_pred=None, use_avg_price=True,
                           skip_dates=None, hourly_stats=True):
        # Default
        if thr_low is None:
            thr_low = thr

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

        def default_agg(ps):
            return default_aggregator(ps, thr, thr_low)

        if agg_pred is None:
            agg_pred = default_agg

        # Fetch the data
        df_input = self._fetch(dt_start, dt_end)
        df_input['name'] = self.name

        # Eventually drop dates
        if len(skip_dates) > 0:
            try:
                skip_dates = [d.tz_localize(self.timezone) for d in skip_dates]
                df_input = df_input.drop(skip_dates)
            except ValueError:
                logger.warning('You are trying to skip dates which are not in index')
                pass
            except pytz.exceptions.AmbiguousTimeError as e:
                logger.error('Ambiguity in skip dates (probably due to DST)')
                pass

        # Input collections
        df_val = df_input.copy()

        if 'market_tendency' not in df_val.columns:
            # Add the market tendency
            df_val = add_market_tendency(df_val)

        # Get hour aggregates
        df_val['threshold'] = thr
        df_val['threshold_low'] = thr_low
        df_val = add_agg_positions(df_val, agg_pred, self.da_coll.freq)

        # Compute the cost/gain
        if 'imbalance_price' not in df_val.columns:
            if use_avg_price is True:
                df_val['imbalance_price'] = df_val[['positive_price', 'negative_price']].mean(axis=1)
            else:  # if pred == 0 we don't care about the imbalance price anyway
                df_val['imbalance_price'] = df_val.apply(lambda x: x['positive_price'] if x['pl_pred'] > 0 else x['negative_price'] if x['pl_pred'] < 0 else x.get('dayahead_price', np.nan), axis=1)
        if 'price_diff' not in df_val.columns:
            df_val['price_diff'] = (df_val['imbalance_price'] - df_val['dayahead_price']).round(3)
        df_val = compute_perfomances(df_val)

        if hourly_stats:
            # Get aggregate hourly statistics
            df_h = get_hourly_stats(df_val)

            # Add to the original DataFrame
            df_val['accuracy_h'] = df_h['accuracy'].reindex(index=df_val.index, method='ffill')

            del df_h
            gc.collect()

        return df_val.copy()

    def _scan(self, eval_metric, min_frac_pos=0., thr_list=None, scan2d=True, thr_list_low=None, **kwargs):
        # List of thresholds to loop over
        if thr_list is None:
            thr_list = DEFAULT_THR_LIST_SCAN
        if thr_list_low is None:
            thr_list_low = list(thr_list)

        # Loop function
        default_pod = {'name': 'unknown', 'frac_pos': 0.}
        for metr in METRICS:
            default_pod[metr] = -1.E+06

        df_base = self.produce_validation(thr=0.5, thr_low=0.5, **kwargs)
        freq, gran = get_freq_from_df(df_base)
        da_freq = self.da_coll.freq
        keep_cols = ['dayahead_price', 'positive_price', 'negative_price', 'prob', 'imbalance_price', 'price_diff',
                     'price_diff_pos', 'price_diff_neg', 'market_tendency', 'name']
        default_pod['name'] = df_base['name'][0]

        def inspect(t, tlo):
            pod = dict(default_pod)
            pod.update({'thr': t, 'thr_low': tlo})
            try:
                # Start from base
                _df_val = df_base[keep_cols].copy()
                # Add positions
                aggf = default_aggregator_gen(t, tlo)
                _df_val['pl_pred'] = _df_val['prob'].groupby(pd.Grouper(freq=da_freq)). \
                    agg(aggf).astype(int).reindex(index=_df_val.index, method='ffill')
                fp = float((_df_val['pl_pred'] != 0).astype(int).sum()) / len(_df_val)
                # If not enough positions taken -> discard
                if fp < min_frac_pos:
                    return pod

                # Add frac_pos to dictionary
                _df_val['frac_pos'] = fp

                # Compute performances
                if eval_metric == 'gain_cum':
                    _df_val['gain'] = ((_df_val['pl_pred'] * _df_val['price_diff']) / gran).round(3)
                    _df_val['gain_cum'] = _df_val['gain'].cumsum().round(3)
                elif eval_metric == 'gain_per_pos':
                    _df_val['gain'] = ((_df_val['pl_pred'] * _df_val['price_diff']) / gran).round(3)
                    _df_val['gain_cum'] = _df_val['gain'].cumsum().round(3)
                    _df_val['gain_per_pos'] = (_df_val['gain_cum'] / (_df_val['pl_pred'] != 0).cumsum()).round(3)
                elif eval_metric == 'accuracy':
                    _df_val['pl_correct'] = (np.sign(_df_val['pl_pred']).astype(int) == np.sign(_df_val['market_tendency']).astype(int)).astype('int8')
                    _df_val['pl_correct'] = _df_val['pl_correct'].replace(0, -1)
                    _df_val.loc[_df_val['pl_pred'] == 0, 'pl_correct'] = 0
                    _df_val['accuracy'] = ((_df_val['pl_correct'] == 1).astype('int8').cumsum().astype(float) / (
                                (_df_val['pl_pred'] != 0) & (_df_val['pl_correct'] != 0)).cumsum()).round(3).fillna(0.5)
                elif eval_metric == 'accuracy_h':
                    _df_val['pl_correct'] = (np.sign(_df_val['pl_pred']).astype(int) == np.sign(_df_val['market_tendency']).astype(int)).astype('int8')
                    _df_val['pl_correct'] = _df_val['pl_correct'].replace(0, -1)
                    _df_val.loc[_df_val['pl_pred'] == 0, 'pl_correct'] = 0
                    df_h = get_hourly_stats(_df_val)
                    _df_val['accuracy_h'] = df_h['accuracy'].reindex(index=_df_val.index, method='ffill')

                # Assign eval metric to output dictionary
                pod[eval_metric] = _df_val[eval_metric][-1]
            except Exception as e:
                print str(e)
                return pod

            return pod

        # Building the (thr, thr_low) combinations
        combs = [(round(th, 2), round(tl, 2)) for th in thr_list for tl in thr_list_low if tl <= th]
        if scan2d is False:
            combs = [c for c in combs if c[0] == c[1]]
        logger.info('Will try {} threshold combinations'.format(len(combs)))

        # Run the scan
        scan_data = Parallel(n_jobs=-1, verbose=10)(delayed(inspect)(th, tl) for th, tl in combs)

        # Sort the scan data
        df_scan = pd.DataFrame(data=scan_data)
        df_scan['index'] = df_scan.apply(lambda x: tuple([x['thr'], x['thr_low']]), axis=1)
        df_scan = df_scan.set_index('index', drop=True)

        return df_scan

    def scan(self, eval_metric, min_frac_pos=0., thr_list=None, scan2d=True, plot=False, out_dir=None, thr_list_low=None, **kwargs):
        # Scan the grid
        df_scan = self._scan(eval_metric, min_frac_pos=min_frac_pos, thr_list=thr_list, scan2d=scan2d,
                             thr_list_low=thr_list_low, **kwargs)
        # Find the best point
        best_thr, best_thr_low = df_scan[eval_metric].idxmax()
        best_metri = df_scan[eval_metric].max()
        logger.info('Best {0} = {3:.2f} achieved with thresholds ({1:.2f}, {2:.2f})'.format(eval_metric, best_thr, best_thr_low, best_metri))
        df_best = self.produce_validation(thr=best_thr, thr_low=best_thr_low, **kwargs)

        # Write to output
        df_scan.to_csv(os.path.join(out_dir, 'scan.csv'))
        df_best.to_csv(os.path.join(out_dir, 'validation.csv'))

        # Eventually, plot
        if plot is True:
            if out_dir is None:
                logger.warning('You must specify the out_dir parameter to plot the scan')
            else:
                if scan2d is True:
                    plot_scan(df_scan, what=eval_metric, out_dir=out_dir)
                else:
                    plot_scan_1d(df_scan, what=eval_metric, out_dir=out_dir)

        return df_best, df_scan

    def dynamic_scan(self, eval_metric, dt_start, dt_end, start_upd=None, start_thrs=None,
                     upd_freq='1W', thr_window=0.03, week_lookback=6, min_frac_pos=0.,
                     thr_list=None, scan2d=True, out_dir=None, thr_list_low=None, use_avg_price=False):
        # Consistency check
        if (start_upd is None and start_thrs is None) or (start_upd is not None and start_thrs is not None):
            msg = 'One and only one of start_upd and start_thrs parameter must be specified'
            logger.error(msg)
            raise ValueError(msg)
        if start_thrs is not None:
            start_upd = dt_start

        # Get the base DataFrame to operate on
        df_base = self.produce_validation(thr=0.5, thr_low=0.5, dt_start=dt_start, dt_end=dt_end,
                                          use_avg_price=use_avg_price)

        # Get the start threshold pairs if not specified
        if start_thrs is None:
            df_best, df_scan = self.scan(eval_metric=eval_metric, min_frac_pos=min_frac_pos, thr_list=thr_list,
                                         scan2d=scan2d, plot=False, out_dir=out_dir, thr_list_low=thr_list_low,
                                         dt_start=dt_start, dt_end=start_upd)
            best_thr, best_thr_low = round(df_best['threshold'][-1], 2), round(df_best['threshold_low'][-1], 2)
        else:
            best_thr, best_thr_low = start_thrs[0], start_thrs[1]

        # Loop over the periods to apply previously optimised thresholds
        for p, df_p in df_base[start_upd:].groupby(pd.Grouper(freq=upd_freq)):
            # Start end dates for update of thresholds
            s, e = df_p.index[0], df_p.index[-1]
            logger.info('New threshold pair ({0:.2f}, {1:.2f}) '
                        'will be applied in period {2} to {3}'.format(best_thr, best_thr_low, s, e))
            # Add threshold info to the original DataFrame
            df_base.loc[s: e, 'threshold'] = best_thr
            df_base.loc[s: e, 'threshold_low'] = best_thr_low
            # Build aggregator from generator
            aggf = default_aggregator_gen(best_thr, best_thr_low)
            # Add predictions according to new thresholds
            df_p = add_agg_positions(df_p, aggf, self.da_coll.freq)
            df_base.loc[s: e, 'pl_pred'] = df_p.loc[s: e, 'pl_pred']
            df_base.loc[s: e, 'pl_pred_str'] = df_p.loc[s: e, 'pl_pred_str']
            # Re-scan the thresholds
            _thr_list = np.arange(best_thr - thr_window, best_thr + thr_window, 0.01)
            _thr_list_low = np.arange(best_thr_low - thr_window, best_thr_low + thr_window, 0.01)
            end_scan = e.tz_localize(None).replace(hour=0, minute=0, second=0)
            start_scan = max(end_scan - pd.Timedelta(weeks=week_lookback), dt_start)
            logger.debug('New scan starts at {0} and ends at {1}'.format(start_scan, end_scan))
            logger.debug('Grid is:\nthr = {0}\nthr_low = {1}'.format(_thr_list, _thr_list_low))
            _df_best, df_scan = self.scan(eval_metric=eval_metric, min_frac_pos=min_frac_pos, thr_list=_thr_list,
                                          scan2d=scan2d, plot=False, out_dir=out_dir, thr_list_low=_thr_list_low,
                                          dt_start=start_scan, dt_end=end_scan)
            best_thr, best_thr_low = round(_df_best['threshold'][-1], 2), round(_df_best['threshold_low'][-1], 2)

        # Amend the fraction of positions
        df_base['frac_pos'] = ((df_base['pl_pred'] != 0).astype(int).cumsum() / df_base['pl_pred'].expanding(
            min_periods=1).count()).round(3)

        # Compute performances
        df_best = compute_perfomances(df_base)

        # Get aggregate hourly statistics
        df_h = get_hourly_stats(df_best)

        # Add to the original DataFrame
        df_best['accuracy_h'] = df_h['accuracy'].reindex(index=df_best.index, method='ffill')

        del df_h
        gc.collect()

        return df_best


def get_freq_from_df(df):
    # Retrieve the frequency
    if df.index.tz is None:
        freq = pd.infer_freq(df.index)
    else:
        df_copy = df.copy()
        df_copy.index = df_copy.index.tz_convert('UTC')
        df_copy = df_copy[~df_copy.index.duplicated(keep='first')].copy()
        freq = pd.infer_freq(df_copy.index)

    # Test the frequency value
    if freq == 'H' or freq == '1H' or freq == '60T':
        gran = 1
    elif freq == '30T' or freq == '0.5H':
        gran = 2
    elif freq == '15T' or freq == '0.25H':
        gran = 4
    else:
        msg = 'It was not possible to infer the time granularity: check the DataFrame!'
        logger.error(msg)
        raise ValueError(msg)

    return freq, gran


def add_market_tendency(df):
    # Input check
    cols = ['positive_price', 'negative_price', 'dayahead_price']
    for col in cols:
        if col not in df.columns:
            msg = 'Column %s is not in DataFrame' % col
            logger.error(msg)
            raise ValueError(msg)

    # Do not write on input object
    _df = df.copy()

    # We use the average price to compute the tendency
    _df['imbalance_price'] = _df[['positive_price', 'negative_price']].mean(axis=1).round(3)
    _df['price_diff'] = (_df['imbalance_price'] - _df['dayahead_price']).round(3)
    _df['price_diff_pos'] = (_df['positive_price'] - _df['dayahead_price']).round(3)
    _df['price_diff_neg'] = (_df['negative_price'] - _df['dayahead_price']).round(3)
    _df['market_tendency'] = np.sign(_df['price_diff'].fillna(0.)).astype(int)
    _df['market_tendency_str'] = _df.apply(lambda x: 'short' if x['market_tendency'] > 0 else 'long' if x['market_tendency'] < 0 else 'none', axis=1)

    return _df


def add_agg_positions(df, aggfunc=None, da_freq='1H'):
    # Input check
    cols = ['prob']
    for col in cols:
        if col not in df.columns and not(col == 'prob' and 'pl_pred' in df.columns):
            msg = 'Column %s is not in DataFrame' % col
            logger.error(msg)
            raise ValueError(msg)

    # Do not write on input object
    _df = df.copy()

    # More check
    if not (aggfunc is None and 'pl_pred' in df.columns):
        # Aggregate the predictions
        _df['pl_pred'] = _df['prob'].groupby(pd.Grouper(freq=da_freq)). \
            agg(aggfunc).astype(int).reindex(index=_df.index, method='ffill')

    _df['pl_pred_str'] = _df.apply(
        lambda x: 'short' if x['pl_pred'] < 0 else 'long' if x['pl_pred'] > 0 else 'none', axis=1)
    _df['frac_pos'] = ((_df['pl_pred'] != 0).astype(int).cumsum() / _df['pl_pred'].expanding(
        min_periods=1).count()).round(3)

    return _df


def compute_perfomances(df, freq=None, gran=None):
    # Input check
    cols = ['pl_pred', 'price_diff', 'market_tendency']
    for col in cols:
        if col not in df.columns:
            msg = 'Column %s is not in DataFrame' % col
            logger.error(msg)
            raise ValueError(msg)

    # Do not write on input object
    _df = df.copy()

    if freq is None or gran is None:
        # Get frequency
        freq, gran = get_freq_from_df(_df)

    # You can always add more
    _df['pl_correct'] = (np.sign(_df['pl_pred']).astype(int) == np.sign(_df['market_tendency']).astype(int)).astype('int8')
    _df['pl_correct'] = _df['pl_correct'].replace(0, -1)
    _df.loc[_df['pl_pred'] == 0, 'pl_correct'] = 0
    _df['gain'] = ((_df['pl_pred'] * _df['price_diff']) / gran).round(3)
    _df['gain_cum'] = _df['gain'].cumsum().round(3)
    _df['gain_per_pos'] = (_df['gain_cum'] / (_df['pl_pred'] != 0).cumsum()).round(3)
    _df['accuracy'] = ((_df['pl_correct'] == 1).astype('int8').cumsum().astype(float) / ((_df['pl_pred'] != 0) & (_df['pl_correct'] != 0)).cumsum()).round(3).fillna(0.5)
    if 'prob' in _df.columns:
        try:
            _df['rocauc'] = round(roc_auc_score(_df.loc[_df['price_diff'] != 0., 'price_diff'].map(lambda x: 1 if x > 0 else 0).values,
                                                _df.loc[_df['price_diff'] != 0., 'prob'].values), 3)
        except ValueError as e:
            if 'Only one class' in str(e):
                logger.warning('Only one class in the data, it is not possible to compute the ROC AUC metric.')
                _df['rocauc'] = np.nan
                pass
            else:
                logger.error('Error occured while computing ROC AUC:\n %s' % str(e))
                raise e
    else:
        _df['rocauc'] = np.nan

    # Add share
    _df['price_diff_abs'] = _df['price_diff'].abs()
    _df['tot_spread_abs_cum'] = _df['price_diff_abs'].cumsum()
    _df['share_cum'] = (_df['gain_cum'] / _df['tot_spread_abs_cum']).round(3)

    return _df


def get_hourly_stats(df):
    # Compute the hourly statistics
    df_h = df[['pl_correct', 'pl_pred']].groupby(pd.Grouper(freq='1H')).agg({'pl_correct': np.mean, 'pl_pred': lambda x: np.sign(x[0])})
    df_h['pl_correct'] = df_h['pl_correct'].map(np.sign).astype('int8')
    df_h['accuracy'] = ((df_h['pl_correct'] == 1).astype('int8').cumsum().astype(float) / ((df_h['pl_pred'] != 0) & (df_h['pl_correct'] != 0)).cumsum()).round(3).fillna(0.5)

    return df_h


def default_aggregator_gen(thr, thr_low):
    def agg(ps):
        preds_h = []
        for p in ps:
            if p < thr_low:
                preds_h.append(-1)
            elif p < thr:
                preds_h.append(0)
            else:
                preds_h.append(1)

        pred_mean = np.mean(preds_h)

        if pred_mean == 0.:
            return 0
        elif pred_mean > 0.:
            return 1
        else:
            return -1
    return agg


def default_aggregator(ps, thr, thr_low):
    preds_h = []
    for p in ps:
        if p < thr_low:
            preds_h.append(-1)
        elif p < thr:
            preds_h.append(0)
        else:
            preds_h.append(1)

    pred_mean = np.mean(preds_h)

    if pred_mean == 0.:
        return 0
    elif pred_mean > 0.:
        return 1
    else:
        return -1


def compute_cost(df, mult=1.):
    # Input check
    cols = ['pl_pred', 'price_diff', 'real']
    for col in cols:
        if col not in df.columns:
            msg = 'Column %s is not in DataFrame' % col
            logger.error(msg)
            raise ValueError(msg)

    # Do not write on input object
    _df = df.copy()

    # Get frequency
    freq, gran = get_freq_from_df(_df)

    # You can always add more
    _df['imbalance'] = _df['real'] - _df['pl_pred']
    _df['cost'] = mult * ((_df['imbalance'] * _df['price_diff']) / gran).round(3)
    _df['cost_cum'] = _df['cost'].cumsum().round(3)

    return _df


class MarketTendencyPlotter(object):
    def __init__(self, df, out_dir, labels=None):
        self.df_val = []
        if not isinstance(df, list):
            df = [df]

        for df_ in df:
            if isinstance(df_, pd.core.frame.DataFrame):
                self.df_val.append(df_.copy())
            else:
                self.df_val.append(pd.read_csv(df_, index_col=0, parse_dates=[0]))
        self.out_dir = out_dir

        # Labels
        self.labels = labels
        if self.labels is not None:
            if len(self.labels) != len(self.df_val):
                msg = 'You must provide an array of labels with the same length as the list of data in input'
                logger.error(msg)
                raise ValueError(msg)

            for i, _ in enumerate(self.df_val):
                self.df_val[i] = self.df_val[i].rename({'name': 'model'})
                self.df_val[i]['name'] = self.labels[i]

    def plot_ts_smooth(self, what, smooth=None, **kwargs):

        xs = self.df_val[0].index

        fname = 'ts_' + what
        if smooth is not None:
            fname += '_smooth' + str(smooth)

        ys, titles = [], []
        for i, df in enumerate(self.df_val):
            titles.append(df['name'][0])
            if smooth is not None:
                ys.append(df[what].rolling(window=smooth, min_periods=1).mean().values)
            else:
                ys.append(df[what].values)

        plot_ts_mpl(xs, ys, title=titles, ylab=kwargs.get('ylab', None),
                    out_dir=self.out_dir, filename=fname)

    def plot_ts_group(self, what, group='1D', func=np.sum, **kwargs):

        xs = self.df_val[0].groupby(pd.Grouper(freq=group)).mean().index
        fname = 'ts_' + what + '_group' + str(group)

        ys, titles = [], []
        for df in self.df_val:
            titles.append(df['name'][0])
            ys.append(df[what].groupby(pd.Grouper(freq=group)).agg(func))
        plot_ts_mpl(xs, ys, title=titles, ylab=kwargs.get('ylab', None),
                    out_dir=self.out_dir, filename=fname)

    def plot_hist_inperiod(self, what, group='1D', func=np.sum, **kwargs):
        for df in self.df_val:
            add_tag = '_' + df['name'][0] if len(self.df_val) > 1 else ''
            plot_hist_period(df[what], self.out_dir, tag=add_tag, group=group, func=func, **kwargs)

    def plot_sns_inclass(self, what, category, **kwargs):
        for df in self.df_val:
            add_tag = '_' + df['name'][0] if len(self.df_val) > 1 else ''
            plot_sns_class(df, what, category,  self.out_dir, tag=add_tag, **kwargs)

    def plot_hist_inclass(self, what, category, **kwargs):
        for df in self.df_val:
            add_tag = '_' + df['name'][0] if len(self.df_val) > 1 else ''
            plot_hist_class(df, what, category,  self.out_dir, tag=add_tag, **kwargs)
