# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     intraday.py
#
# Project:  Predictive Layer for: pl_tools
# Author:   Stefano
# Date:     28 June 2018
#
# Scope:    The file contains PnL evaluation scripts
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
import gc
import logging
import pandas as pd
import numpy as np
from validation.market_tendency import get_hourly_stats, add_agg_positions, add_market_tendency, \
    compute_perfomances, default_aggregator

logger = logging.getLogger(__file__)

# Metrics
METRICS = ['accuracy', 'accuracy_h', 'gain_cum', 'gain_per_pos']


class Model(object):
    def __init__(self, score_coll, threshold, threshold_low=None):
        self.score_coll = score_coll
        self.threshold = threshold
        if threshold_low is None:
            self.threshold_low = self.threshold
        else:
            self.threshold_low = threshold_low

        # Initialize
        self.scores = pd.Series()
        self.positions = pd.Series()

    def _fetch(self, dt_start, dt_end, **kwargs):
        # Scores
        self.scores = self.score_coll.get_data(dt_start=dt_start, dt_end=dt_end, **kwargs)

    def get_scores(self, dt_start, dt_end, **kwargs):
        if not len(self.scores) > 0:
            self._fetch(dt_start, dt_end, **kwargs)

        return self.scores

    def get_positions(self, dt_start, dt_end, aggfunc=None, da_freq='1H'):
        if len(self.positions) > 0:
            return self.positions

        if not len(self.scores) > 0:
            self._fetch(dt_start, dt_end)

        def default_agg(ps):
            return default_aggregator(ps, self.threshold, self.threshold_low)

        if aggfunc is None:
            aggfunc = default_agg

        self.positions = self.scores.groupby(pd.Grouper(freq=da_freq)).\
            agg(aggfunc).astype(int).reindex(index=self.scores.index, method='ffill')

        return self.positions

    def get_name(self):
        return self.score_coll.get_name()


class ModelEnsemble(object):
    def __init__(self, name, model_collection, da_coll, pos_coll, neg_coll=None):
        self.name = name

        # Sanity check
        tot = []
        for _, s, e in model_collection:
            tot += list(range(s, e))
        if sorted(tot) != list(range(0, 24)):
            msg = 'Something is wrong with the hours specified in the model collection:%s' % sorted(tot)
            logger.error(msg)
            raise ValueError(msg)

        # It's a list of:
        # [(Model, start_h, end_h)]
        self.model_collection = model_collection
        self.score_columns = []
        self.pred_columns = []

        self.da_coll = da_coll
        self.pos_coll = pos_coll
        self.df_input = pd.DataFrame()
        self.df_val = pd.DataFrame()
        self.df_scan = pd.DataFrame()

        if neg_coll is None:
            self.neg_coll = self.pos_coll
        else:
            self.neg_coll = neg_coll

    def _fetch(self, dt_start, dt_end):
        # Positive price
        df_pos = self.pos_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='positive_price')

        # Negative price
        df_neg = self.neg_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='negative_price')

        # Day-ahead price
        df_da = self.da_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='dayahead_price')
        df_da = df_da.reindex(index=df_pos.index, method='ffill')

        # Scores
        dfs = []
        for m, _, _ in self.model_collection:
            score_col = 'prob_' + m.get_name()
            if score_col not in self.score_columns:
                dfs.append(m.get_scores(dt_start=dt_start, dt_end=dt_end, rename=score_col))
            self.score_columns.append(score_col)
        df_scores = pd.concat(dfs, axis=1)

        # Concatenate
        df = pd.concat([df_da, df_pos, df_neg, df_scores], axis=1)

        # Printout the nans
        for c in df.columns:
            nans = df[c].isnull()
            if nans.sum() > 0:
                msg = '{0} NaNs found in column {1}'.format(nans.sum(), c)
                nans_dates = list(set(df[nans].index.strftime('%Y-%m-%d')))
                logger.warning(msg)
                if len(nans_dates) < 50:
                    logger.info('Dates with Nans:\n%s' % str(nans_dates))

        return df.dropna()

    def build_convolution(self, prob_col='prob', pred_col='pl_pred'):
        self.df_val['hour'] = self.df_val.index.hour
        for (m, h_start, h_end), pred_col_m, prob_col_m in zip(self.model_collection, self.pred_columns, self.score_columns):
            logger.info('Using model {0} from hour {1} to hour {2}'.format(m.get_name(), h_start, h_end))
            mask = (self.df_val['hour'] >= h_start) & (self.df_val['hour'] < h_end)
            self.df_val.loc[mask, pred_col] = self.df_val.loc[mask, pred_col_m]
            self.df_val.loc[mask, prob_col] = self.df_val.loc[mask, prob_col_m]
        self.df_val['%s_str' % pred_col] = self.df_val.apply(
            lambda x: 'short' if x[pred_col] == 1 else 'long' if x[pred_col] == -1 else 'none', axis=1)
        self.df_val['frac_pos'] = ((self.df_val[pred_col] != 0).astype(int).cumsum() / self.df_val[pred_col].expanding(
            min_periods=1).count()).round(3)

    def produce_validation(self, dt_start, dt_end, use_avg_price=True, skip_dates=None):
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
        if not len(self.df_input) > 0:
            self.df_input = self._fetch(dt_start, dt_end)
            self.df_input['name'] = self.name

        # Eventually drop dates
        try:
            self.df_input = self.df_input.drop(skip_dates)
        except ValueError:
            logger.warning('You are trying to skip dates which are not in index')
            pass

        # Input collections
        self.df_val = self.df_input.copy()

        # Add the market tendency
        self.df_val = add_market_tendency(self.df_val)

        # Get hour aggregates for all the models in the collection
        for (m, _, _), prob_col in zip(self.model_collection, self.score_columns):
            self.df_val['threshold_' + m.get_name()] = m.threshold
            self.df_val['threshold_low' + m.get_name()] = m.threshold_low
            pred_col = 'pred_' + m.get_name()
            if pred_col not in self.df_val.columns:
                self.df_val[pred_col] = m.get_positions(dt_start, dt_end, da_freq=self.da_coll.freq)
            self.pred_columns.append(pred_col)

        # Create the comvolution of models
        self.build_convolution()

        # Compute the cost/gain
        if use_avg_price is True:
            self.df_val['imbalance_price'] = self.df_val[['positive_price', 'negative_price']].mean(axis=1)
        else:
            self.df_val['imbalance_price'] = self.df_val.apply(lambda x: x['positive_price'] if x['pl_pred'] >= 1 else x['negative_price'], axis=1)
        self.df_val['price_diff'] = (self.df_val['imbalance_price'] - self.df_val['dayahead_price']).round(3)
        self.df_val = compute_perfomances(self.df_val)

        # Get aggregate hourly statistics
        df_h = get_hourly_stats(self.df_val)

        # Add to the original DataFrame
        self.df_val['accuracy_h'] = df_h['accuracy'].reindex(index=self.df_val.index, method='ffill')

        del df_h
        gc.collect()

        return self.df_val.copy()

    def dump(self, out_dir):
        self.df_val.to_csv(os.path.join(out_dir, 'validation.csv'))
