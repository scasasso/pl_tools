# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     pnl.py
#
# Project:  Predictive Layer for: pl_tools
# Author:   Stefano
# Date:     27 January 2018
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


class PNLValidatorGermany(object):
    def __init__(self, name, da_coll, pos_coll, neg_coll, positions, pos_coll_m=None, neg_coll_m=None):
        self.name = name
        self.da_coll = da_coll
        self.pos_coll = pos_coll
        self.neg_coll = neg_coll

        # Collection of positions
        if not isinstance(positions, pd.core.series.Series):
            msg = 'The \"positions\" argument must be a pandas Series'
            logger.error(msg)
            raise ValueError(msg)
        self.positions = positions

        # Rebap estimate
        if pos_coll_m is not None:
            self.pos_coll_m = pos_coll_m
        else:
            self.pos_coll_m = self.pos_coll
        if neg_coll_m is not None:
            self.neg_coll_m = neg_coll_m
        else:
            self.neg_coll_m = self.neg_coll

        self.df_input = pd.DataFrame()
        self.df_val = pd.DataFrame()

    def _fetch(self, dt_start, dt_end):
        datetimes = pd.date_range(dt_start, dt_end + pd.Timedelta(hours=23) + pd.Timedelta(minutes=59),
                                  freq=self.pos_coll.freq)
        df = pd.DataFrame(index=datetimes)

        # Positive price
        df_pos = self.pos_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='positive_price')
        if len(df_pos) > 0:
            df = df.join(df_pos, how='outer')
        else:
            df['positive_price'] = np.nan

        # Negative price
        df_neg = self.neg_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='negative_price')
        if len(df_neg) > 0:
            df = df.join(df_neg, how='outer')
        else:
            df['negative_price'] = np.nan

        # Positive price - rebap estimate
        df_pos_m = self.pos_coll_m.get_data(dt_start=dt_start, dt_end=dt_end, rename='positive_price_pl')
        if len(df_pos_m) > 0:
            df = df.join(df_pos_m, how='outer')
        else:
            df['positive_price_pl'] = np.nan

        # Negative price - rebap estimate
        df_neg_m = self.neg_coll_m.get_data(dt_start=dt_start, dt_end=dt_end, rename='negative_price_pl')
        if len(df_neg_m) > 0:
            df = df.join(df_neg_m, how='outer')
        else:
            df['negative_price_pl'] = np.nan

        # Day-ahead price
        df_da = self.da_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='dayahead_price')
        df_da = df_da.reindex(index=df.index, method='ffill')
        df = df.join(df_da, how='outer')

        # # Concatenate
        # df = pd.concat([df_da, df_pos, df_neg, df_pos_m, df_neg_m], axis=1)

        return df.dropna(how='all')

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

        # Fill missing prices with price estimate
        self.df_val['positive_price'] = self.df_val.apply(
            lambda x: x.get('positive_price', None) if not pd.isnull(x.get('positive_price', None)) else x.get(
                'positive_price_pl', None), axis=1)
        self.df_val['negative_price'] = self.df_val.apply(
            lambda x: x.get('negative_price', None) if not pd.isnull(x.get('negative_price', None)) else x.get(
                'negative_price_pl', None), axis=1)

        # Add the market tendency
        self.df_val = add_market_tendency(self.df_val)

        # Get hour aggregates
        self.df_val['pl_pred'] = self.positions.reindex(index=self.df_val.index, method='ffill')
        self.df_val = add_agg_positions(self.df_val, da_freq=self.da_coll.freq)

        # Compute the cost/gain
        if use_avg_price is True:
            self.df_val['imbalance_price'] = self.df_val[['positive_price', 'negative_price']].mean(axis=1)
        else:  # if pred == 0 we don't care about the imbalance price anyway
            self.df_val['imbalance_price'] = self.df_val.apply(
                lambda x: x['positive_price'] if x['pl_pred'] > 0 else x['negative_price'] if x['pl_pred'] < 0 else x.get('dayahead_price', np.nan), axis=1)
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


class PNLValidatorBelgium(object):
    def __init__(self, name, da_coll, pos_coll, neg_coll, positions):
        self.name = name
        self.da_coll = da_coll
        self.pos_coll = pos_coll
        self.neg_coll = neg_coll

        # Collection of positions
        if not isinstance(positions, pd.core.series.Series):
            msg = 'The \"positions\" argument must be a pandas Series'
            logger.error(msg)
            raise ValueError(msg)
        self.positions = positions

        self.df_input = pd.DataFrame()
        self.df_val = pd.DataFrame()

    def _fetch(self, dt_start, dt_end):
        # Positive price
        df_pos = self.pos_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='positive_price')

        # Negative price
        df_neg = self.neg_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='negative_price')

        # Day-ahead price
        df_da = self.da_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='dayahead_price')
        df_da = df_da.reindex(index=df_pos.index, method='ffill')

        # Concatenate
        df = pd.concat([df_da, df_pos, df_neg], axis=1)

        return df.dropna()

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

        # Fill missing prices with price estimate
        self.df_val['positive_price'] = self.df_val.apply(lambda x: x.get('positive_price', None), axis=1)
        self.df_val['negative_price'] = self.df_val.apply(lambda x: x.get('negative_price', None), axis=1)

        # Add the market tendency
        self.df_val = add_market_tendency(self.df_val)

        # Get hour aggregates
        self.df_val['pl_pred'] = self.positions.reindex(index=self.df_val.index, method='ffill')
        self.df_val = add_agg_positions(self.df_val, da_freq=self.da_coll.freq)

        # Compute the cost/gain
        if use_avg_price is True:
            self.df_val['imbalance_price'] = self.df_val[['positive_price', 'negative_price']].mean(axis=1)
        else:  # if pred == 0 we don't care about the imbalance price anyway
            self.df_val['imbalance_price'] = self.df_val.apply(
                lambda x: x['positive_price'] if x['pl_pred'] > 0 else x['negative_price'] if x['pl_pred'] < 0 else x.get('dayahead_price', np.nan), axis=1)
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
