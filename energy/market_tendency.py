import os
import gc
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_tools.timeseries import plot_ts_mpl

logger = logging.getLogger(__file__)

# Fir the scan
DEFAULT_THR_LIST_SCAN = np.round(np.arange(0.2, 0.70001, 0.02), 2)


class MarketTendencyValidator(object):
    def __init__(self, da_coll, pos_coll, score_coll, neg_coll=None):
        self.da_coll = da_coll
        self.pos_coll = pos_coll
        self.score_coll = score_coll
        self.df_input = pd.DataFrame()
        self.df_val = pd.DataFrame()

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
        df_scores = self.score_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='prob')

        # Concatenate
        df = pd.concat([df_da, df_pos, df_neg, df_scores], axis=1)

        return df.copy()

    def produce_validation(self, dt_start, dt_end, thr, thr_low=None, agg_pred=None, use_avg_price=True):
        if thr_low is None:
            thr_low = thr

        def default_aggregator(ps):
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

        if agg_pred is None:
            agg_pred = default_aggregator

        # Fetch the data
        if not len(self.df_input) > 0:
            self.df_input = self._fetch(dt_start, dt_end)

        # Market tendency
        self.df_val = self.df_input.copy()
        self.df_val['imbalance_price'] = self.df_val[['positive_price', 'negative_price']].mean(axis=1)
        self.df_val['price_diff'] = (self.df_val['imbalance_price'] - self.df_val['dayahead_price']).round(3)
        self.df_val['price_diff_pos'] = (self.df_val['positive_price'] - self.df_val['dayahead_price']).round(3)
        self.df_val['price_diff_neg'] = (self.df_val['negative_price'] - self.df_val['dayahead_price']).round(3)
        self.df_val['market_tendency'] = self.df_val['price_diff'].map(lambda x: 1 if x >= 0. else -1)

        # Get hour aggregates
        self.df_val['threshold'] = thr
        self.df_val['threshold_low'] = thr_low
        self.df_val['pl_pred'] = self.df_val['prob'].groupby(pd.Grouper(freq=self.da_coll.freq)).\
            agg(agg_pred).astype(int).reindex(index=self.df_val.index, method='ffill')
        self.df_val['frac_pos'] = ((self.df_val['pl_pred'] != 0).astype(int).cumsum() / self.df_val['pl_pred'].expanding(min_periods=1).count()).round(3)

        # Are we correct?
        self.df_val['pl_correct'] = (self.df_val['pl_pred'] == self.df_val['market_tendency']).astype('int8')
        self.df_val['pl_correct'] = self.df_val['pl_correct'].replace(0, -1)
        self.df_val.loc[self.df_val['pl_pred'] == 0, 'pl_correct'] = 0

        # Compute the cost/gain
        if use_avg_price is True:
            self.df_val['imbalance_price'] = self.df_val[['positive_price', 'negative_price']].mean(axis=1)
        else:  # if pred == 0 we don't care about the imbalance price anyway
            self.df_val['imbalance_price'] = self.df_val.apply(lambda x: x['positive_price'] if x['pl_pred'] == 1 else x['negative_price'], axis=1)
        self.df_val['price_diff'] = (self.df_val['imbalance_price'] - self.df_val['dayahead_price']).round(3)
        self.df_val['gain'] = ((self.df_val['pl_pred'] * self.df_val['price_diff']) / 4.).round(3)
        self.df_val['gain_cum'] = self.df_val['gain'].cumsum().round(3)
        self.df_val['gain_per_pos'] = (self.df_val['gain_cum'] / (self.df_val['pl_pred'] != 0).cumsum()).round(3)
        self.df_val['accuracy'] = ((self.df_val['pl_correct'] == 1).astype('int8').cumsum().astype(float) / (self.df_val['pl_pred'] != 0).cumsum()).round(3)

        return self.df_val.copy()

    def dump(self, out_dir):
        self.df_val.to_csv(os.path.join(out_dir, 'validation.csv'))

    def scan(self, eval_metric, min_frac_pos=0., **kwargs):
        metric_v_best = 0. if eval_metric == 'accuracy' else -1.E+06
        df_best = pd.DataFrame()
        for thr in DEFAULT_THR_LIST_SCAN:
            for thr_low in DEFAULT_THR_LIST_SCAN:
                if thr_low > thr:
                    continue
                df_val = self.produce_validation(thr=thr, thr_low=thr_low, **kwargs)
                if df_val.iloc[-1, df_val.columns.get_loc('frac_pos')] < min_frac_pos:
                    continue
                metric_v = df_val.iloc[-1, df_val.columns.get_loc(eval_metric)]
                logger.debug('Scanning thr = {0:.2f}, thr_low = {1:.2f}: {2} = {3:.2f}'.format(thr, thr_low,
                                                                                               eval_metric, metric_v))
                if metric_v > metric_v_best:
                    logger.info('New best {0} = {1:.2f}'.format(eval_metric, metric_v))
                    metric_v_best = metric_v
                    df_best = df_val.copy()
                del df_val
                gc.collect()

        self.df_val = df_best.copy()
        del df_best
        gc.collect()


class MarketTendencyPlotter(object):
    def __init__(self, df, out_dir):
        if isinstance(df, pd.core.frame.DataFrame):
            self.df_val = df.copy()
        else:
            self.df_val = pd.read_csv(df, index_col=0, parse_dates=[0])
        self.out_dir = out_dir

    def plot_ts_smooth(self, what, smooth=None, **kwargs):

        xs = self.df_val.index
        fname = 'ts_' + what
        if smooth is not None:
            fname += '_smooth' + str(smooth)
            ys = self.df_val[what].rolling(window=smooth, min_periods=1).mean().values
        else:
            ys = self.df_val[what].values
        plot_ts_mpl(xs, ys, title=kwargs.get('title', None), ylab=kwargs.get('ylab', None),
                    out_dir=self.out_dir, filename=fname)

    def plot_ts_group(self, what, group='1D', func=np.sum, **kwargs):

        xs = self.df_val.groupby(pd.Grouper(freq=group)).agg(lambda x: x[0]).index
        fname = 'ts_' + what + '_group' + str(group)
        ys = self.df_val[what].groupby(pd.Grouper(freq=group)).agg(func)
        plot_ts_mpl(xs, ys, title=kwargs.get('title', None), ylab=kwargs.get('ylab', None),
                    out_dir=self.out_dir, filename=fname)

    def plot_hist_group(self, what, group='1D', func=np.sum, **kwargs):
        fname = 'hist_' + what + '_group%s.png' % group
        a = self.df_val[what].groupby(pd.Grouper(freq=group)).agg(func).values

        fig, ax = plt.subplots()
        x_bins = kwargs.get('x_bins', None)
        if x_bins is None:
            x_min, x_max = min(a), max(a)
            x_bins = np.linspace(x_min, x_max, 100)

        ax.hist(a, alpha=0.7, histtype='step', bins=x_bins, color='blue', fill=True)

        if kwargs.get('xlab', None) is not None:
            plt.xlabel(kwargs['xlab'])
        else:
            plt.xlabel(what)
        # plt.legend(loc='best')
        plt.savefig(os.path.join(self.out_dir, fname))

    def plot_hist_class(self, what, **kwargs):
        fname = 'hist_' + what + '_classes.png'
        a_long = self.df_val.loc[self.df_val['market_tendency'] == -1, what].values
        a_short = self.df_val.loc[self.df_val['market_tendency'] == 1, what].values

        fig, ax = plt.subplots()
        x_bins = kwargs.get('x_bins', None)
        if x_bins is None:
            x_min, x_max = min(np.concatenate((a_long, a_short))), max(np.concatenate((a_long, a_short)))
            x_bins = np.linspace(x_min, x_max, 100)

        ax.hist(a_long, alpha=0.2, histtype='step', bins=x_bins, color='green', fill=True, label='Market long')
        ax.hist(a_short, alpha=0.2, histtype='step', bins=x_bins, color='red', fill=True, label='Market short')

        if kwargs.get('xlab', None) is not None:
            plt.xlabel(kwargs['xlab'])
        else:
            plt.xlabel(what)
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.out_dir, fname))


