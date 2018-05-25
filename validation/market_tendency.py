import os
import gc
import logging
import pandas as pd
import numpy as np
from plot_tools.timeseries import plot_ts_mpl
from plot_tools.histogram import plot_hist_period, plot_sns_class, plot_hist_class, plot_scan
from plot_tools.graph import plot_scan_1d
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__file__)

# Fir the scan
DEFAULT_THR_LIST_SCAN = np.round(np.arange(0.2, 0.70001, 0.01), 2)

# Metrics
METRICS = ['accuracy', 'accuracy_h', 'gain_cum', 'gain_per_pos']


class MarketTendencyValidator(object):
    def __init__(self, name, da_coll, pos_coll, score_coll, neg_coll=None):
        self.name = name
        self.da_coll = da_coll
        self.pos_coll = pos_coll
        self.score_coll = score_coll
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
        df_scores = self.score_coll.get_data(dt_start=dt_start, dt_end=dt_end, rename='prob')

        # Concatenate
        df = pd.concat([df_da, df_pos, df_neg, df_scores], axis=1)

        return df.copy()

    def produce_validation(self, dt_start, dt_end, thr, thr_low=None, agg_pred=None, use_avg_price=True, skip_dates=None):
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
            self.df_input['name'] = self.name

        # Eventually drop dates
        try:
            self.df_input = self.df_input.drop(skip_dates)
        except ValueError:
            logger.warning('You are trying to skip dates which are not in index')
            pass

        # Market tendency
        self.df_val = self.df_input.copy()
        self.df_val['imbalance_price'] = self.df_val[['positive_price', 'negative_price']].mean(axis=1).round(3)
        self.df_val['price_diff'] = (self.df_val['imbalance_price'] - self.df_val['dayahead_price']).round(3)
        self.df_val['price_diff_pos'] = (self.df_val['positive_price'] - self.df_val['dayahead_price']).round(3)
        self.df_val['price_diff_neg'] = (self.df_val['negative_price'] - self.df_val['dayahead_price']).round(3)
        self.df_val['market_tendency'] = np.sign(self.df_val['price_diff'].fillna(0.)).astype(int)

        # Get hour aggregates
        self.df_val['threshold'] = thr
        self.df_val['threshold_low'] = thr_low
        self.df_val['pl_pred'] = self.df_val['prob'].groupby(pd.Grouper(freq=self.da_coll.freq)).\
            agg(agg_pred).astype(int).reindex(index=self.df_val.index, method='ffill')
        self.df_val['pl_pred_str'] = self.df_val.apply(lambda x: 'short' if x['pl_pred'] == 1 else 'long' if x['pl_pred'] == -1 else 'none', axis=1)
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
        self.df_val['accuracy'] = ((self.df_val['pl_correct'] == 1).astype('int8').cumsum().astype(float) / (self.df_val['pl_pred'] != 0).cumsum()).round(3).fillna(0.5)
        self.df_val['rocauc'] = round(roc_auc_score(self.df_val['price_diff'].map(lambda x: 1 if x >= 0 else 0).values,
                                                    self.df_val['prob'].values), 3)

        # Compute the hourly statistics
        df_h = self.df_val[['dayahead_price', 'imbalance_price']].groupby(pd.Grouper(freq='1H')).agg(np.mean)
        df_h['price_diff'] = df_h['imbalance_price'] - df_h['dayahead_price']
        df_h['market_tendency'] = np.sign(df_h['price_diff'].fillna(-10.)).astype(int)
        df_h = df_h.loc[df_h['market_tendency'] > -10, :]
        df_h['pl_pred'] = self.df_val['pl_pred']
        df_h['pl_correct'] = (df_h['pl_pred'] == df_h['market_tendency']).astype('int8')
        df_h['pl_correct'] = df_h['pl_correct'].replace(0, -1)
        df_h.loc[df_h['pl_pred'] == 0, 'pl_correct'] = 0
        df_h['accuracy'] = ((df_h['pl_correct'] == 1).astype('int8').cumsum().astype(float) / (df_h['pl_pred'] != 0).cumsum()).round(3).fillna(0.5)

        # Add to the original DataFrame
        self.df_val['accuracy_h'] = df_h['accuracy'].reindex(index=self.df_val.index, method='ffill')

        del df_h
        gc.collect()

        return self.df_val.copy()

    def dump(self, out_dir):
        self.df_val.to_csv(os.path.join(out_dir, 'validation.csv'))
        if len(self.df_scan) > 0:
            self.df_scan.to_csv(os.path.join(out_dir, 'scan.csv'))

    def scan(self, eval_metric, min_frac_pos=0., thr_list=None, scan2d=True, plot=False, out_dir=None, **kwargs):
        metric_v_best = 0. if eval_metric == 'accuracy' else -1.E+06

        # List of thresholds to loop over
        if thr_list is None:
            thr_list = DEFAULT_THR_LIST_SCAN

        # Initialize the scan data
        scan_data = []

        # Deafult results
        df_best = pd.DataFrame()
        for thr in thr_list:
            for thr_low in thr_list:
                if thr_low > thr:
                    continue
                if scan2d is not True and thr != thr_low:
                    continue
                # Do the validation for this point in the scan
                df_val = self.produce_validation(thr=thr, thr_low=thr_low, **kwargs)

                # Add data for this point of the scan
                frac_pos = df_val.iloc[-1, df_val.columns.get_loc('frac_pos')]
                point_dict = {'name': df_val['name'][0],
                              'thr': thr, 'thr_low': thr_low,
                              'frac_pos': frac_pos}
                for met in METRICS:
                    point_dict[met] = df_val.iloc[-1, df_val.columns.get_loc(met)]

                # If not enough positions taken -> discard
                if frac_pos < min_frac_pos:
                    for met in METRICS:
                        point_dict[met] = -1.E+06

                scan_data.append(point_dict)

                # Check the current value of the metric and eventually replace the best value
                metric_v = df_val.iloc[-1, df_val.columns.get_loc(eval_metric)]
                logger.debug('Scanning thr = {0:.2f}, '
                             'thr_low = {1:.2f}, '
                             'fraction of positions = {2:.2f}: {3} = {4:.2f}'.format(thr, thr_low, frac_pos,
                                                                                     eval_metric, metric_v))

                if frac_pos < min_frac_pos:
                    logger.debug('Discard because fraction of positions is below threshold')
                    continue

                if metric_v > metric_v_best:
                    logger.info('New best {0} = {1:.2f}'.format(eval_metric, metric_v))
                    metric_v_best = metric_v
                    df_best = df_val.copy()
                del df_val
                gc.collect()

        # keep the best
        self.df_val = df_best.copy()

        # Write the scan data
        self.df_scan = pd.DataFrame(data=scan_data)

        # Eventually, plot
        if plot is True:
            if out_dir is None:
                logger.warning('You must specify the out_dir parameter to plot the scan')
            else:
                if scan2d is True:
                    plot_scan(self.df_scan, what=eval_metric, out_dir=out_dir)
                else:
                    plot_scan_1d(self.df_scan, what=eval_metric, out_dir=out_dir)

        # Cleanup
        del df_best
        gc.collect()


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

        xs = self.df_val[0].groupby(pd.Grouper(freq=group)).agg(lambda x: x[0]).index
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
