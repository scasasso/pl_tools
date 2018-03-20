import os
from copy import deepcopy
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold

from mlrunner_utils import *
from ml_utils import *
import mlrunner_utils
from api_handler import ApiHandler
from mongo_tools.timeseries import get_daily_ts

logger = logging.getLogger(__file__)


class MLRunner(object):
    def __init__(self, models=None, pl_config=None):
        self.models = models
        self.pl_config = pl_config
        self.target = self.pl_config['target']
        self.target_type = 'target'
        self.target_builder = lambda df1, df2: df2[self.pl_config['target']]  # takes 2 DataFrames and return a Series
        self.preds_builder = lambda a, df1, df2: a  # takes 1 np array and 2 DataFrames and return a Series
        self.df_coll = None  # df with collection timeseries
        self.df_feat = None  # df with feature timeseries
        self.model_filepath = self.pl_config['model_filepath']

        # Needed for Crystal
        self.api_handler = ApiHandler()

    def reset_target(self, target, target_type, target_builder, preds_builder):
        logger.debug('Changing default target to %s' % target)
        self.target = target
        self.target_type = target_type
        self.target_builder = deepcopy(target_builder)
        self.preds_builder = deepcopy(preds_builder)

        # Re-build the target
        if self.df_feat is not None and self.df_coll is not None:
            logger.debug('Calling the target builder')
            self.df_feat['target'] = self.target_builder(self.df_feat, self.df_coll)

    def project_target_onto(self, D):

        def _target_builder(df_feat_, df_coll_):
            return df_coll_[self.pl_config['target']] / df_feat_[D]

        def _preds_builder(a, df_feat_, df_coll_):
            return a * df_feat_[D].values

        self.reset_target('{orig}_div_{denom}'.format(orig=self.pl_config['target'], denom=D), 'target_on_feature',
                          _target_builder, _preds_builder)

    def subtract_from_target(self, S):

        def _target_builder(df_feat_, df_coll_):
            return df_coll_[self.pl_config['target']] - df_feat_[S]

        def _preds_builder(a, df_feat_, df_coll_):
            return a + df_feat_[S].values

        self.reset_target('{orig}_minus_{subt}'.format(orig=self.pl_config['target'], subt=S), 'target_minus_feature',
                          _target_builder, _preds_builder)

    def run_build(self, rebuild_data=True, save=True, tag=''):
        data_structure_train = None
        if rebuild_data is True:
            data_structure = self.build_data_structure(blind=False, raise_on_missing=False)
            data_structure_train = build_data_structure_train(data_structure)
        elif self.df_feat is not None:
            data_structure_train = build_data_structure_train(self.df_feat)
        else:
            msg = 'The feature data structure is not built yet'
            logger.critical(msg)
            raise AttributeError(msg)

        self._fit_models(data_structure_train)

        if save:
            self._save_models(tag=tag)

        return

    def run_predict(self, write=True, field_day='day', field_value='v', tag=''):
        data_structure = self.build_data_structure(blind=True, raise_on_missing=False)
        dt_start = pd.Timestamp(self.pl_config['load_start_dt'])
        dt_end = pd.Timestamp(self.pl_config['load_end_dt']) + pd.Timedelta(hours=23) + pd.Timedelta(minutes=59)
        n_preds = len(data_structure[dt_start: dt_end])
        data_structure_test = build_data_structure_test(data_structure)

        self.models = load_models(self.model_filepath + tag)

        predictions = self._score_models(data_structure_test)
        predictions = self.preds_builder(predictions, self.df_feat, self.df_coll)

        if write:
            self._save_predictions(data_structure[:n_preds], predictions[:n_preds], field_day=field_day,
                                   field_value=field_value, tag=tag)

        return predictions, data_structure_test

    def run_validation(self, validation_type='cross_val', n_splits=5):
        data_structure = self.build_data_structure(blind=False, raise_on_missing=False)
        X, y = build_data_structure_train(data_structure)

        scores = []
        if validation_type == 'cross_val':
            cv = KFold(len(y), n_folds=n_splits, shuffle=True)
        elif validation_type == 'timeseries_val':
            cv = TimeSeriesSplit(n=len(y), n_splits=n_splits).split(X)
        else:
            msg = 'Validation type %s is not supported: choose among \'cross_val\', \'timeseries_val\'' % validation_type
            logger.critical(msg)
            raise ValueError(msg)

        logger.info('Will run validation type {val_type} on {n} folds'.format(val_type=validation_type, n=n_splits))
        for icv, (train_index, test_index) in enumerate(cv):
            logger.info('Runnig validation fold {}'.format(icv))

            # Split the data
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            # Fit the models on the train dataset and predict on the test dataset
            self._reset_models()
            self._fit_models((X_train, y_train))
            predictions = self._score_models(X_test)

            # Transform back the target
            predictions = self.preds_builder(predictions, data_structure.iloc[test_index, :],
                                             self.df_coll.loc[data_structure.iloc[test_index, :].index, :])
            y_test = self.preds_builder(y_test, data_structure.iloc[test_index, :],
                                        self.df_coll.loc[data_structure.iloc[test_index, :].index, :])

            # Get the scores
            cv_scores = get_scores(y_test, predictions, ttype=self.pl_config['learning_type'])

            logger.info('Fold {0}, scores = {1}'.format(icv, str(cv_scores)))
            scores.append(cv_scores)

        # Print average results
        logger.info('CV average results:')
        score_str = ''
        for metric in scores[0].keys():
            score_str += '\n {metric} = {val:.4f}'.format(metric=metric, val=np.mean([d[metric] for d in scores]))
        logger.info(score_str)
        return

    def run_predict_dashboard(self, write=True, field_day='day', field_value='v', tag=''):

        if self.api_handler.url is None or self.api_handler.host is None:
            msg = 'You have to set both URL and HOST on the api_handler'
            logger.critical(msg)
            raise AttributeError(msg)

        predictions_, data_structure_test = self.run_predict(write=write, field_day=field_day, field_value=field_value, tag=tag)
        datetimes, predictions = self._get_grouped_predictions(data_structure_test, predictions_)

        for dt, preds in zip(datetimes, predictions):
            logger.debug('Predict dashboard at date %s' % dt.strftime('%Y-%m-%d'))

            try:
                dict_to_push = {
                    "date": dt.strftime("%Y-%m-%dT00:00:00Z"),
                    "pred": predictions,
                    "s": self.pl_config["pushId"],
                    "p": float(self.pl_config["horizon"]) * float(self.pl_config["granularity"]),
                }
            except KeyError as e:
                msg = 'You have to define \'pushId\', \'horizon\' and \'granularity\' in the pl_config'
                logger.critical(msg)
                raise KeyError(msg)

            self.api_handler.push_update(push_dict=dict_to_push)

    def build_data_structure(self, blind=False, raise_on_missing=False):
        lats = np.array(
            [fe.get('latency', 0) for fe in self.pl_config['fields'] if not pd.isnull(fe.get('latency', None))] + [
                self.pl_config['latency']] + self.pl_config[
                'reference'])
        max_lat = np.max(lats)

        dt_start = self.pl_config['load_start_dt'] - timedelta(
            days=(max_lat * self.pl_config['granularity'] / (24 * 60 * 60))) - timedelta(
            days=60)
        dt_end = self.pl_config['load_end_dt']

        dfs = []
        for external in self.pl_config['externals']:
            client = MongoClient(external['db_uri'])
            db = client[external['db_name']]
            params = external.get('params', {})

            granularity = params.get('granularity', self.pl_config['granularity'])
            if granularity / 60 % 60 == 0:
                freq = str(granularity / 60 / 60) + 'H'
            else:
                freq = str(granularity / 60) + 'T'

            if 'weather' in external['table'] or 'daysoff' in external['table']:
                func = getattr(mlrunner_utils, external['indicators'][0])
                df_tmp = func(db, self.pl_config, dt_start, dt_end, freq)
            else:
                df_tmp = get_daily_ts(db, external['table'], dt_start, dt_end,
                                      date_field=params['datetime_key'], value_field=params['fields'][0],
                                      granularity=freq, out_format='dataframe',
                                      missing_pol='skip')

            dfs.append(df_tmp.rename({'v': external['table']}, axis=1))

        # DataFrame with all the collections
        self.df_coll = pd.concat(dfs, axis=1)

        # DataFrame with all the features
        self.df_feat = pd.DataFrame(index=self.df_coll.index)

        # Build the features
        calendar_features = ['day_of_week', 'hour_of_day', 'month_of_year', 'week_of_year', 'day_of_month']  # others?
        not_vis = []
        for field in self.pl_config['fields']:
            field_name = field['name']

            # Setting the default latency
            if field.get('latency', None) is None:
                field['latency'] = self.pl_config['latency']

            # Keep track of the non-visible features
            if field['visibility'] is False:
                not_vis.append(field_name)

            # Add the lagged value as features for all but the calendar features
            if field_name not in calendar_features:
                field['features'] = ['LaggedValue'] + field['features']

            # Building the features
            for feat in field['features']:
                if feat == 'LaggedValue':
                    fname = '_'.join([field_name, feat.lower()])
                    self.df_feat[fname] = self.df_coll[field_name].shift(field['latency'])
                elif feat == 'Average':
                    for win in self.pl_config['average']:
                        fname = '_'.join([field_name, feat.lower(), str(win)])
                        self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'].rolling(win,
                                                                                                min_periods=1).mean()
                elif feat == 'ExpAverage':
                    for win in self.pl_config['exp_average']:
                        fname = '_'.join([field_name, feat.lower(), str(win)])
                        self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'].ewm(span=win,
                                                                                            min_periods=1).mean()
                elif feat == 'Reference':
                    for lat in [v for v in list(set(self.pl_config['reference'])) if v > field['latency']]:
                        fname = '_'.join([field_name, feat.lower(), str(lat)])
                        self.df_feat[fname] = self.df_coll[field_name].shift(lat)
                elif feat == 'EvolutionDifference':
                    for diff in self.pl_config['evolution_difference']:
                        fname = '_'.join([field_name, feat.lower(), str(diff)])
                        self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'] - self.df_feat[
                            field_name + '_laggedvalue'].shift(diff)
                elif feat == 'Evolution':
                    for diff in self.pl_config['evolution']:
                        fname = '_'.join([field_name, feat.lower(), str(diff)])
                        try:
                            self.df_feat[field_name + '_laggedvalue'] = self.df_feat[field_name + '_laggedvalue']
                            self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'] / self.df_feat[
                                field_name + '_laggedvalue'].shift(diff)
                        except ZeroDivisionError:
                            self.df_feat[fname] = -1
                elif feat == 'MaxPast':
                    for win in self.pl_config['max_past']:
                        fname = '_'.join([field_name, feat.lower(), str(win)])
                        self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'].rolling(win,
                                                                                                min_periods=1).max()
                elif feat == 'MinPast':
                    for win in self.pl_config['min_past']:
                        fname = '_'.join([field_name, feat.lower(), str(win)])
                        self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'].rolling(win,
                                                                                                min_periods=1).min()
                else:
                    raise NotImplementedError('Feature %s is not implemented yet' % feat)

        # Building the calendar features
        calendar_features = [f for f in calendar_features if f not in not_vis]
        for cfeat in calendar_features:
            if cfeat == 'day_of_week':
                self.df_feat[cfeat] = self.df_feat.index.weekday
            elif cfeat == 'day_of_month':
                self.df_feat[cfeat] = self.df_feat.index.day
            elif cfeat == 'hour_of_day':
                self.df_feat[cfeat] = self.df_feat.index.hour
            elif cfeat == 'month_of_year':
                self.df_feat[cfeat] = self.df_feat.index.month
            elif cfeat == 'week_of_year':
                self.df_feat[cfeat] = self.df_feat.index.week
            else:
                raise NotImplementedError('Feature %s is not implemented yet' % cfeat)

        # Daysoff
        for col in [c for c in self.df_coll.columns if 'daysoff' in c]:
            self.df_feat[col] = self.df_coll.loc[self.df_feat.index, col].fillna(0)

        # Attach the target
        if blind is False:
            self.df_feat['target'] = self.target_builder(self.df_feat, self.df_coll)

            # Sanity check
        if self.df_feat.loc[self.pl_config['load_start_dt']:, :].dropna().index[0].to_pydatetime() != self.pl_config[
            'load_start_dt'] and raise_on_missing is True:
            logger.debug(self.df_feat.loc[self.pl_config['load_start_dt']:, :].dropna().index[0].to_pydatetime(),
                         self.pl_config['load_start_dt'])
            msg = 'There are missing values ' \
                  'in the period {s} to {e}'.format(s=self.pl_config['load_start_dt'].strftime('%Y-%m-%d'),
                                                    e=self.pl_config['load_end_dt'].strftime('%Y-%m-%d'))
            logger.critical(msg)
            raise ValueError(msg)
        elif self.df_feat.loc[self.pl_config['load_start_dt']:, :].dropna().index[0].to_pydatetime() != self.pl_config[
            'load_start_dt']:
            # logger.debug(self.df_feat.loc[self.pl_config['load_start_dt']:, :].dropna().index[0].to_pydatetime(), self.pl_config['load_start_dt'])
            msg = 'There are missing values in the period ' \
                  '{s} to {e}'.format(s=self.pl_config['load_start_dt'].strftime('%Y-%m-%d'),
                                      e=self.pl_config['load_end_dt'].strftime('%Y-%m-%d'))
            logger.warning(msg)

        logger.info('Data structure is ready, {0} features have been built '
                    'based on {1} different signals'.format(self.df_feat.shape[1] - 1, self.df_coll.shape[1]))

        # Pragmatic approach
        idxs_inf = np.where(self.df_feat.isin([np.inf, -np.inf]))[0]
        for i in idxs_inf:
            for j, c in enumerate(self.df_feat.columns):
                if not self.df_feat.iat[i, j] in [np.inf, -np.inf]:
                    continue
                v = np.mean([self.df_feat.iat[i - 1, j], self.df_feat.iat[i + 1, j]])
                self.df_feat.iat[i, j] = v

        # Slice in time
        self.df_feat = self.df_feat.loc[self.pl_config['load_start_dt']:, :].dropna()

        return self.df_feat.copy()

    def _fit_models(self, data_structure_train):
        # Some logs
        logger.info('Learning on : %s instances' % len(data_structure_train[0]))

        for index, model in enumerate(self.models):
            logger.info('Learning models: %s %s' % (round(float(index) / len(self.models), 2) * 100, '%'))
            if 'Keras' in model.__class__.__name__:
                min_delta = 0.01 if ('ratio' not in self.target and 'div' not in self.target) else 0.0001
                model.fit_and_eval(data_structure_train[0], data_structure_train[1], min_delta=min_delta)
            elif 'XGBModel' in model.__class__.__name__:
                model.fit_and_eval(data_structure_train[0], data_structure_train[1])
            else:
                model.fit(data_structure_train[0], data_structure_train[1])
        logger.info('Learning models: %s %s' % (100, '%'))

    def _reset_models(self):
        for i, model in enumerate(self.models):
            self.models[i].reset()

        return

    def _score_models(self, data_structure_test):
        logger.info('Model prediction: start')

        predictions = []
        for index, model in enumerate(self.models):
            logger.info('scoring model %s' % str(index))
            preds = model.predict(data_structure_test)
            predictions.append(preds)

        n = np.max(map(len, predictions))
        predictions = [np.pad(a, (n - len(a), 0), 'constant', constant_values=(np.nan, np.nan)) for a in predictions]

        predictions = np.nanmean(predictions, axis=0)
        logger.info('Model prediction: end')
        return predictions

    def _save_models(self, tag):
        logger.info('Save model: start')

        model_filepath = self.model_filepath + tag
        global_json = {'models_filepath': [],
                       'models_type': [],
                       'models_class': [],
                       'models_scaler': [],
                       'models_teacher': []}
        for index, model in enumerate(self.models):
            global_json['models_filepath'].append('%s_%s' % (model_filepath, index))
            global_json['models_type'].append(self.target_type)
            global_json['models_class'].append(str(model.__class__))

            # Save info about scaler
            if getattr(model, 'scaler', None) is not None:
                global_json['models_scaler'].append(str(model.scaler.__class__))
            else:
                global_json['models_scaler'].append(None)

            # Save info about teacher
            if getattr(model, 'sfm', None) is not None:
                global_json['models_teacher'].append(str(model.sfm.__class__))
            else:
                global_json['models_teacher'].append(None)

            model.save_model('%s_%s' % (model_filepath, index))

        fw = open(model_filepath, 'w')
        fw.write(json.dumps(global_json))
        fw.close()
        logger.info('Save model: end')

        return

    def _save_predictions(self, data_structure_test, predictions, field_day='day', field_value='v', tag=''):
        client = MongoClient(self.pl_config['db_uri_running'])
        db = client[self.pl_config['db_name_running']]
        collection = db[self.pl_config['production_table'] + tag]

        datetimes, grouped_preds = self._get_grouped_predictions(data_structure_test, predictions)
        # Put into DB
        for dt, preds in zip(datetimes, grouped_preds):
            query = {field_day: dt}
            preds_day = {field_value: preds.tolist()}
            collection.update(query, {'$set': preds_day}, upsert=True)

        return

    @staticmethod
    def _get_grouped_predictions(data_structure_test, predictions):
        # Get the dates
        datetimes = sorted(
            [datetime(dd.year, dd.month, dd.day) for dd in list(set([d.date() for d in data_structure_test.index]))])

        # Group the predictions in days
        grouped_preds = np.array_split(predictions, len(datetimes))

        # Sanity check
        if len(grouped_preds) != len(datetimes):
            raise ValueError('Mismatch: got {0} dates and {1} predictions'.format(len(datetimes), len(grouped_preds)))

        return datetimes, grouped_preds


# class MLRunnerEnsemble(object):
#     def __init__(self, runner_list):
#         self.runner_list = runner_list
#         self.model_filepath = self.pl_config['model_filepath']
#         self.runner_filepaths = []
#
#     def run_build(self, **kwargs):
#         for i, runn in enumerate(self.runner_list):
#             kwargs['save'] = False
#             runn.model_filepath = runn.model_filepath + '_ens%s' % i
#             runn.run_build(**kwargs)
#             runn._save_models(runn.model_filepath)
#             self.runner_filepaths.append(runn.model_filepath)
#         self._save_models(model_filepath)
#
#     def run_predict(self, **kwargs):
#         preds_all = []
#         for i, runn in enumerate(self.runner_list):
#             kwargs['write'] = False
#             runn.model_filepath = runn.model_filepath + '_ens%s' % i
#             preds = runn.run_predict(**kwargs)
#             preds_all.append(preds)
#         preds_all = np.asarray(preds_all)
#
#         n = np.max(map(len, predictions))
#         preds_all = [np.pad(a, (n - len(a), 0), 'constant', constant_values=(np.nan, np.nan)) for a in preds_all]
#
#         preds_all = np.nanmean(preds_all, axis=0)
#
#         # Borrow data_structure from first runner (ugly)
#         data_structure_test = self.runner_list[0].df_feat.copy()
#
#     def run_validation(self):
#         pass
#
#     def _save_models(self, model_filepath):
#         logger.info('Save model: start')
#         global_json = {'models_filepath': [],
#                        'models_type': [],
#                        'models_class': [],
#                        'models_scaler': []}
#         for index, path in enumerate(self.runner_filepaths):
#             global_json['models_filepath'].append(path)
#             global_json['models_type'].append('runner')
#             global_json['models_class'].append(MLRunner.__class__.__name__)
#             global_json['models_scaler'].append(None)
#
#         fw = open(model_filepath, 'w')
#         fw.write(json.dumps(global_json))
#         fw.close()
#         logger.info('Save model: end')
