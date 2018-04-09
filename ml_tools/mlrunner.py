import os
from copy import deepcopy
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

from mlrunner_utils import *
from ml_utils import *
import mlrunner_utils
from api_handler import ApiHandler
from mongo_tools.timeseries import get_daily_ts, write_daily_ts

logger = logging.getLogger(__file__)


class MLRunner(object):
    def __init__(self, pl_config=None):
        self.pl_config = pl_config
        self.models = self.pl_config['classifiers']
        self.target = self.pl_config['target']
        self.target_type = 'target'
        self.target_builder = lambda df1, df2: df2[self.pl_config['target']]  # takes 2 DataFrames and return a Series
        self.preds_builder = lambda a, df1, df2: a  # takes 1 np array and 2 DataFrames and return a Series
        self.df_coll = None  # df with collection timeseries
        self.df_feat = None  # df with feature timeseries
        self.model_filepath = self.pl_config['model_filepath']

        # Init the target
        if self.pl_config.get('target_transf', None) is not None:
            func, feat, tag = self.pl_config['target_transf']
            getattr(self, func)(feat)

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

    def run_build(self, rebuild_data=True, save=True, eval=False, rnt_tag='', grid_opt=False):
        if rebuild_data is True:
            data_structure = self.build_data_structure(blind=False, raise_on_missing=False)
            data_structure_train = build_data_structure_train(data_structure)
        elif self.df_feat is not None:
            data_structure_train = build_data_structure_train(self.df_feat)
        else:
            msg = 'The feature data structure is not built yet'
            logger.critical(msg)
            raise AttributeError(msg)

        # Fit the models
        self._fit_models(data_structure_train, eval=eval, grid_opt=grid_opt)

        # Feature importance
        feature_names = [f for f in self.df_feat.columns if 'target' not in f]
        self._fill_feature_importance(feature_names)

        if save:
            self._save_models(rnt_tag=rnt_tag)

        return

    def run_predict(self, write=True, field_day='day', field_value='v', rnt_tag=''):
        data_structure = self.build_data_structure(blind=True, raise_on_missing=True)
        dt_start = pd.Timestamp(self.pl_config['load_start_dt'])
        dt_end = pd.Timestamp(self.pl_config['load_end_dt']) + pd.Timedelta(hours=23) + pd.Timedelta(minutes=59)
        n_preds = len(data_structure[dt_start: dt_end])
        data_structure_test = build_data_structure_test(data_structure)

        self.models = load_models(self.model_filepath + rnt_tag)

        predictions = self._score_models(data_structure_test)
        predictions = self.preds_builder(predictions, self.df_feat, self.df_coll)

        if write:
            self._save_predictions(data_structure[:n_preds], predictions[:n_preds], field_day=field_day,
                                   field_value=field_value, rnt_tag=rnt_tag)

        return predictions[:n_preds], data_structure[:n_preds]

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

    def run_predict_dashboard(self, write=True, field_day='day', field_value='v', rnt_tag=''):

        if self.api_handler.url is None or self.api_handler.host is None:
            msg = 'You have to set both URL and HOST on the api_handler'
            logger.critical(msg)
            raise AttributeError(msg)

        predictions_, data_structure_test = self.run_predict(write=write, field_day=field_day, field_value=field_value,
                                                             rnt_tag=rnt_tag)
        datetimes, predictions = self._get_grouped_predictions(data_structure_test, predictions_)

        logger.info('Pushing predictions to dashboard')
        for dt, preds in zip(datetimes, predictions):
            logger.debug('Predict dashboard at date %s' % dt.strftime('%Y-%m-%d'))

            try:
                dict_to_push = {
                    "date": dt.strftime("%Y-%m-%dT00:00:00Z"),
                    "pred": list(preds),
                    "s": self.pl_config["pushId"] + rnt_tag,
                    "p": float(self.pl_config["horizon"]) * float(self.pl_config["granularity"]),
                }
            except KeyError as e:
                msg = 'You have to define \'pushId\', \'horizon\' and \'granularity\' in the pl_config'
                logger.critical(msg)
                raise KeyError(msg)

            self.api_handler.push_update(push_dict=dict_to_push)

    def run_update_dashboard(self, coll=None, push_key='real', db_uri=None, db_name=None, field_day=None, field_value=None, freq=None, rnt_tag=''):

        if self.api_handler.url is None or self.api_handler.host is None:
            msg = 'You have to set both URL and HOST on the api_handler'
            logger.critical(msg)
            raise AttributeError(msg)

        if coll is None:
            coll = self.pl_config['target']

        if db_uri is None or db_name is None or field_day is None or field_value is None or freq is None:
            logger.info('database uri and name not provided: will search in the externals')
            for ext in self.pl_config['externals']:
                if ext['table'] != coll:
                    continue

                # Found. Replace what is needed
                if db_uri is None:
                    db_uri = ext['db_uri']
                if db_name is None:
                    db_name = ext['db_name']
                if field_day is None:
                    field_day = ext['params']['datetime_key']
                if field_value is None:
                    field_value = ext['params']['fields'][0]
                if freq is None:
                    freq = str(ext['params']['granularity'] / 60) + 'T'
                    if freq == '60T':
                        freq = '1H'
                break

        # Instantiate client and database
        client = MongoClient(db_uri)
        db = client[db_name]

        # Get the data
        df_tmp = get_daily_ts(db, coll, self.pl_config['load_start_dt'],
                              self.pl_config['load_end_dt'] + timedelta(hours=23) + timedelta(minutes=59),
                              date_field=field_day, value_field=field_value,
                              granularity=freq, out_format='dataframe',
                              missing_pol='skip')

        logger.info('Updating the dashboard')
        for day, df_day in df_tmp.groupby(pd.Grouper(freq='1D')):
            logger.debug('Update dashboard at date %s' % day.strftime('%Y-%m-%d'))

            try:
                dict_to_push = {
                    "date": day.strftime("%Y-%m-%dT00:00:00Z"),
                    push_key: df_day['v'].tolist(),
                    "s": self.pl_config["pushId"] + rnt_tag,
                    "p": float(self.pl_config["horizon"]) * float(self.pl_config["granularity"]),
                }
            except KeyError as e:
                msg = 'You have to define \'pushId\', \'horizon\' and \'granularity\' in the pl_config'
                logger.critical(msg)
                raise KeyError(msg)

            self.api_handler.push_update(push_dict=dict_to_push)

    def build_data_structure(self, blind=False, raise_on_missing=False):
        ref_list = self.pl_config.get('reference', [])
        lats = np.array(
            [fe.get('latency', 0) for fe in self.pl_config['fields'] if not pd.isnull(fe.get('latency', None))] + [
                self.pl_config['latency']] + ref_list)
        max_lat = np.max(lats)

        dt_start = self.pl_config['load_start_dt'] - timedelta(
            days=(max_lat * self.pl_config['granularity'] / (24 * 60 * 60))) - timedelta(
            days=60)
        dt_end = self.pl_config['load_end_dt'] + timedelta(hours=23) + timedelta(minutes=59)

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
                df_tmp = func(db, self.pl_config, dt_start, dt_end, freq, params=external.get('params', {}))
            else:
                df_tmp = get_daily_ts(db, external['table'], dt_start, dt_end + timedelta(hours=23) + timedelta(minutes=59),
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
                    avg_list = self.pl_config.get('average', [])
                    if len(avg_list) == 0:
                        logger.warning('You call to Average but didn\'t set the \'average\' field')
                    for win in avg_list:
                        fname = '_'.join([field_name, feat.lower(), str(win)])
                        self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'].rolling(win,
                                                                                                min_periods=1).mean()
                elif feat == 'ExpAverage':
                    expavg_list = self.pl_config.get('exp_average', [])
                    if len(expavg_list) == 0:
                        logger.warning('You call to ExpAverage but didn\'t set the \'exp_average\' field')
                    for win in expavg_list:
                        fname = '_'.join([field_name, feat.lower(), str(win)])
                        self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'].ewm(span=win,
                                                                                            min_periods=1).mean()
                elif feat == 'Reference':
                    ref_list = self.pl_config.get('reference', [])
                    if len(ref_list) == 0:
                        logger.warning('You call to Reference but didn\'t set the \'reference\' field')
                    for lat in [v for v in list(set(ref_list)) if v > field['latency']]:
                        fname = '_'.join([field_name, feat.lower(), str(lat)])
                        self.df_feat[fname] = self.df_coll[field_name].shift(lat)
                elif feat == 'EvolutionDifference':
                    evodiff_list = self.pl_config.get('evolution_difference', [])
                    if len(evodiff_list) == 0:
                        logger.warning('You call to EvolutionDifference but didn\'t set the \'evolution_difference\' field')
                    for diff in evodiff_list:
                        fname = '_'.join([field_name, feat.lower(), str(diff)])
                        self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'] - self.df_feat[
                            field_name + '_laggedvalue'].shift(diff)
                elif feat == 'Evolution':
                    evo_list = self.pl_config.get('evolution', [])
                    if len(evo_list) == 0:
                        logger.warning(
                            'You call to Evolution but didn\'t set the \'evolution\' field')
                    for diff in evo_list:
                        fname = '_'.join([field_name, feat.lower(), str(diff)])
                        try:
                            self.df_feat[field_name + '_laggedvalue'] = self.df_feat[field_name + '_laggedvalue']
                            self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'] / self.df_feat[
                                field_name + '_laggedvalue'].shift(diff)
                        except ZeroDivisionError:
                            self.df_feat[fname] = -1
                elif feat == 'MaxPast':
                    maxpast_list = self.pl_config.get('max_past', [])
                    if len(maxpast_list) == 0:
                        logger.warning(
                            'You call to MaxPast but didn\'t set the \'max_past\' field')
                    for win in maxpast_list:
                        fname = '_'.join([field_name, feat.lower(), str(win)])
                        self.df_feat[fname] = self.df_feat[field_name + '_laggedvalue'].rolling(win,
                                                                                                min_periods=1).max()
                elif feat == 'MinPast':
                    minpast_list = self.pl_config.get('min_past', [])
                    if len(minpast_list) == 0:
                        logger.warning(
                            'You call to MinPast but didn\'t set the \'min_past\' field')
                    for win in minpast_list:
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

        # Daysoff - fill NaNs
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

    def _fit_models(self, data_structure_train, eval=False, grid_opt=False):
        # Some logs
        logger.info('Learning on : %s instances' % len(data_structure_train[0]))

        # If optimisation is run
        for index, model in enumerate(self.models):
            logger.info('Learning models: %s %s' % (round(float(index) / len(self.models), 2) * 100, '%'))
            if grid_opt is True and self.pl_config.get('classifiers_grid', None) is not None:
                raise NotImplementedError('This feature is not yet implemented')
                # grid_params = self.pl_config['classifiers_grid'][index]
                # grid_obj = GridSearchCV(estimator=model,
                #                         param_grid=grid_params,
                #                         scoring='roc_auc' if self.pl_config['learning_type'] == 'classification' else 'mean_squared_error',
                #                         n_jobs=8, cv=3, refit=True, verbose=2)
                # grid_obj.fit(data_structure_train[0], data_structure_train[1])
                #
                # # Replace
                # self.models[index] = grid_obj.best_estimator_
            else:
                if 'Keras' in model.__class__.__name__:
                    min_delta = 0.01 if ('ratio' not in self.target and 'div' not in self.target) else 0.0001
                    model.fit_and_eval(data_structure_train[0], data_structure_train[1], min_delta=min_delta)
                else:
                    if eval is True:
                        logger.debug('Will run evaluation before final fit')
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

    def _save_models(self, rnt_tag=''):
        logger.info('Save model: start')

        model_filepath = self.model_filepath + rnt_tag
        global_json = {'models_filepath': [],
                       'models_type': [],
                       'models_class': [],
                       'models_scaler': [],
                       'models_teacher': [],
                       'models_feature_importances': []}
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

            global_json['models_feature_importances'].append(model.feature_importances_)
            model.save_model('%s_%s' % (model_filepath, index))

        fw = open(model_filepath, 'w')
        fw.write(json.dumps(global_json))
        fw.close()
        logger.info('Save model: end')

        return

    def _save_predictions(self, data_structure_test, predictions, field_day='day', field_value='v', rnt_tag=''):
        client = MongoClient(self.pl_config['db_uri_running'])
        db = client[self.pl_config['db_name_running']]
        collection = db[self.pl_config['production_table'] + rnt_tag]

        datetimes, grouped_preds = self._get_grouped_predictions(data_structure_test, predictions)
        # Put into DB
        for dt, preds in zip(datetimes, grouped_preds):
            query = {field_day: dt}
            preds_day = {field_value: list(preds.astype(float))}
            collection.update(query, {'$set': preds_day}, upsert=True)

        return

    def _fill_feature_importance(self, names):
        logger.info('Setting feature importance')
        for model in self.models:
            model.set_feature_importances(names)

        return

    @staticmethod
    def _get_grouped_predictions(data_structure_test, predictions):
        # Get the dates
        datetimes = sorted(
            [datetime(dd.year, dd.month, dd.day) for dd in list(set([d.date() for d in data_structure_test.index]))])

        # Group the predictions in days
        grouped_preds = np.array_split(predictions, len(datetimes))

        # Sanity checks
        if len(grouped_preds) != len(datetimes):
            raise ValueError('Mismatch: got {0} dates and {1} predictions'.format(len(datetimes), len(grouped_preds)))
        lenghts = list(set(map(len, grouped_preds)))
        if not len(lenghts) == 1:
            raise ValueError('Not all the days have the same number of predictions:\n%s' % lenghts)

        return datetimes, grouped_preds


class MLBlender(object):
    def __init__(self, pl_config):
        self.models = []
        self.pl_config = pl_config
        self.target = self.pl_config['target']
        self.df_preds = pd.DataFrame()
        self.target_params = None
        self._init_models()

        # Needed for Crystal
        self.api_handler = ApiHandler()

    def _init_models(self):
        if 'pl_configs' not in self.pl_config:
            msg = '\'pl_configs\' key must be defined in the pl_config of the blender'
            logger.critical(msg)
            raise KeyError(msg)

        for cfg in self.pl_config['pl_configs']:
            target_ = cfg['target']
            model_params = dict()
            model_params['table'] = cfg['production_table']
            model_params['db_uri'] = cfg['db_uri_running']
            model_params['db_name'] = cfg['db_name_running']

            # Now we try to fetch the field_date, field_value and granularity
            # FIXME: could be more elegant here...
            for ext in cfg['externals']:
                if ext['table'] != target_:
                    continue

                # Granularity to freq string
                freq = str(ext['params']['granularity'] / 60) + 'T'
                if freq == '60T':
                    freq = '1H'

                # Retrieve the target info for possible future usage
                if self.target_params is None:
                    self.target_params = {}
                    self.target_params['table'] = target_
                    self.target_params['db_uri'] = ext['db_uri']
                    self.target_params['db_name'] = ext['db_name']
                    self.target_params['field_day'] = ext['params']['datetime_key']
                    self.target_params['field_value'] = ext['params']['fields'][0]
                    self.target_params['freq'] = freq

                # Found. Get what is needed
                model_params['field_day'] = ext['params']['datetime_key']
                model_params['field_value'] = ext['params']['fields'][0]
                model_params['freq'] = freq

                break

            self.models.append(model_params)

        # Sanity check
        if not len(np.unique([m['freq'] for m in self.models])) == 1:
            msg = 'You are trying to blend models with different granularity'
            logger.critical(msg)
            raise ValueError(msg)

    def _get_data(self, blind=True, rnt_tag=''):
        # Get the predictions from models in the list
        dfs_pred = []
        for model in self.models:
            # Model prediction data
            client_model = MongoClient(model['db_uri'])
            df_model = get_daily_ts(client_model[model['db_name']], model['table'] + rnt_tag,
                                    self.pl_config['load_start_dt'],
                                    self.pl_config['load_end_dt'] + timedelta(hours=23) + timedelta(minutes=59),
                                    date_field=model['field_day'], value_field=model['field_value'],
                                    granularity=model['freq'], out_format='dataframe', missing_pol='raise',
                                    verbose=0)
            df_model = df_model.rename({model['field_value']: model['table']}, axis=1)
            dfs_pred.append(df_model)

        # Concatenate predictions
        self.df_preds = pd.concat(dfs_pred, axis=1)
        self.df_preds['blend'] = self.df_preds.mean(axis=1)

        if blind is not True:
            # Model prediction data
            client_model = MongoClient(self.target_params['db_uri'])
            df_target = get_daily_ts(client_model[self.target_params['db_name']], self.target_params['table'],
                                     self.pl_config['load_start_dt'],
                                     self.pl_config['load_end_dt'] + timedelta(hours=23) + timedelta(minutes=59),
                                     date_field=self.target_params['field_day'], value_field=self.target_params['field_value'],
                                     granularity=self.target_params['freq'], out_format='dataframe', missing_pol='raise',
                                     verbose=0)
            df_target = df_target.rename({self.target_params['field_value']: 'target'}, axis=1)
            self.df_preds = pd.concat([self.df_preds, df_target], axis=1)

        return

    def run_build(self):
        msg = 'run_build is not (yet) supported for MLBlender objects. Build all the models of the ensemble separately,' \
              'call run_predict on each of them and then use run_predict/run_validate from the MLBlender objects.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def run_predict(self, write=True, field_day='day', field_value='v', rnt_tag=''):
        client = MongoClient(self.pl_config['db_uri_running'])
        db = client[self.pl_config['db_name_running']]
        collection = self.pl_config['production_table'] + rnt_tag

        self._get_data(rnt_tag=rnt_tag)

        if write:
            write_daily_ts(db, collection, self.df_preds['blend'],
                           date_field=field_day, value_field=field_value)

        return self.df_preds['blend'].copy()

    def run_validation(self, validation_type='cross_val', n_splits=5, rnt_tag=''):
        # We don't have to build the data and the models, just take the preds from database
        self._get_data(blind=False, rnt_tag=rnt_tag)

        # This is jut because of the (stupid) design of TimeSeriesSplit...
        cols = [c for c in self.df_preds.columns if c not in ['blend', 'target']]
        X = self.df_preds[cols].as_matrix()
        y = self.df_preds['target'].values
        preds = self.df_preds['blend'].values

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

            # Use only the test indices...
            y_test, predictions = y[test_index], preds[test_index]

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

    def run_predict_dashboard(self, write=True, field_day='day', field_value='v', rnt_tag=''):

        if self.api_handler.url is None or self.api_handler.host is None:
            msg = 'You have to set both URL and HOST on the api_handler'
            logger.critical(msg)
            raise AttributeError(msg)

        # Get predictions and split them by day
        predictions = self.run_predict(write=write, field_day=field_day, field_value=field_value, rnt_tag=rnt_tag)
        datetimes, predictions = self._get_grouped_predictions_from_series(predictions)

        logger.info('Pushing predictions to dashboard')
        for dt, preds in zip(datetimes, predictions):
            logger.debug('Predict dashboard at date %s' % dt.strftime('%Y-%m-%d'))

            try:
                dict_to_push = {
                    "date": dt.strftime("%Y-%m-%dT00:00:00Z"),
                    "pred": list(preds),
                    "s": self.pl_config["pushId"] + rnt_tag,
                    "p": float(self.pl_config["horizon"]) * float(self.pl_config["granularity"]),
                }
            except KeyError as e:
                msg = 'You have to define \'pushId\', \'horizon\' and \'granularity\' in the pl_config'
                logger.critical(msg)
                raise KeyError(msg)

            self.api_handler.push_update(push_dict=dict_to_push)

    def run_update_dashboard(self, coll=None, push_key='real', db_uri=None, db_name=None, field_day=None, field_value=None, freq=None, rnt_tag=''):

        if self.api_handler.url is None or self.api_handler.host is None:
            msg = 'You have to set both URL and HOST on the api_handler'
            logger.critical(msg)
            raise AttributeError(msg)

        if coll is None:
            coll = self.pl_config['target']

        if db_uri is None or db_name is None or field_day is None or field_value is None or freq is None:
            logger.info('database uri and name not provided: will search in the externals')

            # We stored already the target info in the target_params attribute
            coll = self.target_params['table']
            db_uri = self.target_params['db_uri']
            db_name = self.target_params['db_name']
            field_day = self.target_params['field_day']
            field_value = self.target_params['field_value']
            freq = self.target_params['freq']

        # Instantiate client and database
        client = MongoClient(db_uri)
        db = client[db_name]

        # Get the data
        df_tmp = get_daily_ts(db, coll, self.pl_config['load_start_dt'], self.pl_config['load_end_dt'] + timedelta(hours=23) + timedelta(minutes=59),
                              date_field=field_day, value_field=field_value,
                              granularity=freq, out_format='dataframe',
                              missing_pol='skip')

        logger.info('Updating the dashboard')
        for day, df_day in df_tmp.groupby(pd.Grouper(freq='1D')):
            logger.debug('Update dashboard at date %s' % day.strftime('%Y-%m-%d'))

            try:
                dict_to_push = {
                    "date": day.strftime("%Y-%m-%dT00:00:00Z"),
                    push_key: df_day['v'].tolist(),
                    "s": self.pl_config["pushId"] + rnt_tag,
                    "p": float(self.pl_config["horizon"]) * float(self.pl_config["granularity"]),
                }
            except KeyError as e:
                msg = 'You have to define \'pushId\', \'horizon\' and \'granularity\' in the pl_config'
                logger.critical(msg)
                raise KeyError(msg)

            self.api_handler.push_update(push_dict=dict_to_push)

    @staticmethod
    def _get_grouped_predictions_from_series(predictions):

        datetimes, grouped_preds = [], []
        for day, s_day in predictions.groupby(pd.Grouper(freq='1D')):
            datetimes.append(day.to_pydatetime())
            grouped_preds.append(list(s_day.values))

        # Sanity check
        if len(grouped_preds) != len(datetimes):
            raise ValueError('Mismatch: got {0} dates and {1} predictions'.format(len(datetimes), len(grouped_preds)))

        return datetimes, grouped_preds

