# -*- coding: utf-8 -*-
################################################################################
#
# File:     arimamodel.py
#
# Product:  Predictive Layer ML Library
# Author:   Stefano
# Date:     27 February 2018
#
# Scope:    The file contains the representation of keras model.
#
# Copyright (c) 2018, Predictive Layer Limited.  All Rights Reserved.
#
# The contents of this software are proprietary and confidential to the author.
# No part of this program may be photocopied,  reproduced, or translated into
# another programming language without prior written consent of the author.
#
#
# $Id$
#
################################################################################

import os
from pandas import isnull
from sklearn.externals import joblib
import numpy as np
from numpy.linalg import LinAlgError
from copy import deepcopy
from statsmodels.tsa.arima_model import ARIMA
from ml_clf.plmodel import PLModel
from ml_clf.plmodel import DefaultScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.externals.joblib import Parallel, delayed


class ArimaModel(PLModel):

    def __init__(self, params=(5, 1, 1), lookback=24 * 30, max_iter=400, retry=True, fallbacks=None, verbose=0, n_jobs=1):
        PLModel.__init__(self, model=RandomForestRegressor(), scaler=DefaultScaler())

        self.params = params
        self.lookback = lookback
        self.model = None
        self.model_fit = None
        self.fit_res_dict = None
        self.max_iter = max_iter
        self.retry = retry
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.feature_rank = None
        self.fit_results = []
        if fallbacks is None:
            self.fallbacks = []
        else:
            self.fallbacks = fallbacks

        # These do not make sense for Arima: still we have to set them
        self._is_model_fitted = True
        self._is_scaler_fitted = True

        return

    def fit(self, X_train, y_train):
        pass

    def _fit(self):
        if self.model is None:
            raise AttributeError('Call to fit is not possible before initializing the model')
        return self.model.fit(disp=0, maxiter=self.max_iter)

    def _check_fit_result(self, fit_res):
        if self.retry and (fit_res.get('converged', False) is False or fit_res.get('warnflag', 3) != 0):
            raise ArimaConvergenceError('Fit did not converge!')
        return True

    def _compute_feature_rank(self, features_information):
        pass

    def predict_serial(self, X, lag=24):

        # Output predictions
        y_pred = []

        # Warning
        if self.verbose > 0:
            print 'Lookback is set to %s so there will not be any prediction for the first %s points' % (self.lookback, self.lookback)

        # Prepare the batches
        default_fit_res = {'converged': False, 'gopt': np.array([]), 'fcalls': 0, 'warnflag': 3, 'fopt': None}
        X_batches = []
        for i in range(self.lookback):
            # X_batches.append(X[:self.lookback, :].ravel())  # We can't make predictions on them anyway
            y_pred.append(np.nan)
            self.fit_results.append(default_fit_res)
        for i in range(self.lookback, X.shape[0]):
            xbatch = X[i - self.lookback + 1: i + 1, :].ravel()
            X_batches.append(xbatch)
        X = np.array(X_batches)

        # Run the fit for each batch
        for ib, batch in enumerate(X):
            if self.verbose > 0 and ib % 10 == 0:
                print 'Fitting batch {0}/{1}, {2:.2f}% done'.format(ib + 1, len(X), 100 * float(ib) / len(X))
            n_tot = self.lookback
            is_fit_ok = False
            fit_res_dict = default_fit_res
            while n_tot > 10:
                try:
                    self.model = ARIMA(batch[-n_tot:], self.params)
                    self.model_fit = self._fit()
                    fit_res_dict = self.model_fit.mle_retvals
                    is_fit_ok = self._check_fit_result(fit_res_dict)
                    self.fit_results.append(fit_res_dict)
                    break
                except (LinAlgError, ValueError, AttributeError, ArimaConvergenceError) as e:
                    n_new = n_tot - int(max(np.floor(0.05 * n_tot), 1.))
                    if self.verbose > 1:
                        print 'Got the following exception:\n{exc}\n' \
                              'Will reduce the number of training samples from ' \
                              '{n_before} to {n_now}...'.format(exc=str(e), n_before=n_tot, n_now=n_new)
                    n_tot = n_new

            if is_fit_ok is not True:
                raise ArimaConvergenceError('All the attempts to fit failed miserably.')

            if self.verbose > 1:
                print 'Results of the fit:\n%s' % str(fit_res_dict)
            output = self.model_fit.forecast(steps=lag)
            y_pred.append(output[0][-1])

        return np.array(y_pred)

    def predict(self, X, lag=24):

        # Output predictions
        y_pred = []

        # Warning
        if self.verbose > 0:
            print 'Lookback is set to %s so there will not be any prediction for the first %s points' % (self.lookback, self.lookback)

        # Prepare the batches
        default_fit_res = {'converged': False, 'gopt': np.array([]), 'fcalls': 0, 'warnflag': 3, 'fopt': None}
        X_batches = []
        for i in range(self.lookback):
            # X_batches.append(X[:self.lookback, :].ravel())  # We can't make predictions on them anyway
            y_pred.append(np.nan)
            self.fit_results.append(default_fit_res)
        for i in range(self.lookback, X.shape[0]):
            xbatch = X[i - self.lookback + 1: i + 1, :].ravel()
            X_batches.append(xbatch)
        X = np.array(X_batches)

        # Run the fit for each batch
        params = [self.params] + self.fallbacks
        y_pred = Parallel(n_jobs=self.n_jobs, verbose=self.verbose + 10)(
            delayed(_predict_routine)(
                params, ib=ib, n_tot=len(X), default_fit_res=default_fit_res, batch=batch, lag=lag,
                lookback=self.lookback, retry=self.retry, max_iter=self.max_iter, verbose=self.verbose)
            for ib, batch in enumerate(X)
        )

        return np.array(y_pred)

    def predict_proba(self, data_structure_data):
        pass

    def set_default_value(self, default_value):
        pass

    def set_feature_rank(self, feature_rank):
        self.feature_rank = feature_rank
        return

    def save_model(self, model_filepath):
        joblib.dump(self, model_filepath, compress=1)
        pass

    @classmethod
    def load_model(cls, model_filepath):
        return joblib.load(model_filepath)

    @classmethod
    def create_from_json(cls, model_filepath):
        pass


def _predict_routine(param_list, **kwargs):

    for params in param_list:
        try:
            res = _predict_batch_parallel(params, **kwargs)
            return res
        except Exception as e:
            print 'Got following exception, will scroll down the fallback list\n%s' % str(e)

    raise ArimaConvergenceError('All the attempts to fit failed miserably.')


def _predict_batch_parallel(params, ib, n_tot, default_fit_res, batch, lag=24, lookback=24 * 5, retry=True, max_iter=200, verbose=0):

    print 'Fitting batch {0}/{1}, {2:.2f}% done'.format(ib + 1, n_tot, 100 * float(ib) / n_tot)

    n_tot = lookback
    is_fit_ok, model_fit = False, None
    fit_res_dict = default_fit_res
    while n_tot > 10:
        try:
            model = ARIMA(batch[-n_tot:], params)
            model_fit = model.fit(disp=0, maxiter=max_iter)
            fit_res_dict = model_fit.mle_retvals
            if retry and (fit_res_dict.get('converged', False) is False or
                          fit_res_dict.get('warnflag', 3) != 0 or
                          isnull(fit_res_dict.get('fopt', np.nan)) is True):
                raise ArimaConvergenceError('Fit did not converge!')
            is_fit_ok = True

            break
        except (LinAlgError, ValueError, AttributeError, ArimaConvergenceError) as e:
            n_new = n_tot - int(max(np.floor(0.05 * n_tot), 1.))
            if verbose > 1:
                print 'Got the following exception:\n{exc}\n' \
                      'Will reduce the number of training samples from ' \
                      '{n_before} to {n_now}...'.format(exc=str(e), n_before=n_tot, n_now=n_new)
            n_tot = n_new

    if is_fit_ok is not True:
        raise ArimaConvergenceError('All the attempts to fit failed miserably.')

    if verbose > 1:
        print 'Results of the fit:\n%s' % str(fit_res_dict)
    output = model_fit.forecast(steps=lag)

    # Result
    res = output[0][-1]

    # Check
    if isnull(res) or np.abs(res) > 100 * np.max(np.abs(batch)):
        raise ArimaConvergenceError('Result is NaN or NoN(sense): %s' % res)

    return res


class ArimaConvergenceError(Exception):
    pass
