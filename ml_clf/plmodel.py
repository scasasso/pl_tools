# -*- coding: utf-8 -*-
################################################################################
#
# File:     plmodel.py
#
# Product:  Predictive Layer Classif Library
# Author:   Stefano  
# Date:     13 March 2018
#
# Scope:    The file contains the wrapper to scikit-learn, xgboost, keras models.
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
import os
import json
import logging
import numpy
from pydoc import locate
from ml_tools.ml_utils import NotFittedError, clone_clf, clone_scaler, DefaultScaler
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import clone as skclone

logger = logging.getLogger(__file__)


class PLModel:
    def __init__(self, model=None, scaler='default'):
        self.model = model
        self._is_model_fitted = False
        self.scaler = scaler
        self._is_scaler_fitted = False
        self.class_obj = self.model.__class__
        self.feature_importances_ = None
        
    def set_model(self, model):
        self.model = model
        self._is_scaler_fitted = False

    def fit_scaler(self, X_train):

        if self.scaler == 'minmax' or 'MinMaxScaler' in str(self.scaler.__class__):
            self.scaler = MinMaxScaler()
        elif self.scaler == 'std' or 'StandardScaler' in str(self.scaler.__class__):
            self.scaler = StandardScaler()
        elif self.scaler == 'default' or 'DefaultScaler' in str(self.scaler.__class__):
            self.scaler = DefaultScaler()
        else:
            raise NotImplementedError('Scaler %s not supported' % self.scaler)
        self.scaler.fit(X_train)
        self._is_scaler_fitted = True

    def fit(self, X_train, y_train):
        if self._is_scaler_fitted is False:
            self.fit_scaler(X_train)
        X_train = self.scaler.transform(X_train)

        # Fit
        self.model.fit(X_train, y_train)
        self._is_model_fitted = True

        return

    # This should be implemented separately in each case
    def _fit_and_eval(self, X_train_val, y_train_val, **kwargs):
        if self._is_scaler_fitted is False:
            self.fit_scaler(X_train_val)
        X_train_val = self.scaler.transform(X_train_val)

        # Fit
        self.fit(X_train_val, y_train_val)
        self._is_model_fitted = True

        return

    def fit_and_eval(self, X_train_val, y_train_val, **kwargs):

        # Fit
        self._fit_and_eval(X_train_val, y_train_val, **kwargs)
        self._is_model_fitted = True

        return

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)

        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        X_test = self.scaler.transform(X_test)

        return self.model.predict_proba(X_test)
        
    def save_model(self, model_filepath):
        if self.scaler not in [None, 'default']:
            joblib.dump(self.scaler, model_filepath + '_sc', compress=1)
        joblib.dump(self.model, model_filepath, compress=1)

    def set_feature_importances(self, names, **kwargs):
        # Sanity check
        if self._is_model_fitted is not True:
            msg = 'You must fit before setting feature importance'
            logger.error(msg)
            raise NotFittedError(msg)

        try:
            self.feature_importances_ = [(name, round(float(imp), 5)) for name, imp in zip(names, self.model.feature_importances_)]
            # Sort
            self.feature_importances_ = sorted(self.feature_importances_, key=lambda x: x[1], reverse=True)

        except AttributeError as e:
            msg = 'Classifier %s doesn\'t have a feature_importances_ attribute. Will set dummy values.' % self.model.__class__.__name__
            logger.warning(msg)
            default_imp = round(1. / len(names), 5)
            self.feature_importances_ = [(name, default_imp) for name in names]

        return
        
    @classmethod
    def load_model(cls, model_filepath):
        model = joblib.load(model_filepath)

        scaler = None
        if os.path.exists(model_filepath + '_sc'):
            scaler = joblib.load(model_filepath + '_sc')

        return cls(model=model, scaler=scaler)

    def reset(self):
        self.model = clone_clf(self.model)
        self._is_model_fitted = False
        self.scaler = clone_scaler(self.scaler)
        self._is_scaler_fitted = False
