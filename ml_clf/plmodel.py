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
import numpy
from pydoc import locate
from ml_tools.ml_utils import NotFittedError, clone_clf, clone_scaler, DefaultScaler
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import clone as skclone

class PLModel:
    def __init__(self, model=None, scaler='default'):
        self.model = model
        self._is_model_fitted = False
        self.scaler = scaler
        self._is_scaler_fitted = False
        self.class_obj = self.model.__class__
        
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

        return self.model.fit(X_train, y_train)

    def fit_and_eval(self, X_train, y_train):
        return self.fit(X_train, y_train)

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
