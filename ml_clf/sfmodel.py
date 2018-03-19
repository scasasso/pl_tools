# -*- coding: utf-8 -*-
################################################################################
#
# File:     sfmodel.py
#
# Product:  Predictive Layer Classif Library
# Author:   Stefano  
# Date:     13 April 2018
#
# Scope:    The file contains the wrapper to scikit-learn SelectFromModel class
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
import json
import numpy
import pickle
from skmodel import SKModel
from plmodel import PLModel, clone_clf
from sklearn.externals import joblib
from sklearn.base import clone as skclone
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from ml_tools.ml_utils import SelectFromModel, NotFittedError
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from kerasmodel import DefaultScaler


class SFModel(PLModel):

    def __init__(self, model, sfm=None, teacher=None, scaler='default', sfm_type='sfm', **sfm_args):
        PLModel.__init__(self, model=model, scaler=scaler)

        if sfm_type not in ['sfm', 'rfecv']:
            raise NotImplementedError('Sfm type %s not supported' % sfm_type)

        if sfm is not None:
            if teacher is not None:
                print 'Value for arguments teacher and threshold will be neglected, as sfm argument is specified'
            self.sfm = sfm
            self.sfm_type = 'sfm' if 'SelectFromModel' in self.sfm.__class__.__name__ else 'rfecv'
            self.teacher = self.sfm.estimator_
            self.sfm_args = dict()
        elif teacher is not None:
            self.teacher = teacher
            self.sfm = None
            self.sfm_type = sfm_type
            self.sfm_args = dict(threshold='0.2*mean') if self.sfm_type == 'sfm' else dict(step=10, cv=3, verbose=2)
        else:
            print 'Teacher model has not been specified: will use linear regression default'
            self.teacher = None
            self.sfm = None
            self.sfm_type = sfm_type
            self.sfm_args = dict(threshold='0.2*mean') if self.sfm_type == 'sfm' else dict(step=10, cv=3, verbose=2)

        for k, v in sfm_args.iteritems():
            self.sfm_args[k] = v

        return

    def _extract_features(self, X_train, y_train=None):

        print 'Extracting features from teacher model'
        print 'Number of input features ', X_train.shape[1]

        if self.teacher is None and self.sfm is None:
            self.teacher = Lasso() if self.sfm_type == 'rfecv' else ElasticNet()
        if self.sfm is None:
            self.sfm = RFECV(self.teacher, **self.sfm_args) if self.sfm_type == 'rfecv' \
                else SelectFromModel(self.teacher, **self.sfm_args) 

        # Scale
        try:
            X_train_ = self.scaler.transform(X_train)
        except AttributeError as e:
            self.fit_scaler(X_train)
            X_train_ = self.scaler.transform(X_train)

        # Select
        try:
            X_train_sel = self.sfm.transform(X_train_)
        except (ValueError, NotFittedError, AttributeError) as e:
            try:
                X_train_sel = self.sfm.fit_transform(X_train_, y_train)
            except TypeError as e:
                msg = 'You are probably trying to call predict without fitting the teacher first:\n%s' % str(e)
                raise NotFittedError(msg)

        print 'Number of output features ', X_train_sel.shape[1]
        return X_train_sel

    def fit_and_eval(self, X_train, y_train):
        X_train_ = self._extract_features(X_train, y_train)
        
        return self.model.fit_and_eval(X_train_, y_train)

    def fit(self, X_train, y_train):
        X_train_ = self._extract_features(X_train, y_train)

        return self.model.fit(X_train_, y_train)

    def predict(self, X_test):
        X_test_ = self._extract_features(X_test)

        return self.model.predict(X_test_)

    def predict_proba(self, X_test):
        X_test_ = self._extract_features(X_test)

        return self.model.predict_proba(X_test_)

    def save_model(self, model_filepath):
        joblib.dump(self.sfm, model_filepath + '_sfm', compress=1)
        if self.scaler is not None:
            joblib.dump(self.scaler, model_filepath + '_sc', compress=1)
        PLModel.save_model(self, model_filepath)

    @classmethod
    def load_model(cls, model_filepath):
        sfm = joblib.load(model_filepath + '_sfm')
        model = joblib.load(model_filepath)

        if os.path.exists(model_filepath + '_sc'):
            scaler = joblib.load(model_filepath + '_sc')
        else:
            scaler = None

        return SFModel(model=model, sfm=sfm, teacher=None, scaler=scaler)

    def reset(self):

        self.sfm = clone_clf(self.sfm)
        self.teacher = clone_clf(self.teacher)

        PLModel.reset(self)
