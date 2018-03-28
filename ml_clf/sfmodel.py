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
import logging
import numpy as np
import pickle
from plmodel import PLModel, clone_clf
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from ml_tools.ml_utils import SelectFromModel, NotFittedError


logger = logging.getLogger(__file__)


class SFModel(PLModel):

    def __init__(self, model, sfm=None, teacher=None, scaler='default', sfm_type='sfm', **sfm_args):
        PLModel.__init__(self, model=model, scaler=scaler)

        if sfm_type not in ['sfm']:
            raise NotImplementedError('Sfm type %s not supported' % sfm_type)

        if sfm is not None:
            if teacher is not None:
                logger.debug('Value for arguments teacher and threshold will be neglected, as sfm argument is specified')
            self.sfm = sfm
            self.sfm_type = 'sfm' if 'SelectFromModel' in self.sfm.__class__.__name__ else None
            self.teacher = self.sfm.estimator_
            self.sfm_args = dict()
        elif teacher is not None:
            self.teacher = teacher
            self.sfm = None
            self.sfm_type = sfm_type
            self.sfm_args = dict(threshold='0.2*mean') if self.sfm_type == 'sfm' else dict(step=10, cv=3, verbose=2)
        else:
            logger.info('Teacher model has not been specified: will use linear regression default')
            self.teacher = None
            self.sfm = None
            self.sfm_type = sfm_type
            self.sfm_args = dict(threshold='0.2*mean') if self.sfm_type == 'sfm' else dict(step=10, cv=3, verbose=2)

        for k, v in sfm_args.iteritems():
            self.sfm_args[k] = v

        return

    def _extract_features(self, X_train, y_train=None):

        logger.info('Extracting features from teacher model')
        logger.info('Number of input features %s' % X_train.shape[1])

        if self.teacher is None and self.sfm is None:
            self.teacher = LinearRegression()  # typically LogisticRegression(), Lasso()
        if self.sfm is None:
            self.sfm = SelectFromModel(self.teacher, **self.sfm_args)

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

        logger.info('Number of output features %s' % X_train_sel.shape[1])
        return X_train_sel

    def fit_and_eval(self, X_train, y_train, **kwargs):
        X_train_ = self._extract_features(X_train, y_train)

        # Fit
        self.model.fit_and_eval(X_train_, y_train)
        self._is_model_fitted = True

        return

    def fit(self, X_train, y_train):
        X_train_ = self._extract_features(X_train, y_train)

        # Fit
        self.model.fit(X_train_, y_train)
        self._is_model_fitted = True

        return

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

    def set_feature_importances(self, names, **kwargs):

        # Get the support features and slice the list of names
        indices = self.sfm.get_support(indices=True)

        names_sfm = np.array(names)[indices].tolist()
        PLModel.set_feature_importances(self, names_sfm, **kwargs)

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
