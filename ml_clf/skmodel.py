# -*- coding: utf-8 -*-
################################################################################
#
# File:     skmodel.py
#
# Product:  Predictive Layer Genius Classif Library
# Author:   Momo  
# Date:     03 April 2015
#
# Scope:    The file contains the representation of scikit learn model.
#
# Copyright (c) 2015, Predictive Layer Limited.  All Rights Reserved.
#
# The contents of this software are proprietary and confidential to the author.
# No part of this program may be photocopied,  reproduced, or translated into
# another programming language without prior written consent of the author.
#
#
# $Id$
#
################################################################################
import json
import logging
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn.base import clone as skclone
from sklearn.metrics import mean_squared_error, roc_auc_score
from plmodel import PLModel

logger = logging.getLogger(__file__)


class SKModel(PLModel):
    
    def __init__(self, model, scaler='default'):
        PLModel.__init__(self, model, scaler)
        return

    def _fit_and_eval(self, X_train_val, y_train_val):

        # Split train/validation dataset
        i_val = int(np.floor(0.8 * len(X_train_val)))
        X_val, y_val = X_train_val[i_val: ], y_train_val[i_val: ]
        X_train, y_train = X_train_val[: i_val], y_train_val[: i_val]

        # Fit the scaler
        if self._is_scaler_fitted is False:
            self.fit_scaler(X_train)
        X_train = self.scaler.transform(X_train)

        # Fit the model
        self.model.fit(X_train, y_train)

        # Validate the model
        if 'regressor' in self.model.__class__.__name__.lower():
            preds = self.model.predict(X_val)
            logger.info('Validation RMSE = {0:.4f}'.format(mean_squared_error(y_val, preds)))
        elif 'classifier' in self.model.__class__.__name__.lower():
            preds = self.model.predict_proba(X_val)
            logger.info('Validation ROC AUC = {0:.4f}'.format(roc_auc_score(y_val, preds)))
        else:
            msg = 'Cannot understand if model %s is a regressor or a classifier: will skip validation' % self.model.__class__.__name__
            logger.warning(msg)

        # Re-fit with all teh data
        self.model.fit(X_train_val, y_train_val)


