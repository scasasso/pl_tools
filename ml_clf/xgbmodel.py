# -*- coding: utf-8 -*-
################################################################################
#
# File:     xgboost_wrapper.py
#
# Product:  Predictive Layer Genius Classif Library
# Author:   Momo
# Date:     12 May 2015
#
# Scope:    The file contains the PLModel class which provides functions
#           to handle Predictive Layer Model based on PLData
#
# Copyright (c) 2015, Predictive Layer Limited.  All Rights Reserved.
#
# The contents of this software are proprietary and confidential to the author.
# No part of this program may be photocopied,  reproduced, or translated into
# another programming language without prior written consent of the author.
#
#
#
################################################################################
import logging
from plmodel import PLModel

logger = logging.getLogger(__file__)


class XGBModel(PLModel):
    def __init__(self, model, scaler='default'):
        PLModel.__init__(self, model, scaler)
        return

    def _fit_and_eval(self, X_train_val, y_train_val):
        return self.model.fit_and_eval(X_train_val, y_train_val)
