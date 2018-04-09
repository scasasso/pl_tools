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
import multiprocessing
import numpy as np
from pickle import dumps, loads
from xgboost import XGBRegressor, XGBClassifier
from sklearn.externals import joblib
import xgboost as xgb

logger = logging.getLogger(__file__)


class XGBoost:
    '''
    XGBoost wrapper class
    Meta class to provide an interface for accessing XGBoost models
    '''

    def __init__(self, booster="gbtree", n_jobs=1, learning_rate=0.3, max_depth=6,
                 min_samples_leaf=6, subsample=1, n_estimators=100, objective="reg:linear",
                 base_score=0.5, eval_metric=None, random_state=0, colsample_bytree=1, eval_set=None):
        """
        @param booster: [gbtree , gblinear]
        @param objective: ["reg:linear", "reg:logistic", "binary:logistic", "binary:logitraw", "count:poisson",
                            "multi:softmax", "multi:softprob", "rank:pairwise"]
        """
        self.n_estimators = n_estimators
        self.eval_metric = eval_metric  # rmse,logloss,error,merror,mlogloss,auc,ndcg:Normalized Discounted Cumulative Gain,"map"
        self.eval_set = eval_set
        self.param = {}
        self.param["booster"] = booster  # Can be gblinear
        self.param["silent"] = 1
        self.param["nthread"] = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.param["eta"] = learning_rate
        self.param["gamma"] = 0
        self.param["max_depth"] = max_depth
        self.param["min_child_weight"] = min_samples_leaf
        self.param["subsample"] = subsample
        self.param["colsample_bytree"] = colsample_bytree
        self.param["lambda"] = 0
        self.param["lambda_bias"] = 0
        self.param["objective"] = objective
        self.param["base_score"] = base_score
        self.param["seed"] = random_state
        return

    def fit(self, X_train, y_train):
        """
        Fit a model on the data
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        if self.param["objective"] in ["multi:softmax", "multi:softprob"]:
            self.param["num_class"] = int(max(dtrain.get_label()) + 1)
        plst = self.param.items()
        if self.eval_metric:
            if not isinstance(self.eval_metric, list):
                self.eval_metric = [self.eval_metric]
            for eval_metric in self.eval_metric:
                plst.append(("eval_metric", eval_metric))
        eval_list = [(dtrain, "train")]
        if self.eval_set is not None:
            eval_list.append((self.eval_set, 'validation'))
        self.model = xgb.train(plst, dtrain, self.n_estimators, evals=eval_list)
        self._set_feature_importances(X_train.shape[1])

        return

    def fit_and_eval(self, X_train_val, y_train_val):
        """
        Fit a model on the data
        """

        # Split train/validation dataset
        i_val = int(np.floor(0.8 * len(X_train_val)))
        X_val, y_val = X_train_val[i_val: ], y_train_val[i_val: ]
        X_train, y_train = X_train_val[: i_val], y_train_val[: i_val]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        if self.param["objective"] in ["multi:softmax", "multi:softprob"]:
            self.param["num_class"] = int(max(dtrain.get_label()) + 1)
        plst = self.param.items()
        if self.eval_metric:
            if not isinstance(self.eval_metric, list):
                self.eval_metric = [self.eval_metric]
            for eval_metric in self.eval_metric:
                plst.append(("eval_metric", eval_metric))
        eval_list = [(dtrain, "train"), (dval, "val")]
        self.model = xgb.train(plst, dtrain, self.n_estimators, evals=eval_list)
        self._set_feature_importances(X_train_val.shape[1])

        # Re-fit on all the data
        self.fit(X_train_val, y_train_val)

        return

    def predict(self, X_test):
        list_last_element = [element[-1] for element in X_test]
        all_zeros = True
        for element in list_last_element:
            if element != 0.0:
                all_zeros = False
        if all_zeros:
            X_test[0][-1] = 1
        dtest = xgb.DMatrix(X_test)
        predictions = self.model.predict(dtest)
        return predictions

    def predict_proba(self, X_test):
        dtest = xgb.DMatrix(X_test)
        if self.param.get("num_class", None):
            predictions = self.model.predict(dtest).reshape(X_test.shape[0], self.param["num_class"])
        else:
            predictions = np.array([np.array([1 - predicted_value, predicted_value]) for predicted_value in self.model.predict(dtest)])
        return predictions

    def _set_feature_importances(self, n_feat):

        # n_feat = len(self.model.get_fscore().keys())

        sum_importance = sum([v for _, v in self.model.get_fscore().iteritems()])
        self.feature_importances_ = [0.] * n_feat
        for i, (key, value) in enumerate(self.model.get_fscore().iteritems()):
            index = int(key.split("f")[-1])
            self.feature_importances_[index] = float(value) / sum_importance

        return

