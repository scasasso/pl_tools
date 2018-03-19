# -*- coding: utf-8 -*-
################################################################################
#
# File:     kerasmodel.py
#
# Product:  Predictive Layer Classif Library
# Author:   Stefano
# Date:     13 March 2018
#
# Scope:    The file contains the wrapper of keras models.
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
from sklearn.externals import joblib
import numpy as np
from copy import deepcopy

from plmodel import PLModel, DefaultScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.initializers import Constant
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

from modelcontainer import DefaultModelContainer


class KerasModelContainer(DefaultModelContainer):

    def __init__(self, model=None, task='regression', metric='mean_squared_error'):
        DefaultModelContainer.__init__(self, model)
        self.model = model

        # Set the metric
        if metric in ['RMSE', 'mean_squared_error']:
            self.loss = 'mean_squared_error'
            self.task = 'regression'
            self.last_act = 'linear'
        elif metric in ['MAE', 'mean_absolute_error']:
            self.loss = 'mean_absolute_error'
            self.task = 'regression'
            self.last_act = 'linear'
        elif metric in ['MAPE', 'mean_absolute_percentage_error']:
            self.loss = 'mean_absolute_percentage_error'
            self.task = 'regression'
            self.last_act = 'linear'
        elif metric in ['RMSLE', 'mean_squared_logarithmic_error']:
            self.loss = 'mean_squared_logarithmic_error'
            self.task = 'regression'
            self.last_act = 'linear'
        elif 'crossentropy' in metric:
            self.loss = metric
            self.task = 'classification'
            self.last_act = 'sigmoid'
        elif task == 'classification':
            self.loss = 'categorical_crossentropy'
            self.task = 'classification'
            self.last_act = 'softmax'
        else:
            raise NotImplementedError('Learning type {lt} and metric {me} not supported.'.format(lt=task, me=metric))

    def build_model(self, n_feats):
        return self.model


class KerasMLP(KerasModelContainer):

    def __init__(self, model=None, task='regression', metric='mean_squared_error',
                 dense_units=20, dropout=0.2, lr=0.001):
        KerasModelContainer.__init__(self, model, task, metric)

        # Dense layer options
        self.dense_units = dense_units
        self.dropout = dropout
        self.lr = lr

        return

    def build_model(self, n_feats):
        self.model = Sequential()

        # Adding the dense layer
        self.model.add(Dense(input_shape=(n_feats, ), units=self.dense_units, bias_initializer=Constant(1.E-04),
                             kernel_initializer='glorot_normal', activation='relu'))

        self.model.add(Dropout(self.dropout))

        # Adding the output layer
        self.model.add(Dense(units=1, bias_initializer=Constant(1.E-04),
                             kernel_initializer='glorot_normal', activation=self.last_act))

        # Compiling the MLP
        self.model.compile(optimizer=Adam(lr=self.lr), loss=self.loss)

        # Summary
        print self.model.summary()

        return self.model


class KerasMLP2(KerasModelContainer):

    def __init__(self, model=None, task='regression', metric='mean_squared_error',
                 dense_units=(100, 20), dropout=(0.3, 0.3), lr=0.001):
        KerasModelContainer.__init__(self, model, task, metric)

        # LSTM layer options
        self.dense_units = dense_units
        self.dropout = dropout
        self.lr = lr

        return

    def build_model(self, n_feats):

        self.model = Sequential()

        # Adding the dense layer
        self.model.add(Dense(input_shape=(n_feats, ), units=self.dense_units[0], bias_initializer=Constant(1.E-04),
                             kernel_initializer='glorot_normal', activation='relu'))

        self.model.add(Dropout(self.dropout[0]))

        # One more layer
        self.model.add(Dense(units=self.dense_units[1], bias_initializer=Constant(1.E-04),
                             kernel_initializer='glorot_normal', activation='relu'))

        self.model.add(Dropout(self.dropout[1]))

        # Adding the output layer
        self.model.add(Dense(units=1, bias_initializer=Constant(1.E-04),
                             kernel_initializer='glorot_normal', activation=self.last_act))

        # Compiling the MLP
        self.model.compile(optimizer=Adam(lr=self.lr), loss=self.loss)

        # Summary
        print self.model.summary()

        return self.model


class KerasLSTM(KerasModelContainer):

    def __init__(self, model=None, task='regression', metric='mean_squared_error',
                 n_lookback=1, lstm_units=20):
        KerasModelContainer.__init__(self, model, task, metric)

        # LSTM layer options
        self.lstm_units = lstm_units
        self.n_lookback = n_lookback

        return

    def build_model(self, n_feats):

        self.model = Sequential()

        # Adding the input layerand the LSTM layer
        self.model.add(LSTM(units=self.lstm_units, activation='sigmoid', recurrent_activation='hard_sigmoid',
                            input_shape=(self.n_lookback, n_feats), dropout=0.2, recurrent_dropout=0.2)
                       )

        # Adding the output layer
        self.model.add(Dense(units=1, bias_initializer=Constant(1.E-04),
                             kernel_initializer='glorot_normal', activation=self.last_act))

        # Compiling the RNN
        self.model.compile(optimizer='adam', loss=self.loss)

        # Summary
        print self.model.summary()

        return self.model


class KerasLSTM2(KerasModelContainer):

    def __init__(self, model=None, task='regression', metric='mean_squared_error',
                 n_lookback=1, lstm_units=(20, 5)):
        KerasModelContainer.__init__(self, model, task, metric)

        # LSTM layer options
        self.lstm_units = lstm_units
        self.n_lookback = n_lookback

        return

    def build_model(self, n_feats):
        # tf.reset_default_graph()
        self.model = Sequential()

        # Adding the input layerand the LSTM layer
        self.model.add(LSTM(units=self.lstm_units[0], activation='sigmoid', recurrent_activation='hard_sigmoid',
                            input_shape=(self.n_lookback, n_feats), dropout=0.0, recurrent_dropout=0.0,
                            return_sequences=True)
                       )

        # Adding the input layerand the LSTM layer
        self.model.add(LSTM(units=self.lstm_units[1], activation='sigmoid', recurrent_activation='hard_sigmoid',
                            dropout=0.0, recurrent_dropout=0.0))

        # Adding the output layer
        self.model.add(Dense(units=1, bias_initializer=Constant(1.E-04),
                             kernel_initializer='glorot_normal', activation=self.last_act))

        # Compiling the RNN
        self.model.compile(optimizer='adam', loss=self.loss)

        # Summary
        print self.model.summary()

        return self.model


class KerasLSTM3(KerasModelContainer):

    def __init__(self, model=None, task='regression', metric='mean_squared_error',
                 n_lookback=1, lstm_units=20, dense_units=10):
        KerasModelContainer.__init__(self, model, task, metric)

        # LSTM layer options
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.n_lookback = n_lookback

        return

    def build_model(self, n_feats):
        # tf.reset_default_graph()
        self.model = Sequential()

        # Adding the input layerand the LSTM layer
        self.model.add(LSTM(units=self.lstm_units, activation='sigmoid', recurrent_activation='hard_sigmoid',
                            input_shape=(self.n_lookback, n_feats), dropout=0.2, recurrent_dropout=0.2,
                            return_sequences=False)
                       )

        # Adding the output layer
        self.model.add(Dense(**{'units': self.dense_units, 'bias_initializer': Constant(1.E-04),
                                'kernel_initializer': 'glorot_normal', 'activation': 'relu'}))

        self.model.add(Dropout(0.25))

        # Adding the output layer
        self.model.add(Dense(units=1, bias_initializer=Constant(1.E-04),
                             kernel_initializer='glorot_normal', activation=self.last_act))

        # Compiling the RNN
        self.model.compile(optimizer='adam', loss=self.loss)

        # Summary
        print self.model.summary()

        return self.model


class KerasModel(PLModel):

    def __init__(self, model_container, nb_epochs=5, **model_args):
        PLModel.__init__(self, **model_args)
        self.model_container = model_container
        self.nb_epochs = nb_epochs
        self.fit_res = None
        
        return

    def fit_and_eval(self, X_train, y_train, f_val=0.1, early_stop=True, min_delta=0.1, use_best=True):
        # Scale
        try:
            X_train_ = self.scaler.transform(X_train)
        except AttributeError as e:
            self.fit_scaler(X_train)
            X_train_ = self.scaler.transform(X_train)

        # Get lookback from keras model
        if getattr(self.model_container, 'n_lookback', None) is None:
            self.model_container.n_lookback = None
            if self.model is not None and len(self.model.get_input_shape_at(0)) > 2:
                self.model_container.n_lookback = self.model.get_input_shape_at(0)[1]

        if self.model_container.n_lookback is not None:
            # Prepare batches
            X_batches, y_batches = [], []
            for i in range(self.model_container.n_lookback, X_train_.shape[0]):
                xbatch = X_train_[i - self.model_container.n_lookback: i, :]
                ybatch = y_train[i - 1]
                X_batches.append(xbatch)
                y_batches.append(ybatch)
            X_train_ = np.array(X_batches)
            y_train = np.array(y_batches)

        # Split train/val
        ival = int(np.floor((1 - f_val) * X_train_.shape[0]))
        X_val, y_val = X_train_[ival:], y_train[ival:]
        X_train_, y_train = X_train_[:ival], y_train[:ival]

        # Build and fit the model
        callbacks = []
        if early_stop:
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=2, verbose=1, mode='auto')]
        callbacks.append(ModelCheckpoint(filepath='/tmp/weights.h5', verbose=1, save_best_only=True))

        if self.model is None:
            self.set_model(self.model_container.build_model(X_train_.shape[1]))
        self.fit_res = self.model.fit(X_train_, y_train, epochs=self.nb_epochs, validation_data=(X_val, y_val),
                                      verbose=2, callbacks=callbacks)

        if use_best:
            print 'Taking the best model from: /tmp/weights.h5'
            self.model = load_model('/tmp/weights.h5')

        return self.fit_res

    def fit(self, X_train, y_train):
        # Scale
        try:
            X_train_ = self.scaler.transform(X_train)
        except AttributeError as e:
            self.fit_scaler(X_train)
            X_train_ = self.scaler.transform(X_train)

        # Get lookback from keras model
        if getattr(self.model_container, 'n_lookback', None) is None:
            self.model_container.n_lookback = None
            if self.model is not None and len(self.model.get_input_shape_at(0)) > 2:
                self.model_container.n_lookback = self.model.get_input_shape_at(0)[1]

        if self.model_container.n_lookback is not None:
            # Prepare batches
            X_batches, y_batches = [], []
            for i in range(self.model_container.n_lookback, X_train_.shape[0]):
                xbatch = X_train_[i - self.model_container.n_lookback: i, :]
                ybatch = y_train[i - 1]
                X_batches.append(xbatch)
                y_batches.append(ybatch)
            X_train_ = np.array(X_batches)
            y_train = np.array(y_batches)

        # Build and fit the model
        if self.model is None:
            self.set_model(self.model_container.build_model(X_train_.shape[1]))
        self.fit_res = self.model.fit(X_train_, y_train, epochs=self.nb_epochs, validation_split=0., verbose=2)

        return self.fit_res

    def predict(self, X):
        if self.model is None:
            self.set_model(self.model_container.build_model(X.shape[1]))

        # Scale
        X = self.scaler.transform(X)

        # Get lookback from keras model
        if getattr(self.model_container, 'n_lookback', None) is None:
            self.model_container.n_lookback = None
            if self.model is not None and len(self.model.get_input_shape_at(0)) > 2:
                self.model_container.n_lookback = self.model.get_input_shape_at(0)[1]

        if self.model_container.n_lookback is not None:
            # Prepare batches
            X_batches = []
            for i in range(self.model_container.n_lookback, X.shape[0]):
                xbatch = X[i - self.model_container.n_lookback: i, :]
                X_batches.append(xbatch)
            X = np.array(X_batches)

        # Score
        y_pred = self.model.predict(X)

        return y_pred.ravel()

    def predict_proba(self, data_structure_data):
        pass

    def save_model(self, model_filepath):
        # if self.model is None:
        #     self.model = self.model_container.build_model()
        self.model.save(model_filepath)
        joblib.dump(self.scaler, model_filepath + '_sc', compress=1)

    @classmethod
    def load_model(cls, model_filepath):
        keras_model = load_model(model_filepath)
        keras_loss = keras_model.loss

        model_inst = cls(KerasModelContainer(model=keras_model, metric=keras_loss))
        model_inst.scaler = joblib.load(model_filepath + '_sc')

        return model_inst
