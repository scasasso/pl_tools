# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     collection.py
#
# Project:  Predictive Layer for: pl_tools
# Author:   Stefano
# Date:     27 January 2018
#
# Scope:    The file contains a pandas wrapper for MongoDB collections
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
"""

from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure
import logging
from mongo_tools.timeseries import *
from mongo_tools.record import *
from pymongo.database import Database as mongoDB

logger = logging.getLogger(__file__)


class Record(object):
    def __init__(self, name, db_uri, db_name=None):

        self.name = name
        self.db_uri = db_uri

        # This means we have already the connection
        if db_name is None and isinstance(self.db_uri, mongoDB):
            self.conn = self.db_uri
            self.db_name = self.conn.name
            self.client = self.conn.client
            if not isinstance(self.client, MongoClient):
                self.client = self.conn.connection  # pymongo 2.8
            self.db_uri = 'mongodb://%s:%s/' % (self.client.HOST, self.client.PORT)
        elif db_name is not None:  # will connect later
            self.db_name = db_name
            self.client = None
            self.conn = None
        else:
            msg = 'You must provide either valid db_uri and db_name or a valid connection'
            logger.error(msg)
            raise AttributeError(msg)

    @classmethod
    def create_from_external(cls, _dict):
        try:
            name = _dict['table']
            db_uri = _dict['db_uri']
            db_name = _dict['db_name']
            return cls(name, db_uri, db_name)
        except KeyError as e:
            msg = 'The external doens\'t contain all the keys:\n%s' % str(e)
            logger.error(msg)
            raise KeyError(msg)

    def connect(self):
        if self.conn is not None:
            return

        try:
            self.client = MongoClient(self.db_uri)
            self.conn = self.client[self.db_name]
        except Exception as e:
            msg = 'Could not connect to the database due to the following exception:\n%s' % str(e)
            logger.error(msg)
            raise ConnectionFailure(msg)

    def close(self):
        self.client.close()

    def _get_data(self, **kwargs):

        self.data = get_record(self.conn, self.name, out_format='dataframe', **kwargs)

        return

    def get_data(self, **kwargs):
        self.connect()
        self._get_data(**kwargs)
        self.close()

        return self.data.copy()

    def put_data(self, s, **kwargs):
        self.connect()
        write_record(self.conn, self.name, s, **kwargs)
        self.close()

    def get_name(self):
        return self.name


class Collection(object):
    def __init__(self, name, db_uri, db_name=None, freq='1H', field_day='day', field_value='v', add_params=None):

        self.name = name
        self.db_uri = db_uri

        # This means we have already the connection
        if db_name is None and isinstance(self.db_uri, mongoDB):
            self.conn = self.db_uri
            self.db_name = self.conn.name
            self.client = self.conn.client
            if not isinstance(self.client, MongoClient):
                self.client = self.conn.connection  # pymongo 2.8
            self.db_uri = 'mongodb://%s:%s/' % (self.client.HOST, self.client.PORT)
        elif db_name is not None:  # will connect later
            self.db_name = db_name
            self.client = None
            self.conn = None
        else:
            msg = 'You must provide either valid db_uri and db_name or a valid connection'
            logger.error(msg)
            raise AttributeError(msg)

        self.freq = freq
        self.field_day = field_day
        self.field_value = field_value
        if add_params is None:
            self.add_params = dict()
        else:
            self.add_params = add_params

    @classmethod
    def create_from_external(cls, _dict):
        try:
            name = _dict['table']
            db_uri = _dict['db_uri']
            db_name = _dict['db_name']
            field_value = _dict['params']['fields'][0]
            if _dict['params']['keys'] is not None:
                condit = [{k: v} for k, v in zip(_dict['params']['keys_field'], _dict['params']['keys'])]
                add_params = {'$and': condit}
            else:
                add_params = None
            return cls(name, db_uri=db_uri, db_name=db_name, field_value=field_value, add_params=add_params)
        except KeyError as e:
            msg = 'The external doens\'t contain all the keys:\n%s' % str(e)
            logger.error(msg)
            raise KeyError(msg)

    def connect(self):
        if self.conn is not None:
            return

        try:
            self.client = MongoClient(self.db_uri)
            self.conn = self.client[self.db_name]
        except Exception as e:
            msg = 'Could not connect to the database due to the following exception:\n%s' % str(e)
            logger.error(msg)
            raise ConnectionFailure(msg)

    def close(self):
        self.client.close()

    def _get_data(self, dt_start, dt_end, **kwargs):

        self.data = get_daily_ts(self.conn, self.name, dt_start, dt_end,
                                 date_field=self.field_day, value_field=self.field_value,
                                 granularity=self.freq, out_format='dataframe', missing_pol='pad',
                                 add_query=self.add_params,
                                 **kwargs)

        if kwargs.get('rename', False) is True:
            self.data = self.data.rename({self.field_value: self.name}, axis=1)
        elif isinstance(kwargs.get('rename', False), str):
            self.data = self.data.rename({self.field_value: kwargs['rename']}, axis=1)

        return

    def get_data(self, dt_start, dt_end, **kwargs):
        self.connect()
        self._get_data(dt_start, dt_end, **kwargs)
        self.close()

        return self.data.copy()

    def put_data(self, s, **kwargs):
        self.connect()
        write_daily_ts(self.conn, self.name, s, self.field_day, self.field_value, add_query=self.add_params, **kwargs)
        self.close()


class CollectionND(object):
    def __init__(self, name, db_uri, db_name=None, freq='1H', field_day='day', field_value=None, add_params=None):

        self.name = name
        self.db_uri = db_uri

        # This means we have already the connection
        if db_name is None and isinstance(self.db_uri, mongoDB):
            self.conn = self.db_uri
            self.db_name = self.conn.name
            self.client = self.conn.client
            if not isinstance(self.client, MongoClient):
                self.client = self.conn.connection  # pymongo 2.8
            self.db_uri = 'mongodb://%s:%s/' % (self.client.HOST, self.client.PORT)
        elif db_name is not None:  # will connect later
            self.db_name = db_name
            self.client = None
            self.conn = None
        else:
            msg = 'You must provide either valid db_uri and db_name or a valid connection'
            logger.error(msg)
            raise AttributeError(msg)

        # Day field
        self.field_day = field_day

        # Value field
        if field_value is None:
            self.field_value = ['v']
        elif isinstance(field_value, str):
            logger.warning('String passed to field_value: a single key will be used')
            self.field_value = [field_value]
        elif isinstance(field_value, list):
            self.field_value = field_value
        else:
            msg = 'You must pass a list to the field_value parameter'
            logger.error(msg)
            raise AttributeError(msg)

        # Frequency
        self.freq = freq

        if add_params is None:
            self.add_params = dict()
        else:
            self.add_params = add_params

    @classmethod
    def create_from_external(cls, _dict):
        try:
            name = _dict['table']
            db_uri = _dict['db_uri']
            db_name = _dict['db_name']
            field_value = _dict['params']['fields'][0]
            if _dict['params']['keys'] is not None:
                condit = [{k: v} for k, v in zip(_dict['params']['keys_field'], _dict['params']['keys'])]
                add_params = {'$and': condit}
            else:
                add_params = None
            return cls(name, db_uri=db_uri, db_name=db_name, field_value=field_value, add_params=add_params)
        except KeyError as e:
            msg = 'The external doens\'t contain all the keys:\n%s' % str(e)
            logger.error(msg)
            raise KeyError(msg)

    def connect(self):
        if self.conn is not None:
            return

        try:
            self.client = MongoClient(self.db_uri)
            self.conn = self.client[self.db_name]
        except Exception as e:
            msg = 'Could not connect to the database due to the following exception:\n%s' % str(e)
            logger.error(msg)
            raise ConnectionFailure(msg)

    def close(self):
        self.client.close()

    def _get_data(self, dt_start, dt_end, **kwargs):

        self.data = get_daily_ts_multi(self.conn, self.name, dt_start, dt_end,
                                       date_field=self.field_day, value_field=self.field_value,
                                       granularity=self.freq, out_format='dataframe', missing_pol='pad',
                                       add_query=self.add_params,
                                       **kwargs)

        if kwargs.get('rename', None) is not None:
            rename_arg = kwargs['rename']
            if not isinstance(rename_arg, list):
                logger.warning('rename parameter is not a list, it will be ignored')
                pass
            else:
                rename_dict = {vf: rn for vf, rn in zip(self.field_value, rename_arg)}
                self.data = self.data.rename(rename_dict, axis=1)

        return

    def get_data(self, dt_start, dt_end, **kwargs):
        self.connect()
        self._get_data(dt_start, dt_end, **kwargs)
        self.close()

        return self.data.copy()

    def put_data(self, df, cols, **kwargs):
        self.connect()
        write_daily_ts(self.conn, self.name, df, date_field=self.field_day, value_field=self.field_value,
                       dfcol=cols,
                       add_query=self.add_params, **kwargs)
        self.close()
