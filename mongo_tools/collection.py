import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure
import logging
from mongo_tools.timeseries import get_daily_ts, write_daily_ts
from pymongo.database import Database as mongoDB

logger = logging.getLogger(__file__)


class Collection(object):
    def __init__(self, name, db_uri, db_name=None, freq='1H', field_day='day', field_value='v', add_params=None):

        self.name = name
        self.db_uri = db_uri

        # This means we have already the connection
        if db_name is None and isinstance(self.db_uri, mongoDB):
            self.conn = self.db_uri
            self.db_name = self.conn.name
            self.client = self.conn.client
            self.db_uri = 'mongodb://%s:%s/' % (self.client.HOST, self.client.PORT)
        elif db_name is not None:  # will connect later
            self.db_name = db_name
            self.client = None
            self.conn = None
            self.coll = None
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
            return cls(name, db_uri, db_name, field_value, add_params)
        except KeyError as e:
            msg = 'The external doens\'t contain all the keys:\n%s' % str(e)
            logger.error(msg)
            raise KeyError(msg)

    def connect(self):
        if self.conn is not None:
            return

        try:
            self.client = MongoClient(self.db_uri)
            self.conn = self.client[self.db_name][self.name]
        except Exception as e:
            msg = 'Could not connect to the database due to the following exception:\n%s' % str(e)
            logger.error(msg)
            raise ConnectionFailure(msg)

    def close(self):
        self.client.close()

    def _get_data(self, **kwargs):

        try:
            dt_start = kwargs['dt_start']
            dt_end = kwargs['dt_end']
        except KeyError:
            msg = 'You must specify \"dt_start\" and \"dt_end\" arguments'
            logger.error(msg)
            raise KeyError(msg)

        self.data = get_daily_ts(self.conn, self.name, dt_start, dt_end,
                                 date_field=self.field_day, value_field=self.field_value,
                                 granularity=self.freq, out_format='dataframe', missing_pol='raise',
                                 add_query=self.add_params,
                                 verbose=1)

        if kwargs.get('rename', False) is True:
            self.data = self.data.rename({self.field_value: self.name}, axis=1)
        elif isinstance(kwargs.get('rename', False), str):
            self.data = self.data.rename({self.field_value: kwargs['rename']}, axis=1)

        return

    def get_data(self, **kwargs):
        self.connect()
        self._get_data(**kwargs)
        self.close()

        return self.data.copy()

    def put_data(self, s):
        self.connect()
        write_daily_ts(self.conn, self.name, s, self.field_day, self.field_value, add_query=self.add_params)
        self.close()
