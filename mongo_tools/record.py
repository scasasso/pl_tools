# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     record.py
#
# Project:  Predictive Layer for: pl_tools
# Author:   Stefano
# Date:     14 September 2018
#
# Scope:    The file contains a utilities functions to transform MongoDB records into Pandas Dataframe
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
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__file__)


def get_record(db, coll_name, out_format='dataframe', add_query=None, sort=None, verbose=0):
    # Check arguments
    allowed_out_formats = ['list', 'dataframe']

    if verbose > 1:
        logger.info(' Database instance:\n%s' % str(db))

    if verbose > 0:
        logger.info('Will fetch collection {coll}.'
                    '\nAdditional query parameters {add_query}'.format(coll=coll_name,
                                                                       add_query=add_query))
    # Get collection
    collection = db[coll_name]

    # Query the database
    and_list = []
    if add_query is not None:
        if isinstance(add_query, list):
            and_list.extend(add_query)
        elif isinstance(add_query, dict):
            and_list.append(add_query)
        else:
            msg = 'add_query parameter must be either a list of dict or a dict'
            logger.error(msg)
            raise ValueError(msg)
        query = {"$and": and_list}
    else:
        query = {}

    if verbose > 1:
        logger.info(' Total query:\n%s' % str(query))

    if sort is None:
        res = collection.find(query)
    elif isinstance(sort, tuple):
        res = collection.find(query).sort(*sort)
    else:
        msg = 'Argument sort must be a tuple, not {}'.format(type(sort))
        logger.error(msg)
        raise ValueError(msg)

    if verbose > 0:
        logger.info('Found {0} results'.format(res.count()))

    res_as_list = list(res)

    if out_format == 'list':
        return res_as_list
    elif out_format == 'dataframe':
        df = pd.DataFrame(data=res_as_list)
        return df
    else:
        msg = 'Format {fo} is not supported. Valid ones are:\n {l}'.format(fo=out_format, l=allowed_out_formats)
        logger.error(msg)
        raise NotImplementedError(msg)


def to_pydatetime(r):
    for c in r:
        if isinstance(r[c], pd._libs.tslib.Timestamp):
            r[c] = r[c].to_pydatetime()

    return r


def write_record(db, coll_name, df, subset=None):
    # Default
    if subset is None:
        subset = df.keys().tolist()
    if isinstance(df, dict):
        df = [df]
    elif isinstance(df, pd.core.series.Series):
        df = [df.to_dict()]

    if isinstance(df, pd.core.frame.DataFrame):
        for _, row in df[subset].iterrows():
            row = to_pydatetime(row)
            # Write to DB
            db[coll_name].insert(row)
    elif isinstance(df, list):
        for _row in df:
            row = {k: v for k, v in _row.iteritems() if k in subset}
            row = to_pydatetime(row)
            db[coll_name].insert(row)
    else:
        msg = 'Parameter df must be either a pandas DataFrame, a dictionary or a list of dictionaries'
        logger.error(msg)
        raise ValueError(msg)

    return
