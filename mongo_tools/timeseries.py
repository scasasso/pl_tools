# -*- coding: utf-8 -*-
"""
################################################################################
#
# File:     timeseries.py
#
# Project:  Predictive Layer for: pl_tools
# Author:   Stefano
# Date:     27 January 2018
#
# Scope:    The file contains a utilities functions to transform MongoDB collections into Pandas Dataframe
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
from datetime import timedelta, datetime
from pytz import timezone
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__file__)


def get_day_saving_time_day(tz='Europe/Paris', year=2018):
    results = []
    try:
        tz = timezone(tz)
        for day in tz._utc_transition_times:
            if day.year == year:
                results.append(datetime(day.year, day.month, day.day))
    except Exception:
        pass
    return results


def get_gran_from_freq(freq):
    if 'H' in freq:
        return 1
    elif 'T' in freq:
        n = int(freq.replace('T', ''))
        return 60 // n
    else:
        msg = 'Don\'t know how to convert frequency %s to granularity' % freq
        logger.error(msg)
        raise ValueError(msg)


def get_daily_ts(db, coll_name, date_start, date_end, granularity='1H', date_field='day', value_field='v',
                 out_format='dict', missing_pol='auto', inexcess_pol='slice', add_query=None, tz=None, verbose=0):
    # Check arguments
    allowed_out_formats = ['dict', 'list', 'dataframe']
    allowed_missing_pol = ['raise', 'pad', 'prepend', 'append', 'skip', 'auto']
    allowed_inexcess_pol = ['raise', 'slice']

    if verbose > 1:
        logger.info(' Database instance:\n%s' % str(db))

    if verbose > 0:
        logger.info('Will fetch collection {coll} from {start} to end {end}.' \
                    '\nAdditional query parameters {add_query}'.format(coll=coll_name,
                                                                       start=date_start.strftime('%Y-%m-%d'),
                                                                       end=date_end.strftime('%Y-%m-%d'),
                                                                       add_query=add_query))
    # Get collection
    collection = db[coll_name]

    # Query the database
    and_list = [{date_field: {"$gte": date_start}}, {date_field: {"$lte": date_end}}]
    if add_query is not None:
        if isinstance(add_query, list):
            and_list.extend(add_query)
        elif isinstance(add_query, dict):
            and_list.append(add_query)
        else:
            msg = 'add_query parameter must be either a list of dict or a dict'
            logger.error(msg)
            raise AttributeError(msg)

    query = {"$and": and_list}

    if verbose > 1:
        logger.info(' Total query:\n%s' % str(query))

    dates, vals = [], []
    res = collection.find(query).sort(date_field, 1)

    if verbose > 0:
        logger.info('Found {0} results'.format(res.count()))
    for row in res:

        dt = row[date_field]
        values = row[value_field]

        # Daylight saving time
        res = get_day_saving_time_day(tz=tz, year=dt.year)
        h_to_remove, h_to_clone = None, None
        if tz is not None:
            if dt == res[0]:
                if tz.startswith('Europe'):
                    h_to_remove = 1
                else:
                    h_to_remove = 2
            elif dt == res[1]:
                h_to_clone = 1

        # Build the datetime list
        datetimes = pd.date_range(start=dt,
                                  end=dt + timedelta(hours=23) + timedelta(minutes=59) + timedelta(seconds=59),
                                  freq=granularity)
        if not len(values) > 0:
            logger.info('For day %s values is an empty list. It will be filled with NaNs' % dt)
            values = [np.nan] * len(datetimes)
        if h_to_clone is not None:
            datetimes_l = datetimes.tolist()
            dts_to_clone = [d for d in datetimes_l if d.hour == h_to_clone]
            g = get_gran_from_freq(granularity)
            datetimes_l = datetimes_l[:g + g * h_to_clone] + dts_to_clone + datetimes_l[g + g * h_to_clone:]
            datetimes = pd.DatetimeIndex(datetimes_l)

        if h_to_remove is not None:
            datetimes = [d for d in datetimes if d.hour != h_to_remove]

        if len(datetimes) > len(values):
            if missing_pol == 'raise':
                msg = 'For date {date} found {nf} values instead {ne}'.format(date=dt.strftime('%Y-%m-%d'),
                                                                              nf=len(values),
                                                                              ne=len(datetimes))
                logger.error(msg)
                raise ValueError(msg)
            elif missing_pol == 'pad':
                if h_to_clone is not None:
                    datetimes_l = datetimes.tolist()
                    dts_to_clone = [d for d in datetimes_l if d.hour == h_to_clone]
                    imin, imax = datetimes_l.index(dts_to_clone[0]), datetimes_l.index(dts_to_clone[-1]) + 1
                    values = values[:imax] + values[imin: imax] + values[imax:]
                else:
                    values.extend(values[-1:] * (len(datetimes) - len(values)))
            elif missing_pol == 'prepend':
                datetimes = datetimes[:len(values)]
            elif missing_pol == 'append':
                datetimes = datetimes[-len(values):]
            elif missing_pol == 'skip':
                continue
            elif missing_pol == 'auto':
                if dt == date_start:
                    datetimes = datetimes[-len(values):]
                elif dt == date_end:
                    datetimes = datetimes[:len(values)]
                else:
                    msg = 'For date {date} found {nf} values instead {ne}'.format(date=dt.strftime('%Y-%m-%d'),
                                                                                  nf=len(values),
                                                                                  ne=len(datetimes))
                    logger.error(msg)
                    raise ValueError(msg)
            else:
                msg = 'Unknown policy for missing values: {po}. ' \
                      'Valid ones are:\n {l}'.format(po=missing_pol, l=str(allowed_missing_pol))
                logger.error(msg)
                raise NotImplementedError(msg)

        if len(datetimes) < len(values):
            if inexcess_pol == 'raise':
                msg = 'For date {date} found {nf} values instead {ne}'.format(date=dt.strftime('%Y-%m-%d'),
                                                                              nf=len(values),
                                                                              ne=len(datetimes))
                logger.error(msg)
                raise ValueError(msg)
            elif inexcess_pol == 'slice':
                values = values[:len(datetimes)]
            else:
                msg = 'Unknown policy for values in excess: {po}. ' \
                      'Valid ones are:\n {l}'.format(po=inexcess_pol, l=str(allowed_inexcess_pol))
                logger.error(msg)
                raise NotImplementedError(msg)

        dates.extend(datetimes)
        vals.extend(values)

    if out_format == 'dict':
        return dict(zip([d.to_pydatetime() for d in dates], vals))
    elif out_format == 'list':
        return dates, vals
    elif out_format == 'dataframe':
        df = pd.DataFrame(index=dates, columns=[value_field])
        df[value_field] = vals
        df.sort_index(inplace=True)
        return df
    else:
        msg = 'Format {fo} is not supported. ' \
              'Valid ones are:\n {l}'.format(fo=out_format, l=allowed_out_formats)
        logger.error(msg)
        raise NotImplementedError(msg)


def get_daily_ts_multi(db, coll_name, date_start, date_end, granularity='1H', date_field='day', value_field=None,
                       out_format='dict', missing_pol='auto', inexcess_pol='slice', add_query=None, tz=None, verbose=0):
    # Check arguments
    allowed_out_formats = ['dict', 'list', 'dataframe']
    allowed_missing_pol = ['raise', 'pad', 'prepend', 'append', 'skip', 'auto']
    allowed_inexcess_pol = ['raise', 'slice']

    if verbose > 1:
        logger.info(' Database instance:\n%s' % str(db))

    if verbose > 0:
        logger.info('Will fetch collection {coll} from {start} to end {end}.' \
                    '\nAdditional query parameters {add_query}'.format(coll=coll_name,
                                                                       start=date_start.strftime('%Y-%m-%d'),
                                                                       end=date_end.strftime('%Y-%m-%d'),
                                                                       add_query=add_query))

    # Value field check
    if value_field is None:
        value_field = ['v']
    elif isinstance(value_field, str):
        value_field = [value_field]

    # Get collection
    collection = db[coll_name]

    # Query the database
    and_list = [{date_field: {"$gte": date_start}}, {date_field: {"$lte": date_end}}]
    if add_query is not None:
        if isinstance(add_query, list):
            and_list.extend(add_query)
        elif isinstance(add_query, dict):
            and_list.append(add_query)
        else:
            msg = 'add_query parameter must be either a list of dict or a dict'
            logger.error(msg)
            raise AttributeError(msg)

    # Define the queri field
    query = {"$and": and_list}

    if verbose > 1:
        logger.info(' Total query:\n%s' % str(query))

    dates, vals = [], {}
    res = collection.find(query).sort(date_field, 1)

    if verbose > 0:
        logger.info('Found {0} results'.format(res.count()))

    # Loop over days
    for row in res:
        dt = row[date_field]

        # Daylight saving time
        res = get_day_saving_time_day(tz=tz, year=dt.year)
        h_to_remove, h_to_clone = None, None
        if tz is not None:
            if dt == res[0]:
                if tz.startswith('Europe'):
                    h_to_remove = 1
                else:
                    h_to_remove = 2
            elif dt == res[1]:
                h_to_clone = 1

        # Build the datetime list
        datetimes = pd.date_range(start=dt,
                                  end=dt + timedelta(hours=23) + timedelta(minutes=59) + timedelta(seconds=59),
                                  freq=granularity)

        if h_to_clone is not None:
            datetimes_l = datetimes.tolist()
            dts_to_clone = [d for d in datetimes_l if d.hour == h_to_clone]
            g = get_gran_from_freq(granularity)
            datetimes_l = datetimes_l[:g + g * h_to_clone] + dts_to_clone + datetimes_l[g + g * h_to_clone:]
            datetimes = pd.DatetimeIndex(datetimes_l)

        if h_to_remove is not None:
            datetimes = [d for d in datetimes if d.hour != h_to_remove]

        # Loop over value fields
        for value_f in value_field:
            values = row[value_f]

            if not len(values) > 0:
                logger.info('For day %s values in key %s is an empty list. It will be filled with NaNs' % (dt, value_f))
                values = [np.nan] * len(datetimes)

            if len(datetimes) > len(values):
                if missing_pol == 'raise':
                    msg = 'For date {date} found {nf} values instead {ne}'.format(date=dt.strftime('%Y-%m-%d'),
                                                                                  nf=len(values),
                                                                                  ne=len(datetimes))
                    logger.error(msg)
                    raise ValueError(msg)
                elif missing_pol == 'pad':
                    if h_to_clone is not None:
                        datetimes_l = datetimes.tolist()
                        dts_to_clone = [d for d in datetimes_l if d.hour == h_to_clone]
                        imin, imax = datetimes_l.index(dts_to_clone[0]), datetimes_l.index(dts_to_clone[-1]) + 1
                        values = values[:imax] + values[imin: imax] + values[imax:]
                    else:
                        values.extend(values[-1:] * (len(datetimes) - len(values)))
                elif missing_pol == 'prepend':
                    datetimes = datetimes[:len(values)]
                elif missing_pol == 'append':
                    datetimes = datetimes[-len(values):]
                elif missing_pol == 'skip':
                    continue
                elif missing_pol == 'auto':
                    if dt == date_start:
                        datetimes = datetimes[-len(values):]
                    elif dt == date_end:
                        datetimes = datetimes[:len(values)]
                    else:
                        msg = 'For date {date} found {nf} values instead {ne}'.format(date=dt.strftime('%Y-%m-%d'),
                                                                                      nf=len(values),
                                                                                      ne=len(datetimes))
                        logger.error(msg)
                        raise ValueError(msg)
                else:
                    msg = 'Unknown policy for missing values: {po}. ' \
                          'Valid ones are:\n {l}'.format(po=inexcess_pol,
                                                         l=str(allowed_missing_pol))
                    logger.error(msg)
                    raise NotImplementedError(msg)

            if len(datetimes) < len(values):
                if inexcess_pol == 'raise':
                    msg = 'For date {date} found {nf} values instead {ne}'.format(date=dt.strftime('%Y-%m-%d'),
                                                                                  nf=len(values),
                                                                                  ne=len(datetimes))
                    logger.error(msg)
                    raise ValueError(msg)
                elif inexcess_pol == 'slice':
                    values = values[:len(datetimes)]
                else:
                    msg = 'Unknown policy for values in excess: {po}. ' \
                          'Valid ones are:\n {l}'.format(po=inexcess_pol,
                                                         l=str(allowed_inexcess_pol))
                    logger.error(msg)
                    raise NotImplementedError(msg)

            vals.setdefault(value_f, [])
            vals[value_f].extend(values)

        dates.extend(datetimes)

    if out_format == 'dict':
        out_dict = {}
        for idate, d in enumerate(dates):
            out_dict[d] = {}
            for vf, values in vals.iteritems():
                out_dict[d][vf] = values[idate]
        return out_dict
    elif out_format == 'list':
        return dates, vals
    elif out_format == 'dataframe':
        df = pd.DataFrame(index=dates, data=vals)
        df.sort_index(inplace=True)
        return df
    else:
        msg = 'Format {fo} is not supported. ' \
              'Valid ones are:\n {l}'.format(fo=out_format, l=allowed_out_formats)
        logger.error(msg)
        raise NotImplementedError(msg)


def write_daily_ts(db, coll_name, df, date_field='day', value_field=None, dfcol=None, add_query=None, add_fields=None):
    if dfcol is not None and not isinstance(df, pd.core.frame.DataFrame) and dfcol not in df.columns:
        msg = 'Column {} is not in the DataFrame'.format(dfcol)
        logger.error(msg)
        raise AttributeError(msg)
    elif dfcol is None and not isinstance(df, pd.core.series.Series):
        msg = 'You have either to specify the column you want to write or pass a DataFrame with one column'
        logger.error(msg)
        raise ValueError(msg)

    # Value field - sanity check
    if value_field is None:
        value_field = ['v']
    elif isinstance(value_field, str) and isinstance(dfcol, str):
        value_field = [value_field]
        dfcol = [dfcol]
    if dfcol is not None and not (isinstance(value_field, list) and isinstance(dfcol, list)):
        msg = 'For multiple keys both value_field and dfcol must be a list'
        logger.error(msg)
        raise ValueError(msg)

    for ts, day_df in df.groupby(pd.Grouper(freq='1D')):
        query = {date_field: ts.to_pydatetime()}

        out_dict = {}
        for ivf, vf in enumerate(value_field):
            if dfcol is not None:
                out_list = day_df[dfcol[ivf]].tolist()
            else:
                out_list = day_df.tolist()
            out_dict[vf] = out_list

        # Add fields
        if add_query is not None:
            for k, v in add_query.iteritems():
                query[k] = v

        if add_fields is not None and isinstance(add_fields, dict):
            out_dict.update(add_fields)

        # Write to DB
        db[coll_name].update(query, {'$set': out_dict}, upsert=True)

    return
