from pymongo.mongo_client import MongoClient
from datetime import timedelta, datetime
import pandas as pd


def get_daily_ts(db, coll_name, date_start, date_end, granularity='1H', date_field='day', value_field='v',
                 out_format='dict', missing_pol='auto', inexcess_pol='slice', add_query=None, verbose=0):
    # Check arguments
    allowed_out_formats = ['dict', 'list', 'dataframe']
    allowed_missing_pol = ['raise', 'prepend', 'append', 'skip', 'auto']
    allowed_inexcess_pol = ['raise', 'slice']

    if verbose > 1:
        print ' Database instance:\n%s' % str(db)

    if verbose > 0:
        print 'Will fetch collection {coll} from {start} to end {end}.' \
              '\nAdditional query parameters {add_query}'.format(coll=coll_name,
                                                                 start=date_start.strftime('%Y-%m-%d'),
                                                                 end=date_end.strftime('%Y-%m-%d'),
                                                                 add_query=add_query)
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
            raise AttributeError('add_query parameter must be either a list of dict or a dict')

    query = {"$and": and_list}

    if verbose > 1:
        print ' Total query:\n%s' % str(query)

    dates, vals = [], []
    res = collection.find(query).sort(date_field, 1)

    if verbose > 0:
        print 'Found {0} results'.format(res.count())
    for row in res:

        dt = row[date_field]
        values = row[value_field]
        
        # Build the datetime list
        datetimes = pd.date_range(start=dt,
                                  end=dt + timedelta(hours=23) + timedelta(minutes=59) + timedelta(seconds=59),
                                  freq=granularity)

        if len(datetimes) > len(values):
            if missing_pol == 'raise':
                raise ValueError('For date {date} found {nf} values instead {ne}'.format(date=dt.strftime('%Y-%m-%d'),
                                                                                         nf=len(values),
                                                                                         ne=len(datetimes)))
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
                    raise ValueError(
                        'For date {date} found {nf} values instead {ne}'.format(date=dt.strftime('%Y-%m-%d'),
                                                                                nf=len(values),
                                                                                ne=len(datetimes)))
            else:
                raise NotImplementedError('Unknown policy for missing values: {po}. '
                                          'Valid ones are:\n {l}'.format(po=inexcess_pol, l=str(allowed_missing_pol)))

        if len(datetimes) < len(values):
            if inexcess_pol == 'raise':
                raise ValueError('For date {date} found {nf} values instead {ne}'.format(date=dt.strftime('%Y-%m-%d'),
                                                                                         nf=len(values),
                                                                                         ne=len(datetimes)))
            elif inexcess_pol == 'slice':
                values = values[:len(datetimes)]
            else:
                raise NotImplementedError('Unknown policy for values in excess: {po}. '
                                          'Valid ones are:\n {l}'.format(po=inexcess_pol, l=str(allowed_inexcess_pol)))

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
        raise NotImplementedError('Format {fo} is not supported. '
                                  'Valid ones are:\n {l}'.format(fo=out_format, l=allowed_out_formats))


def write_daily_ts(db, coll_name, df, date_field='day', value_field='v', dfcol=None, add_query=None):

    if dfcol is not None and not isinstance(df, pd.core.frame.DataFrame) and dfcol not in df.columns:
        raise AttributeError('Column {} is not in the DataFrame'.format(dfcol))
    elif dfcol is None and not isinstance(df, pd.core.series.Series):
        raise ValueError('You have either to specify the column you want to write or pass a DataFrame with one column')

    for ts, day_df in df.groupby(pd.Grouper(freq='1D')):
        
        query = {date_field: ts.to_pydatetime()}
        
        if dfcol is not None:
            out_list = day_df[dfcol].tolist()
        else:
            out_list = day_df.tolist()
        out_dict = {value_field: out_list}

        # Add fields
        if add_query is not None:
            for k, v in add_query.iteritems():
                query[k] = v

        # Write to DB
        db[coll_name].update(query, {'$set': out_dict}, upsert=True)
        
    return
