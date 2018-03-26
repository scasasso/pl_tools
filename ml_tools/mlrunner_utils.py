from datetime import datetime, timedelta
from bson.objectid import ObjectId
from dateutil.relativedelta import relativedelta
import logging
import json
import re
import pandas as pd
import numpy as np
from pydoc import locate
import traceback

logger = logging.getLogger(__file__)


def load_models(model_filepath):
    logger.info('Load model: start')
    models = []
    fr = open(model_filepath, 'r')
    content = json.loads(fr.read())
    for model, scaler, teacher, type_ in zip(content['models_filepath'], content['models_scaler'], content['models_teacher'], content['models_class']):
        try:
            class_obj = locate(type_)
            model_to_load = class_obj.load_model(model)
            models.append(model_to_load)
        except Exception as e:
            logger.error('Error while loading model %s:\n%s' % (model, traceback.print_exc()))

    fr.close()
    logger.info('Load model: end')

    return models


def build_data_structure_train(df):
    X = df.iloc[:, :-1].as_matrix()
    y = df.iloc[:, -1].as_matrix()

    return X, y


def build_data_structure_test(df):
    X = df.as_matrix()

    return X


def get_weather_average(db, dt_end, city, country):

    dt_start = datetime(2005, 1, 1)

    # Collections
    coll_weather = db['weather']
    coll_town = db["address"]

    id_city = coll_town.find_one({"country": country, "town": city})["_id"]

    # Compute averages
    q = {"$and": [{'date': {"$gte": dt_start}},
                  {'date': {"$lte": dt_end}},
                  {'point': id_city}
                  ]
         }

    # Get from database
    res = coll_weather.find(q).sort('date', 1)

    dates = []
    collection_rows = []
    for row in res:
        dates.append(row["date"])
        collection_rows.append(row)

    rows = {'temp': [], 'wind': [], 'cloud': [], 'sky': [], 'sunrise': [], 'sunset': []}
    for wrow in collection_rows:
        for var in ['temp', 'wind', 'cloud', 'sky', 'sunrise', 'sunset']:
            rows[var].append(np.mean(wrow[var]))

    rows = {city + '_' + k + '_avg1M': v for k, v in rows.iteritems()}
    df = pd.DataFrame(index=dates, data=rows)

    df['month_of_year'] = df.index.month
    df_agg_month = df.groupby('month_of_year').agg(np.mean)

    return df_agg_month.copy()


def weather_forecast_function(db, weather_config, date_start, date_end, freq='1H', params=None):
    weather_vars = ['temp', 'wind', 'cloud', 'sky', 'sunrise', 'sunset']
    # Start date
    date_start_safe = date_start - timedelta(days=60)

    # Collections
    coll_weather = db['weather']
    coll_weather_frcst = db['weather_forecast']
    coll_town = db["address"]

    # Cities and country
    cities = weather_config["weather_cities"]
    country = weather_config["country"]
    city_country = None
    if "weather_cities_country" in weather_config:
        city_country = weather_config["weather_cities_country"]
        cities = cities + city_country.keys()

    # Loop over cities
    dfs, df_aggs = [], []
    for city in cities:
        real_country = country
        if city_country is not None:
            if city in city_country:
                real_country = city_country[city]

        logger.info('Looking for city: %s in country: %s' % (city, real_country))
        id_city = coll_town.find_one({"country": real_country, "town": city})["_id"]

        q = {"$and": [{'date': {"$gte": date_start_safe}},
                      {'date': {"$lte": date_end}},
                      {'point': id_city}
                      ]
             }

        res_frcst = sorted([entry for entry in coll_weather_frcst.find(q)], key=lambda x: x["date"])

        dates = []
        collection_rows = []

        for row in res_frcst:
            if not row["date"] + timedelta(days=1) in dates:
                dates.append(row["date"] + timedelta(days=1))
                row = {
                    "date": row["date"] + timedelta(days=1),
                    "temp": row["h1"]["temp"],
                    "wind": row["h1"]["wind"],
                    "cloud": row["h1"]["cloud"],
                    "sky": row["h1"]["sky"],
                    "sunrise": row["j_1"]["sunrise"],
                    "sunset": row["j_1"]["sunset"]
                }
                collection_rows.append(row)
        for row in res_frcst:
            if not row["date"] + timedelta(days=2) in dates:
                if row.get("h2", None):
                    dates.append(row["date"] + timedelta(days=2))
                    row = {
                        "date": row["date"] + timedelta(days=2),
                        "temp": row["h2"]["temp"],
                        "wind": row["h2"]["wind"],
                        "cloud": row["h2"]["cloud"],
                        "sky": row["h2"]["sky"],
                        "sunrise": row["j_1"]["sunrise"],
                        "sunset": row["j_1"]["sunset"]
                    }
                    collection_rows.append(row)

        for row in sorted([entry for entry in coll_weather.find(q)], key=lambda x: x["date"]):
            if row["date"] not in dates and date_start_safe <= row['date'] <= date_end:
                dates.append(row["date"])
                collection_rows.append(row)

        dates = sorted(dates)
        collection_rows = sorted(collection_rows, key=lambda x: x["date"])
        datetimes = pd.date_range(dates[0], dates[-1] + timedelta(hours=23) + timedelta(minutes=59), freq='1H')

        if len(dates) != len(collection_rows):
            raise ValueError('Something is wrong with the weather signals: {0} {1}'.format(len(dates), len(collection_rows)))

        # Initialization
        rows = {v: [] for v in weather_vars}

        for wrow in collection_rows:
            for var in ['temp', 'wind', 'cloud', 'sky']:
                l = list(np.lib.pad(wrow[var], (0, max(24 - len(wrow[var]), 0)), 'constant', constant_values=wrow[var][-1]))[:24]
                rows[var].extend(l)
            for var in ['sunrise', 'sunset']:
                rows[var].extend([wrow[var]] * 24)

        rows = {city + '_' + k + '_frcst': v for k, v in rows.iteritems()}
        df_city = pd.DataFrame(index=datetimes, data=rows)

        # Get weather monthly averages
        df_agg = get_weather_average(db, date_end, city, real_country)
        df_aggs.append(df_agg.copy())
        df_city['month_of_year'] = df_city.index.month
        for m, row in df_agg.iterrows():
            for c in weather_vars:
                df_city.loc[df_city['month_of_year'] == m, city + '_' + c + '_frcst_diff_month_avg'] = \
                    df_city.loc[df_city['month_of_year'] == m, city + '_' + c + '_frcst'] - row[city + '_' + c + '_avg1M']
                df_city.loc[df_city['month_of_year'] == m, city + '_' + c + '_frcst_on_month_avg'] = \
                    df_city.loc[df_city['month_of_year'] == m, city + '_' + c + '_frcst'] / row[city + '_' + c + '_avg1M']

        df_city = df_city.drop('month_of_year', axis=1)

        dfs.append(df_city.copy())

    # Concatenate
    df = pd.concat(dfs, axis=1)

    # Add aggregates
    if len(cities) > 1:
        df['agg_temp_frcst'] = df[[c for c in df.columns if c.endswith('_temp_frcst')]].mean(axis=1)
        df['agg_wind_frcst'] = df[[c for c in df.columns if c.endswith('_wind_frcst')]].mean(axis=1)
        df['agg_cloud_frcst'] = df[[c for c in df.columns if c.endswith('_cloud_frcst')]].mean(axis=1)

        df['month_of_year'] = df.index.month
        df_agg_agg = pd.concat(df_aggs, axis=1)
        df_agg_agg['agg_temp_avg1M'] = df_agg_agg[[c for c in df_agg_agg.columns if '_temp_avg1M' in c]].mean(axis=1)
        df_agg_agg['agg_wind_avg1M'] = df_agg_agg[[c for c in df_agg_agg.columns if '_wind_avg1M' in c]].mean(axis=1)
        df_agg_agg['agg_cloud_avg1M'] = df_agg_agg[[c for c in df_agg_agg.columns if '_cloud_avg1M' in c]].mean(axis=1)
        for m, row in df_agg_agg.iterrows():
            for c in ['temp', 'wind', 'cloud']:
                df.loc[df['month_of_year'] == m, 'agg_' + c + '_frcst_diff_month_avg'] = \
                    df.loc[df['month_of_year'] == m, 'agg_' + c + '_frcst'] - row['agg_' + c + '_avg1M']
                df.loc[df['month_of_year'] == m, 'agg_' + c + '_frcst_on_month_avg'] = \
                    df.loc[df['month_of_year'] == m, 'agg_' + c + '_frcst'] / row['agg_' + c + '_avg1M']
        df = df.drop('month_of_year', axis=1)

    df_out = df.resample(freq, convention='start').ffill()

    return df_out


def weather_actual_function(db, weather_config, date_start, date_end, freq='1H', params=None):
    weather_vars = ['temp', 'wind', 'cloud', 'sky', 'sunrise', 'sunset']

    # Start date
    date_start_safe = date_start - timedelta(days=60)

    # Collections
    coll_weather = db['weather']
    coll_town = db["address"]

    # Cities and country
    cities = weather_config["weather_cities"]
    country = weather_config["country"]
    city_country = None
    if "weather_cities_country" in weather_config:
        city_country = weather_config["weather_cities_country"]
        cities = cities + city_country.keys()

    # Loop over cities
    dfs, df_aggs = [], []
    for city in cities:
        real_country = country
        if city_country is not None:
            if city in city_country:
                real_country = city_country[city]

        logger.info('Looking for city: %s in country: %s' % (city, real_country))
        id_city = coll_town.find_one({"country": real_country, "town": city})["_id"]

        q = {"$and": [{'date': {"$gte": date_start_safe}},
                      {'date': {"$lte": date_end}},
                      {'point': id_city}
                      ]
             }

        res = sorted([entry for entry in coll_weather.find(q)], key=lambda x: x["date"])

        dates = []
        collection_rows = []

        for row in res:
            if row["date"] not in dates and date_start_safe <= row['date'] <= date_end:
                dates.append(row["date"])
                collection_rows.append(row)

        dates = sorted(dates)
        collection_rows = sorted(collection_rows, key=lambda x: x["date"])
        datetimes = pd.date_range(dates[0], dates[-1] + timedelta(hours=23) + timedelta(minutes=59), freq='1H')

        if len(dates) != len(collection_rows):
            raise ValueError('Something is wrong with the weather signals: {0} {1}'.format(len(dates), len(collection_rows)))

        rows = {'temp': [], 'wind': [], 'cloud': [], 'sky': [], 'sunrise': [], 'sunset': []}
        for wrow in collection_rows:
            for var in ['temp', 'wind', 'cloud', 'sky']:
                l = list(np.lib.pad(wrow[var], (0, max(24 - len(wrow[var]), 0)), 'constant', constant_values=wrow[var][-1]))[:24]
                rows[var].extend(l)
            for var in ['sunrise', 'sunset']:
                rows[var].extend([wrow[var]] * 24)

        rows = {city + '_' + k: v for k, v in rows.iteritems()}
        df_city = pd.DataFrame(index=datetimes, data=rows)

        # Get weather monthly averages
        df_agg = get_weather_average(db, date_end, city, real_country)
        df_aggs.append(df_agg.copy())
        df_city['month_of_year'] = df_city.index.month
        for m, row in df_agg.iterrows():
            for c in weather_vars:
                df_city.loc[df_city['month_of_year'] == m, city + '_' + c + '_diff_month_avg'] = \
                    df_city.loc[df_city['month_of_year'] == m, city + '_' + c] - row[city + '_' + c + '_avg1M']
                df_city.loc[df_city['month_of_year'] == m, city + '_' + c + '_on_month_avg'] = \
                    df_city.loc[df_city['month_of_year'] == m, city + '_' + c] / row[city + '_' + c + '_avg1M']

        df_city = df_city.drop('month_of_year', axis=1)

        dfs.append(df_city.copy())

    # Concatenate
    df = pd.concat(dfs, axis=1)

    # Add aggregates
    if len(cities) > 1:
        df['agg_temp'] = df[[c for c in df.columns if '_temp' in c]].mean(axis=1)
        df['agg_wind'] = df[[c for c in df.columns if '_wind' in c]].mean(axis=1)
        df['agg_cloud'] = df[[c for c in df.columns if '_cloud' in c]].mean(axis=1)

        df['month_of_year'] = df.index.month
        df_agg_agg = pd.concat(df_aggs, axis=1)
        df_agg_agg['agg_temp_avg1M'] = df_agg_agg[[c for c in df_agg_agg.columns if '_temp_avg1M' in c]].mean(axis=1)
        df_agg_agg['agg_wind_avg1M'] = df_agg_agg[[c for c in df_agg_agg.columns if '_wind_avg1M' in c]].mean(axis=1)
        df_agg_agg['agg_cloud_avg1M'] = df_agg_agg[[c for c in df_agg_agg.columns if '_cloud_avg1M' in c]].mean(axis=1)
        for m, row in df_agg_agg.iterrows():
            for c in ['temp', 'wind', 'cloud']:
                df.loc[df['month_of_year'] == m, 'agg_' + c + '_diff_month_avg'] = \
                    df.loc[df['month_of_year'] == m, 'agg_' + c] - row['agg_' + c + '_avg1M']
                df.loc[df['month_of_year'] == m, 'agg_' + c + '_on_month_avg'] = \
                    df.loc[df['month_of_year'] == m, 'agg_' + c] / row['agg_' + c + '_avg1M']
        df = df.drop('month_of_year', axis=1)

    df_out = df.resample(freq, convention='start').ffill()

    return df_out


def is_daysoff(db, pl_config, date_start, date_end, freq='1H', params=None):

    logger.info('Adding daysoff')

    # Date range
    dt_start_incl = datetime(2005, 1, 1)
    dt_end_incl = datetime(datetime.now().year, 12, 31)

    # Collections
    coll_daysoff = db['daysoff']

    q = {"$and": [{'day': {"$gte": dt_start_incl}},
                  {'day': {"$lte": dt_end_incl}},
                  {'country': pl_config['country']},
                  {'cat': 0}  # only national festivities
                  ]
         }

    # Get from database
    daysoff = [row for row in coll_daysoff.find(q)]

    # Prepare the output DataFrame
    datetimes = pd.date_range(dt_start_incl, dt_end_incl + timedelta(hours=23) + timedelta(minutes=59), freq=freq)
    df = pd.DataFrame(index=datetimes)

    # Loop over the daysoff
    for do in daysoff:
        do_date = do['day']

        # Not always there's a desc field...
        try:
            df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_desc'] = 'national' + '_'.join(do['desc'].encode('utf-8').decode('unicode_escape').encode('ascii', 'ignore').split())
        except KeyError:
            df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_desc'] = 'unknown'
        # try:
        #     df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_cat'] = do['cat']
        # except KeyError:
        #     df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_cat'] = -1

        # Add bridge days
        if do_date.weekday() == 1 and params.get('add_bridge', False) is True:
            df.loc[do_date - timedelta(days=1): do_date - timedelta(minutes=1), 'is_daysoff_bridge'] = 1
        if do_date.weekday() == 3 and params.get('add_bridge', False) is True:
            df.loc[do_date + timedelta(days=1): do_date + timedelta(days=2) - timedelta(minutes=1), 'is_daysoff_bridge'] = 1

    # Remaining days are normal working days
    df['daysoff_desc'] = df['daysoff_desc'].fillna('work')
    # df['daysoff_cat'] = df['daysoff_cat'].fillna(-2)

    # Bridges
    if params.get('add_bridge', False) is True and 'is_daysoff_bridge' in df.columns:
        df['is_daysoff_bridge'] = df['is_daysoff_bridge'].fillna(0)

    # One-hot encode
    df = pd.concat([df, pd.get_dummies(df['daysoff_desc'], prefix='is_daysoff')], axis=1)
    df = df.drop('daysoff_desc', axis=1)
    # df = pd.concat([df, pd.get_dummies(df['daysoff_cat'], prefix='is_daysoff_cat')], axis=1)
    # df = df.drop('daysoff_cat', axis=1)

    # Add days to dayoff
    ts_list = df.loc[df['is_daysoff_work'] == 0, :].index
    datetimes = sorted([datetime(dd.year, dd.month, dd.day) for dd in list(set([d.date() for d in ts_list]))])
    for i in range(1, 3):
        df['daysoff_%s_days_to_daysoff' % i] = 0
        df['daysoff_%s_days_after_daysoff' % i] = 0
        for dt in datetimes:
            dmi = dt - timedelta(days=i)
            df.loc[dmi: dmi + timedelta(hours=23) + timedelta(minutes=59), 'daysoff_%s_days_to_daysoff' % i] = 1
            dpi = dt + timedelta(days=i)
            df.loc[dpi: dpi + timedelta(hours=23) + timedelta(minutes=59), 'daysoff_%s_days_after_daysoff' % i] = 1

    return df[date_start: date_end].copy()


def is_daysoff_regional(db, pl_config, date_start, date_end, freq='1H', params=None):

    logger.info('Adding regional daysoff')

    # These are basically regexp we match to the desc field
    patterns = params.get('patterns', [])

    # Utils
    def count_regexp_occ(regexp, text):
        return len(re.findall(regexp, text.lower()))

    # Date range
    dt_start_incl = datetime(2005, 1, 1)
    dt_end_incl = datetime(datetime.now().year, 12, 31)

    # Collections
    coll_daysoff = db['daysoff']

    q = {"$and": [{'day': {"$gte": dt_start_incl}},
                  {'day': {"$lte": dt_end_incl}},
                  {'country': pl_config['country']},
                  {'cat': 1}  # only regional festivities
                  ]
         }

    # Get from database
    daysoff = [row for row in coll_daysoff.find(q)]

    # Prepare the output DataFrame
    datetimes = pd.date_range(dt_start_incl, dt_end_incl + timedelta(hours=23) + timedelta(minutes=59), freq=freq)
    df = pd.DataFrame(index=datetimes)

    # Default initialization
    for i, pt in enumerate(patterns):
        df['is_daysoff_reg' + str(i)] = 0

    # Loop over the daysoff
    for do in daysoff:
        do_date = do['day']
        info = do['info'].encode('utf-8').decode('unicode_escape').encode('ascii', 'ignore').lower()

        # First case: few regions are excluded
        if 'except' in info or 'excluded' in info:
            if 'except' in info:
                non_incl = info.split('except')[1].strip().split('.')[0]
            else:
                non_incl = info.split('excluded')[1].strip().split('.')[0]

            # Check for relevant regions
            relevant_for = []
            for i, pt in enumerate(patterns):
                nocc = count_regexp_occ(pt, non_incl)
                relevant_for.append(nocc == 0)
                if nocc == 0:
                    df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'is_daysoff_reg' + str(i)] = 1
            if any(relevant_for):
                try:
                    df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_desc'] = 'regional' + '_'.join(
                        do['desc'].encode('utf-8').decode('unicode_escape').encode('ascii', 'ignore').split())
                except KeyError:
                    df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_desc'] = 'unknown'

                # Add bridge days
                if do_date.weekday() == 1 and params.get('add_bridge', False) is True:
                    df.loc[do_date - timedelta(days=1): do_date - timedelta(minutes=1), 'is_daysoff_bridge_reg'] = 1
                if do_date.weekday() == 3 and params.get('add_bridge', False) is True:
                    df.loc[do_date + timedelta(days=1): do_date + timedelta(days=2) - timedelta(minutes=1), 'is_daysoff_bridge_reg'] = 1

        # Second case: just all the regions
        elif ' all ' in info:  # with spaces otherwise we get St Gallen or sth...
            try:
                df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_desc'] = 'regional' + '_'.join(
                    do['desc'].encode('utf-8').decode('unicode_escape').encode('ascii', 'ignore').split())
            except KeyError:
                df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_desc'] = 'unknown'

            # Add bridge days
            if do_date.weekday() == 1 and params.get('add_bridge', False) is True:
                df.loc[do_date - timedelta(days=1): do_date - timedelta(minutes=1), 'is_daysoff_bridge_reg'] = 1
            if do_date.weekday() == 3 and params.get('add_bridge', False) is True:
                df.loc[do_date + timedelta(days=1): do_date + timedelta(days=2) - timedelta(minutes=1), 'is_daysoff_bridge_reg'] = 1

            # Check for relevant regions
            for i, pt in enumerate(patterns):
                df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'is_daysoff_reg' + str(i)] = 1
        # Third case: only some regions. We parse for our patterns
        else:
            # Check for relevant regions
            relevant_for = []
            for i, pt in enumerate(patterns):
                nocc = count_regexp_occ(pt, info)
                relevant_for.append(nocc > 0)
                if nocc > 0:
                    df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'is_daysoff_reg' + str(i)] = 1
            if any(relevant_for):
                try:
                    df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_desc'] = 'regional' + '_'.join(
                        do['desc'].encode('utf-8').decode('unicode_escape').encode('ascii', 'ignore').split())
                except KeyError:
                    df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_desc'] = 'unknown'

                # Add bridge days
                if do_date.weekday() == 1 and params.get('add_bridge', False) is True:
                    df.loc[do_date - timedelta(days=1): do_date - timedelta(minutes=1), 'is_daysoff_bridge_reg'] = 1
                if do_date.weekday() == 3 and params.get('add_bridge', False) is True:
                    df.loc[do_date + timedelta(days=1): do_date + timedelta(days=2) - timedelta(minutes=1), 'is_daysoff_bridge_reg'] = 1

        # try:
        #     df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_cat'] = do['cat']
        # except KeyError:
        #     df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_cat'] = -1

    # Remaining days are normal working days
    df['daysoff_desc'] = df['daysoff_desc'].fillna('work_reg_all')
    # df['daysoff_cat'] = df['daysoff_cat'].fillna(-2)

    # One-hot encode
    df = pd.concat([df, pd.get_dummies(df['daysoff_desc'], prefix='is_daysoff')], axis=1)
    df = df.drop('daysoff_desc', axis=1)

    # Add days to dayoff
    ts_list = df.loc[df['is_daysoff_work_reg_all'] == 0, :].index
    datetimes = sorted([datetime(dd.year, dd.month, dd.day) for dd in list(set([d.date() for d in ts_list]))])
    for i in range(1, 3):
        df['daysoff_%s_days_to_daysoff_reg' % i] = 0
        df['daysoff_%s_days_after_daysoff_reg' % i] = 0
        for dt in datetimes:
            dmi = dt - timedelta(days=i)
            df.loc[dmi: dmi + timedelta(hours=23) + timedelta(minutes=59), 'daysoff_%s_days_to_daysoff_reg' % i] = 1
            dpi = dt + timedelta(days=i)
            df.loc[dpi: dpi + timedelta(hours=23) + timedelta(minutes=59), 'daysoff_%s_days_after_daysoff_reg' % i] = 1

    df = df.drop('is_daysoff_work_reg_all', axis=1)
    # df = pd.concat([df, pd.get_dummies(df['daysoff_cat'], prefix='is_daysoff_cat')], axis=1)
    # df = df.drop('daysoff_cat', axis=1)

    return df[date_start: date_end].copy()
