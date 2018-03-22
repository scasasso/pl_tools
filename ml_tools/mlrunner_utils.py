from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
import json
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


def weather_function(db, weather_config, date_start, date_end, freq='1H'):

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
    dfs = []
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

        rows = {'temp': [], 'wind': [], 'cloud': [], 'sky': [], 'sunrise': [], 'sunset': []}
        for wrow in collection_rows:
            for var in ['temp', 'wind', 'cloud', 'sky']:
                l = list(np.lib.pad(wrow[var], (0, max(24 - len(wrow[var]), 0)), 'constant', constant_values=wrow[var][-1]))[:24]
                rows[var].extend(l)
            for var in ['sunrise', 'sunset']:
                rows[var].extend([wrow[var]] * 24)

        rows = {city + '_' + k: v for k, v in rows.iteritems()}
        df_city = pd.DataFrame(index=datetimes, data=rows)

        dfs.append(df_city.copy())

    # Concatenate
    df = pd.concat(dfs, axis=1)

    # Add aggregates
    if len(cities) > 1:
        df['agg_temp'] = df[[c for c in df.columns if '_temp' in c]].mean(axis=1)
        df['agg_wind'] = df[[c for c in df.columns if '_wind' in c]].mean(axis=1)
        df['agg_cloud'] = df[[c for c in df.columns if '_cloud' in c]].mean(axis=1)

    return df.resample(freq, convention='start').ffill()


def is_daysoff(db, pl_config, date_start, date_end, freq='1H'):

    logger.info('Adding daysoff')

    # Date range
    dt_start_incl = datetime(2005, 1, 1)
    dt_end_incl = datetime(datetime.now().year, 12, 31)

    # Collections
    coll_daysoff = db['daysoff']

    q = {"$and": [{'day': {"$gte": dt_start_incl}},
                  {'day': {"$lte": dt_end_incl}},
                  {'country': pl_config['country']}
                  ]
         }

    daysoff = [row for row in coll_daysoff.find(q)]

    datetimes = pd.date_range(dt_start_incl, dt_end_incl + timedelta(hours=23) + timedelta(minutes=59), freq=freq)
    df = pd.DataFrame(index=datetimes)

    for do in daysoff:
        do_date = do['day']
        df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'is_daysoff'] = 1
        try:
            df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_desc'] = '_'.join(do['desc'].encode('utf-8').decode('unicode_escape').encode('ascii', 'ignore').split())
        except KeyError:
            df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_desc'] = 'unknown'
        try:
            df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_cat'] = do['cat']
        except KeyError:
            df.loc[do_date: do_date + timedelta(days=1) - timedelta(minutes=1), 'daysoff_cat'] = -1
    df['is_daysoff'] = df['is_daysoff'].fillna(0)
    df['daysoff_desc'] = df['daysoff_desc'].fillna('work')
    df['daysoff_cat'] = df['daysoff_cat'].fillna(-2)

    df = pd.concat([df, pd.get_dummies(df['daysoff_desc'], prefix='is_daysoff')], axis=1)
    df = df.drop('daysoff_desc', axis=1)
    df = pd.concat([df, pd.get_dummies(df['daysoff_cat'], prefix='is_daysoff_cat')], axis=1)
    df = df.drop('daysoff_cat', axis=1)

    return df[date_start: date_end].copy()
