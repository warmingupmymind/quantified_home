# standard
import datetime as dt
from dateutil.parser import parse
import requests

# third party
import bs4
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


def download_winter_data(weather_station_id, year=2019):
    datas = []
    granularity = "daily"
    year = year
    months = range(11, 13)
    days = range(1, 32)
    for month in months:
        for day in days:
            try:
                url = f"https://www.wunderground.com/dashboard/pws/{weather_station_id}/table/{year}-{month}-{day}/{year}-{month}-{day}/{granularity}"
                print(url)
                body = get_and_parse_url(url)
                df = build_dataset(body, granularity)
                df["date"] = dt.date(year, month, day)
                print(df.shape)
                datas.append(df)
            except Exception as e:
                print(e)

    year += 1
    months = range(1, 3)
    days = range(1, 32)
    for month in months:
        for day in days:
            try:
                url = f"https://www.wunderground.com/dashboard/pws/{weather_station_id}/table/{year}-{month}-{day}/{year}-{month}-{day}/{granularity}"
                print(url)
                body = get_and_parse_url(url)
                df = build_dataset(body, granularity)
                df["date"] = dt.date(year, month, day)
                print(df.shape)
                datas.append(df)
            except Exception as e:
                print(e)

    df = pd.concat(datas)
    datetimeys = []
    for date, time in zip(df.date.values, df.time.values):
        datetimey = dt.datetime.combine(date, parse(time).time())
        datetimeys.append(datetimey)
    df["datetime"] = datetimeys
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def get_and_parse_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    html = list(soup.children)[1]
    body = list(html.children)[2] # list(body.children)[1]

    return body


def build_dataset(body, granularity):
    class_ = "ng-star-inserted"
    inserts = [
        p.get_text().split("\xa0")[0]
        for p in body.find_all(class_=class_)
    ]

    if granularity == "monthly":
        for ix, insert in enumerate(inserts):
            if insert == last_day:
                break
        inserts = inserts[ix + 1:-3]

        # 1 date col, 15 feature cols, 15 unit cols
        df = pd.DataFrame(
            np.array(inserts).reshape((int(len(inserts)/31), 31)),
            columns = [
                "date",
                "temperature_f_high",
                "temperature_f_high_unit",
                "temperature_f_avg",
                "temperature_f_avg_unit",
                "temperature_f_low",
                "temperature_f_low_unit",
                "dew_point_f_high",
                "dew_point_f_high_unit",
                "dew_point_f_avg",
                "dew_point_f_avg_unit",
                "dew_point_f_low",
                "dew_point_f_low_unit",
                "humidity_percent_high",
                "humidity_percent_high_unit",
                "humidity_percent_avg",
                "humidity_percent_avg_unit",
                "humidity_percent_low",
                "humidity_percent_low_unit",
                "windspeed_mph_high",
                "windspeed_mph_high_unit",
                "windspeed_mph_avg",
                "windspeed_mph_avg_unit",
                "windspeed_mph_low",
                "windspeed_mph_low_unit",
                "pressure_in_high",
                "pressure_in_high_unit",
                "pressure_in_low",
                "pressure_in_low_unit",
                "precip_accum_in_sum",
                "precip_accum_in_sum_unit",
            ]
        )
        df["date"] = [date.split(year)[0]+year for date in df["date"]]
        df = df[[col for col in df.columns if "unit" not in col]]
        df = df.set_index(df["date"])

    if granularity == "daily":
        last_time = _get_last_time_str(inserts)
        for ix, insert in enumerate(inserts):
            if insert == last_time:
                break
        inserts = inserts[ix + 1:-3]

        # 1 time col, 8 feature cols, 8 unit cols
        df = pd.DataFrame(
            np.array(inserts).reshape((int(len(inserts)/17), 17)),
            columns = [
                "time",
                "temperature_f",
                "temperature_f_unit",
                "dew_point_f",
                "dew_point_f_unit",
                "humidity_percent",
                "humidity_percent_unit",
                "windspeed_mph",
                "windspeed_mph_unit",
                "windspeed_gust_mph",
                "windspeed_gust_mph_unit",
                "pressure_in",
                "pressure_in_unit",
                "precip_rate_in",
                "precip_rate_in_unit",
                "precip_accum_in",
                "precip_accum_in_unit",
            ]
        )
        df["time"] = [t.split("M")[0]+"M" for t in df["time"]]
        df = df[[col for col in df.columns if "unit" not in col]]

    for col in df.columns:
        if col not in ["date", "time"]:
            df[col] = df[col].replace("--", np.nan).astype(float)

    return df


def _get_last_time_str(inserts):
    time_inserts = []
    for ix, insert in enumerate(inserts):
        if any(("AM" in insert, "PM" in insert)):
            if len(insert) < 10:
                time_inserts.append(insert)

    dts = [parse(t.split("M")[0] + "M") for t in time_inserts]
    return max(dts).strftime("%-I:%M %p")


# for chill hours calculation
def get_chill_hours(df, MAX_MINUTES):
    # total hours below 45
    chill_hours_sub_df = df.loc[df.temperature_f < 45]
    chill_hours = chill_hours_sub_df.duration.sum() / 3600
    # confirmed chill hours = confirmed durations (<MIN_MINUTES dur)
    # + first MIN_MINUTES mins of every unconfirmed duration
    chill_hours_confd = get_confirmed_chill_hours(chill_hours_sub_df, MAX_MINUTES)
    return chill_hours, chill_hours_confd


def get_confirmed_chill_hours(chill_hours_sub_df, MAX_MINUTES):
    return (
        chill_hours_sub_df.loc[
            df.duration <= (60 * MAX_MINUTES)
        ].duration.sum() / 3600
        + chill_hours_sub_df.loc[
            df.duration > (60 * MAX_MINUTES)
        ].duration.count() * (MAX_MINUTES * 60) / 3600
    )


def get_modified_chill_hours(df, MAX_MINUTES):
    # hours between 32 and 45
    chill_hours_sub_df = df.loc[
        (df.temperature_f >= 32) & (df.temperature_f < 45)
    ]
    mod_chill_hours = chill_hours_sub_df.duration.sum() / 3600
    # confirmed hours between 32 and 45
    mod_chill_hours_confd = get_confirmed_chill_hours(chill_hours_sub_df, MAX_MINUTES)
    return mod_chill_hours, mod_chill_hours_confd


def get_utah_chill_units(df):
    # utah model
    # 1 hour below 34°F = 0.0 chill unit
    # 1 hour 34.01 - 36°F = 0.5 chill unit
    # 1 hour 36.01 - 48°F = 1.0 chill unit
    # 1 hour 48.01 - 54°F = 0.5 chill unit
    # 1 hour 54.01 - 60°F = 0.0 chill unit
    # 1 hour 60.01 - 65°F = -0.5 chill unit
    # 1 hour >65.01°F = -1.0 chill unit
    sub_34_df = df.loc[df.temperature_f < 34]
    btw_34_and_36_df = df.loc[(df.temperature_f >= 34) & (df.temperature_f < 36)]
    btw_36_and_48_df = df.loc[(df.temperature_f >= 36) & (df.temperature_f < 48)]
    btw_48_and_54_df = df.loc[(df.temperature_f >= 48) & (df.temperature_f < 54)]
    btw_54_and_60_df = df.loc[(df.temperature_f >= 54) & (df.temperature_f < 60)]
    btw_60_and_65_df = df.loc[(df.temperature_f >= 60) & (df.temperature_f < 65)]
    over_65_df = df.loc[df.temperature_f >= 65]
    chill_units = (
        sub_34_df.duration.sum() / 3600 * 0
        + btw_34_and_36_df.duration.sum() / 3600 * 0.5
        + btw_36_and_48_df.duration.sum() / 3600 * 1
        + btw_48_and_54_df.duration.sum() / 3600 * 0.5
        + btw_54_and_60_df.duration.sum() / 3600 * 0
        + btw_60_and_65_df.duration.sum() / 3600 * -0.5
        + over_65_df.duration.sum() / 3600 * -1
    )
    chill_units_confd = (
        get_confirmed_chill_hours(sub_34_df, MAX_MINUTES) * 0
        + get_confirmed_chill_hours(btw_34_and_36_df, MAX_MINUTES) * 0.5
        + get_confirmed_chill_hours(btw_36_and_48_df, MAX_MINUTES) * 1
        + get_confirmed_chill_hours(btw_48_and_54_df, MAX_MINUTES) * 0.5
        + get_confirmed_chill_hours(btw_54_and_60_df, MAX_MINUTES) * 0
        + get_confirmed_chill_hours(btw_60_and_65_df, MAX_MINUTES) * -0.5
        + get_confirmed_chill_hours(over_65_df, MAX_MINUTES) * -1
    )

    return chill_units, chill_units_confd
