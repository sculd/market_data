import datetime
import pytz


def truncate_utc_timestamp_to_minute(timestamp_seconds):
    t_tz = epoch_seconds_to_datetime(timestamp_seconds)
    t_tz_minute = t_tz.replace(second=0, microsecond=0)
    return t_tz_minute

def epoch_seconds_to_datetime(timestamp_seconds):
    t = datetime.datetime.utcfromtimestamp(timestamp_seconds)
    return pytz.utc.localize(t)

def epoch_seconds_to_str(timestamp_seconds):
    return str(epoch_seconds_to_datetime(timestamp_seconds))

def date_str_et_to_t(date_str):
    '''

    :param date_str: e.g. 2024-03-11
    :return:
    '''
    t = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return pytz.timezone('America/New_York').localize(t)
