import datetime
import pytz
import typing


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


def to_t(
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
        ) -> typing.Tuple[datetime.datetime, datetime.datetime]:
    if t_from is not None and t_to is not None:
        pass
    elif epoch_seconds_from is not None and epoch_seconds_to is not None:
        t_from = epoch_seconds_to_datetime(epoch_seconds_from)
        t_to = epoch_seconds_to_datetime(epoch_seconds_to)
    elif date_str_from is not None and date_str_to is not None:
        t_from = date_str_et_to_t(date_str_from)
        t_to = date_str_et_to_t(date_str_to)
    else:
        raise Exception("the time range must be specified.")

    return t_from, t_to
