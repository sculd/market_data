import datetime
import pytz
import os

from market_data.util.cache.path import get_cache_base_path


# the cache will be stored per day.
cache_interval = datetime.timedelta(days=1)
cache_base_path = get_cache_base_path()
try:
    os.mkdir(cache_base_path)
except FileExistsError:
    pass

# the timezone of dividing range into days.
cache_timezone = pytz.timezone('America/New_York')

timestamp_index_name = 'timestamp'


def to_local_filename(folder_path: str, t_from: datetime.date, t_to: datetime.datetime) -> str:
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    fn = os.path.join(folder_path, f"{t_str_from}_{t_str_to}.parquet")
    dir = os.path.dirname(fn)
    try:
        os.makedirs(dir, exist_ok=True)
    except FileExistsError:
        pass

    return fn
