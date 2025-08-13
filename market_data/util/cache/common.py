import datetime
import os

import pytz

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
