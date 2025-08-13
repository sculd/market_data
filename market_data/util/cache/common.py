import datetime

import pytz

# the cache will be stored per day.
cache_interval = datetime.timedelta(days=1)

# the timezone of dividing range into days.
cache_timezone = pytz.timezone('America/New_York')

timestamp_index_name = 'timestamp'
