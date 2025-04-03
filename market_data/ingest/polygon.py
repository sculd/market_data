import time

import logging
import requests
import pandas as pd
import numpy as np
import os

import urllib.parse as urlparse
from urllib.parse import urlencode
from ..util.cache.dataframe import cache_to_disk

polygon_api_key = os.getenv("QUANT_GALORE_POLYGON_API_KEY")
_POLYGON_V2_BASE_URL = "https://api.polygon.io/v2"
_POLYGON_V3_BASE_URL = "https://api.polygon.io/v3"

def add_params_to_url(url, params):
    url_parts = list(urlparse.urlparse(url))
    query = dict(urlparse.parse_qsl(url_parts[4]))
    query.update(params)
    url_parts[4] = urlencode(query)
    return urlparse.urlunparse(url_parts)

def add_default_ticker_params_to_polygon_url(url, additional_params = None):
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': '50000',
        'apiKey': polygon_api_key,
    }
    if additional_params is not None:
        params.update(additional_params)
    return add_params_to_url(url, params)

def add_default_options_params_to_polygon_url(url, additional_params = None):
    params = {
        'limit': '1000',
        'apiKey': polygon_api_key,
    }
    if additional_params is not None:
        params.update(additional_params)
    return add_params_to_url(url, params)

def get_polygon_range_query_url(ticker, date_str_begin, date_str_end):
    return add_default_ticker_params_to_polygon_url(f"{_POLYGON_V2_BASE_URL}/aggs/ticker/{ticker}/range/1/day/{date_str_begin}/{date_str_end}")

def get_polygon_intraday_query_url(ticker, date_str):
    return add_default_ticker_params_to_polygon_url(f"{_POLYGON_V2_BASE_URL}/aggs/ticker/{ticker}/range/1/minute/{date_str}/{date_str}")

def get_polygon_options_contracts_query_url(underlying_ticker, option_type, date_str, exp_date_str):
    params = {
        "underlying_ticker": underlying_ticker,
        "contract_type": option_type,
        "as_of": date_str,
    }
    if exp_date_str is not None:
        params["expiration_date"] = exp_date_str

    return add_default_options_params_to_polygon_url(f"{_POLYGON_V3_BASE_URL}/reference/options/contracts", additional_params = params)

def get_polygon_quotes_url(ticker, epoch_nano_gte, epoch_nano_lt):
    return add_default_ticker_params_to_polygon_url(f"https://api.polygon.io/v3/quotes/{ticker}?timestamp.gte={epoch_nano_gte}&timestamp.lt={epoch_nano_lt}&order=asc&limit=10&sort=timestamp&apiKey={polygon_api_key}")


_max_tries = 3

# cached for 3 days.
@cache_to_disk(10)
def get_polygon_result_dict(url, tries_remaining=_max_tries):
    try:
        js = requests.get(url).json()
        if "results" not in js:
            return {}
        return js["results"]
    except requests.exceptions.ConnectionError as ex:
        print(f"{tries_remaining=}\n{ex}")
        if tries_remaining == 0:
            return {}
        else:
            sleep_seconds = (_max_tries - tries_remaining) * 1.0
            logging.info(f"sleeping {sleep_seconds} seconds before making retry.")
            time.sleep(sleep_seconds)
            return get_polygon_result_dict(url, tries_remaining=tries_remaining-1)

def polygon_result_to_dataframe(result):
    df = pd.json_normalize(result)
    if df.empty:
        return df
    if "t" in df.columns:
        df = df.set_index("t")
        df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("America/New_York")
        #df['date'] = df.index.strftime('%Y-%m-%d')
    return df

def polygon_url_to_dataframe(url):
    return polygon_result_to_dataframe(get_polygon_result_dict(url))