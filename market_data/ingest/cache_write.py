import pandas as pd
import datetime
import logging
import typing
import os


import market_data.ingest.cache_common
from market_data.util.cache.time import anchor_to_begin_of_day, is_exact_cache_interval


def cache_locally_daily_df(
    df: pd.DataFrame, 
    folder_path: str, 
    t_from: datetime.datetime, 
    t_to: datetime.datetime, 
    overwrite=True,
):
    """
    Cache a DataFrame that covers one-day range exactly.
    """
    if not is_exact_cache_interval(t_from, t_to):
        logging.info(f"{t_from}-{t_to} does not match {market_data.ingest.cache_common.cache_interval=} thus will not be cached.")
        return None

    if len(df) == 0:
        logging.info(f"df for {t_from}-{t_to} is empty thus will not be cached.")
        return None

    filename = market_data.ingest.cache_common.to_local_filename(folder_path, t_from, t_to)
    if os.path.exists(filename):
        logging.info(f"{filename} already exists.")
        if overwrite:
            logging.info(f"and would overwrite it.")
            df.to_parquet(filename)
        else:
            logging.info(f"and would not write it.")
    else:
        df.to_parquet(filename)


def cache_locally_df(
        df: pd.DataFrame,
        folder_path: str,
        overwrite = False,
        warm_up_period_days = 1,
) -> None:
    """
    Cache a DataFrame.

    Chop it into daily pieces and cache them one-by-one.
    skip_first_day is present as the first day of data is often incomplete due to warm-up period.
    """
    if len(df) == 0:
        logging.info(f"df is empty thus will be skipped.")
        return

    def _split_df_by_day() -> typing.List[pd.DataFrame]:
        dfs = [group[1] for group in df.groupby(df.index.get_level_values(market_data.ingest.cache_common.timestamp_index_name).date)]
        return dfs

    df_dailys = _split_df_by_day()
    for i, df_daily in enumerate(df_dailys):
        if i < warm_up_period_days:
            continue

        timestamps = df_daily.index.get_level_values(market_data.ingest.cache_common.timestamp_index_name).unique()
        t_begin = anchor_to_begin_of_day(timestamps[0])
        t_end = anchor_to_begin_of_day(t_begin + market_data.ingest.cache_common.cache_interval)
        cache_locally_daily_df(df_daily, folder_path, t_begin, t_end, overwrite=overwrite)
        del df_daily
