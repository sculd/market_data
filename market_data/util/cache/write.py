import datetime
import logging
import os
import typing
from typing import Callable, TypeVar

import pandas as pd

import market_data.util.cache.common
from market_data.util.cache.time import (anchor_to_begin_of_day,
                                         is_exact_cache_interval,
                                         split_t_range)
from market_data.util.time import TimeRange

logger = logging.getLogger(__name__)


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
        logging.info(f"{t_from}-{t_to} does not match {market_data.util.cache.common.cache_interval=} thus will not be cached.")
        return None

    if len(df) == 0:
        logging.info(f"df for {t_from}-{t_to} is empty thus will not be cached.")
        return None

    filename = market_data.util.cache.common.to_local_filename(folder_path, t_from, t_to)
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
        dfs = [group[1] for group in df.groupby(df.index.get_level_values(market_data.util.cache.common.timestamp_index_name).date)]
        return dfs

    df_dailys = _split_df_by_day()
    for i, df_daily in enumerate(df_dailys):
        if i < warm_up_period_days:
            continue

        timestamps = df_daily.index.get_level_values(market_data.util.cache.common.timestamp_index_name).unique()
        t_begin = anchor_to_begin_of_day(timestamps[0])
        t_end = anchor_to_begin_of_day(t_begin + market_data.util.cache.common.cache_interval)
        cache_locally_daily_df(df_daily, folder_path, t_begin, t_end, overwrite=overwrite)
        del df_daily


# Type variable for the params
T = TypeVar('T')

def calculate_and_cache_data(
    raw_data_folder_path: str,
    folder_path: str,
    params: T,
    time_range: TimeRange,
    calculation_batch_days: int,
    warm_up_days: int,
    overwrite_cache: bool = True,
    calculate_batch_fn: Callable[[pd.DataFrame, T], pd.DataFrame] = None,
) -> None:
    # Resolve time range
    t_from, t_to = time_range.to_datetime()
    calculation_interval = datetime.timedelta(days=calculation_batch_days if calculation_batch_days >= 1 else 1)
    warm_up_period = datetime.timedelta(days=warm_up_days)
    calculation_ranges = split_t_range(t_from, t_to, interval=calculation_interval, warm_up=warm_up_period)
 
    for calc_range in calculation_ranges:
        calc_t_from, calc_t_to = calc_range
        logger.info(f"Processing calculation batch {calc_t_from} to {calc_t_to}")
        
        # Calculate data for this batch
        try:
            raw_df = market_data.util.cache.read.read_from_local_cache(
                folder_path=raw_data_folder_path,
                time_range=TimeRange(calc_t_from, calc_t_to),
            )
            if raw_df is None or len(raw_df) == 0:
                logger.warning(f"No raw data available for {calc_t_from} to {calc_t_to}")
                continue

            result_df = calculate_batch_fn(raw_df, params)
            if result_df is None or len(result_df) == 0:
                logger.warning(f"calculation returned empty result for {calc_t_from} to {calc_t_to}")
                continue
                
            cache_locally_daily_df(
                df=result_df, 
                folder_path=folder_path, 
                t_from=calc_t_from + warm_up_period, 
                t_to=calc_t_to, 
                overwrite=overwrite_cache,
            )
            
        except Exception as e:
            logger.error(f"Error calculating for {calc_t_from} to {calc_t_to}: {e}")
            continue
