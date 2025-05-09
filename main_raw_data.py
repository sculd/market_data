import argparse
import datetime
import pandas as pd
import os
from pathlib import Path

import setup_env # needed for env variables

import main_util
from market_data.ingest.bq.cache import query_and_cache, to_filename, _cache_base_path
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.util.time import TimeRange
from market_data.util.cache.time import split_t_range
import market_data.util.cache.missing_data_finder

def main():
    parser = argparse.ArgumentParser(
        description='Query and cache raw market data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Action argument
    parser.add_argument('--action', type=str, default='cache',
                        choices=['check', 'cache'],
                        help='Action to perform: check missing data or cache data')
    
    # Arguments with defaults
    parser.add_argument('--dataset_mode', type=str, default='OKX', 
                        choices=[mode.name for mode in DATASET_MODE],
                        help='Dataset mode')
    
    parser.add_argument('--export_mode', type=str, default='BY_MINUTE',
                        choices=[mode.name for mode in EXPORT_MODE],
                        help='Export mode')
    
    parser.add_argument('--aggregation_mode', type=str, default='TAKE_LASTEST',
                        choices=[mode.name for mode in AGGREGATION_MODE],
                        help='Aggregation mode')
    
    # Time range arguments - can be specified as date strings
    parser.add_argument('--from', dest='date_from', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    
    parser.add_argument('--to', dest='date_to', type=str, required=True,
                        help='End date in YYYY-MM-DD format')
    
    # Optional arguments
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite existing cache files')
    
    parser.add_argument('--skip_first_day', action='store_true',
                        help='Skip the first day in the range')
    
    args = parser.parse_args()
    
    # Get enum values by name
    dataset_mode = getattr(DATASET_MODE, args.dataset_mode)
    export_mode = getattr(EXPORT_MODE, args.export_mode)
    aggregation_mode = getattr(AGGREGATION_MODE, args.aggregation_mode)
    
    # Create TimeRange object
    time_range = TimeRange(date_str_from=args.date_from, date_str_to=args.date_to)
    
    print(f"Processing with parameters:")
    print(f"  Action: {args.action}")
    print(f"  Dataset Mode: {dataset_mode}")
    print(f"  Export Mode: {export_mode}")
    print(f"  Aggregation Mode: {aggregation_mode}")
    print(f"  Time Range: {args.date_from} to {args.date_to}")
    
    if args.action == 'check':
        # Check which date ranges are missing from the cache
        missing_ranges = market_data.util.cache.missing_data_finder.check_missing_raw_data(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range
        )
        
        if not missing_ranges:
            print("\nAll data for the specified range is present in the cache.")
        else:
            # Group consecutive dates
            grouped_ranges = main_util.group_consecutive_dates(missing_ranges)
            
            total_missing_days = len(missing_ranges)
            print(f"\nMissing data for {total_missing_days} day(s), grouped into {len(grouped_ranges)} range(s):")
            
            for i, (d_from, d_to) in enumerate(grouped_ranges):
                if d_from.date() == d_to.date() - datetime.timedelta(days=1):
                    # Single day range (common when using daily intervals)
                    print(f"  {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    print(f"  {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
                
            # Suggest command to cache the missing data
            suggest_cmd = f"python main_raw_data.py --action cache --dataset_mode {args.dataset_mode} --export_mode {args.export_mode} --aggregation_mode {args.aggregation_mode} --from {args.date_from} --to {args.date_to}"
            print(f"\nTo cache the missing data, run:")
            print(f"  {suggest_cmd}")
    
    else:  # args.action == 'cache'
        print(f"  Overwrite Cache: {args.overwrite_cache}")
        print(f"  Skip First Day: {args.skip_first_day}")
        
        # Query and cache data
        result_df = query_and_cache(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            t_from=None,
            t_to=None,
            date_str_from=args.date_from,
            date_str_to=args.date_to,
            overwirte_cache=args.overwrite_cache,
            skip_first_day=args.skip_first_day
        )
        
        # Print summary of results
        if result_df is not None and not result_df.empty:
            print(f"\nSuccessfully cached {len(result_df)} rows of data")
            print(f"Data spans from {result_df.index.min()} to {result_df.index.max()}")
            print(f"Columns: {result_df.columns.tolist()}")
            print(f"Number of unique symbols: {result_df['symbol'].nunique()}")
        else:
            print("\nNo data was returned or cached.")

if __name__ == "__main__":
    main()
