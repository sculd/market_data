import argparse
import datetime
import pandas as pd
import os
from pathlib import Path

import setup_env # needed for env variables

import main_util
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.machine_learning.resample import ResampleParams
from market_data.machine_learning.cache_resample import calculate_and_cache_resampled
import market_data.util.cache.missing_data_finder

def main():
    parser = argparse.ArgumentParser(
        description='Resampled data management tool',
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
    parser.add_argument('--calculation_batch_days', type=int, default=1,
                        help='Number of days to calculate features for in each batch')
    
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite existing cache files')
    
    # Resample parameters if using the --resample flag
    parser.add_argument('--resample_params', type=str, default=None,
                        help='Resampling parameters in format "price_col,threshold" (e.g., "close,0.07")')
    
    args = parser.parse_args()
    
    # Get enum values by name
    dataset_mode = getattr(DATASET_MODE, args.dataset_mode)
    export_mode = getattr(EXPORT_MODE, args.export_mode)
    aggregation_mode = getattr(AGGREGATION_MODE, args.aggregation_mode)
    
    # Create TimeRange object
    time_range = TimeRange(date_str_from=args.date_from, date_str_to=args.date_to)
    
    # Parse resample parameters if provided
    resample_params = None
    if args.resample_params:
        resample_params = main_util.parse_resample_params(args.resample_params)
    
    print(f"Processing with parameters:")
    print(f"  Action: {args.action}")
    print(f"  Processing: Resampled data")
    if resample_params:
        print(f"  Resample Params: price_col={resample_params.price_col}, threshold={resample_params.threshold}")
    
    print(f"  Dataset Mode: {dataset_mode}")
    print(f"  Export Mode: {export_mode}")
    print(f"  Aggregation Mode: {aggregation_mode}")
    print(f"  Time Range: {args.date_from} to {args.date_to}")

    if args.action == 'check':
        print("\nChecking resampled data")
        missing_ranges = market_data.util.cache.missing_data_finder.check_missing_resampled_data(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            resample_params=resample_params
        )
        
        if not missing_ranges:
            print(f"  All resampled data is present in the cache.")
        else:
            # Group consecutive dates
            grouped_ranges = market_data.util.cache.missing_data_finder.group_consecutive_dates(missing_ranges)
            
            total_missing_days = len(missing_ranges)
            print(f"  Missing resampled data: {total_missing_days} day(s), grouped into {len(grouped_ranges)} range(s):")
            
            for i, (d_from, d_to) in enumerate(grouped_ranges):
                if d_from.date() == d_to.date() - datetime.timedelta(days=1):
                    # Single day range (common when using daily intervals)
                    print(f"    {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    print(f"    {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
            
            # Suggest command to cache resampled data
            resample_param_option = f" --resample_params {args.resample_params}" if args.resample_params else ""
            suggest_cmd = f"python main_resampled_data.py --action cache {resample_param_option} --from {args.date_from} --to {args.date_to}"
            print(f"\n  To cache resampled data, run:")
            print(f"    {suggest_cmd}")
    
    elif args.action == 'cache':
        print("\nCaching resampled data")
        try:
            calculate_and_cache_resampled(
                dataset_mode=dataset_mode,
                export_mode=export_mode,
                aggregation_mode=aggregation_mode,
                params=resample_params,
                time_range=time_range,
                calculation_batch_days=args.calculation_batch_days,
                overwrite_cache=args.overwrite_cache
            )
            print("  Successfully cached resampled data")
        except Exception as e:
            print(f"  Failed to cache resampled data: {e}")

if __name__ == "__main__":
    main()
