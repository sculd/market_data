import argparse
import datetime
import pandas as pd
import os
import multiprocessing
from pathlib import Path
from functools import partial

import setup_env # needed for env variables

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.machine_learning.resample import (
    get_resample_params_class,
    get_resample_function,
    list_registered_resample_methods
)
from market_data.machine_learning.resample.cache_resample import calculate_and_cache_resampled
import market_data.util.cache.time
import market_data.util.cache.missing_data_finder
import market_data.util.cache.dataframe


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
    
    # Resample type and parameters
    available_methods = list_registered_resample_methods()
    parser.add_argument('--resample_type_label', type=str, default='cumsum',
                        choices=available_methods,
                        help=f'Resampling method to use. Available: {", ".join(available_methods)}')
    
    parser.add_argument('--resample_params', type=str, default=None,
                        help='Resampling parameters. Format depends on method: '
                             'cumsum: "price_col,threshold" (e.g., "close,0.07") '
                             'reversal: "price_col,threshold,threshold_reversal" (e.g., "close,0.1,0.03")')
    
    # Multiprocessing arguments
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing using multiprocessing')
    
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: number of CPU cores)')
    
    args = parser.parse_args()
    
    # Get enum values by name
    dataset_mode = getattr(DATASET_MODE, args.dataset_mode)
    export_mode = getattr(EXPORT_MODE, args.export_mode)
    aggregation_mode = getattr(AGGREGATION_MODE, args.aggregation_mode)
    
    # Create TimeRange object
    time_range = TimeRange(date_str_from=args.date_from, date_str_to=args.date_to)
    
    # Get resample method components from registry
    resample_params_class = get_resample_params_class(args.resample_type_label)
    resample_function = get_resample_function(args.resample_type_label)
    
    if resample_params_class is None:
        print(f"Error: Unknown resample type '{args.resample_type_label}'")
        print(f"Available methods: {', '.join(list_registered_resample_methods())}")
        return 1
    
    if resample_function is None:
        print(f"Error: No function registered for resample type '{args.resample_type_label}'")
        return 1
    
    # Parse resample parameters if provided
    resample_params = None
    if args.resample_params:
        resample_params = resample_params_class.parse_resample_params(args.resample_params)
    
    print(f"Processing with parameters:")
    print(f"  Action: {args.action}")
    print(f"  Processing: Resampled data")
    print(f"  Resample Type: {args.resample_type_label}")
    if resample_params:
        # Create a display string for the parameters
        param_display = []
        for field_name, field_value in resample_params.__dict__.items():
            param_display.append(f"{field_name}={field_value}")
        print(f"  Resample Params: {', '.join(param_display)}")
    
    print(f"  Dataset Mode: {str(dataset_mode)}")
    print(f"  Export Mode: {str(export_mode)}")
    print(f"  Aggregation Mode: {str(aggregation_mode)}")
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
            grouped_ranges = market_data.util.cache.time.group_consecutive_dates(missing_ranges)
            
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
            resample_type_option = f" --resample_type_label {args.resample_type_label}" if args.resample_type_label != 'cumsum' else ""
            suggest_cmd = f"python main_resampled_data.py --action cache{resample_type_option}{resample_param_option} --from {args.date_from} --to {args.date_to}"
            print(f"\n  To cache resampled data, run:")
            print(f"    {suggest_cmd}")
    
    elif args.action == 'cache':
        print("\nCaching resampled data")
        try:
            # Set up calculation parameters
            calculation_batch_days = args.calculation_batch_days
            if calculation_batch_days <= 0:
                calculation_batch_days = 1
            
            missing_range_finder_func = partial(
                market_data.util.cache.missing_data_finder.check_missing_resampled_data,
                dataset_mode=dataset_mode,
                export_mode=export_mode,
                aggregation_mode=aggregation_mode,
                resample_params=resample_params,
                )

            calculation_ranges = market_data.util.cache.time.chop_missing_time_range(
                missing_range_finder_func=missing_range_finder_func,
                time_range=time_range,
                overwrite_cache=args.overwrite_cache,
                calculation_batch_days=calculation_batch_days
            )

            # Process each calculation range
            if args.parallel:
                # Parallel processing
                if args.workers is None:
                    workers = multiprocessing.cpu_count()
                else:
                    workers = args.workers
                
                print(f"  Using parallel processing with {workers} workers")

                cache_func = partial(
                    calculate_and_cache_resampled,
                    dataset_mode=dataset_mode,
                    export_mode=export_mode,
                    aggregation_mode=aggregation_mode,
                    resample_at_events_func=resample_function,
                    params=resample_params,
                    calculation_batch_days=1,  # Process each range as single batch
                    overwrite_cache=args.overwrite_cache,
                )

                time_ranges = [TimeRange(t_from=t_from, t_to=t_to) for t_from, t_to in calculation_ranges]
                market_data.util.cache.dataframe.cache_multiprocess(
                    cache_func=cache_func,
                    time_ranges=time_ranges,
                    workers=workers
                )
            else:
                # Sequential processing (original behavior)
                for i, calc_range in enumerate(calculation_ranges):
                    calc_t_from, calc_t_to = calc_range
                    print(f"  Processing batch {i+1}/{len(calculation_ranges)}: {calc_t_from.date()} to {calc_t_to.date()}")
                    
                    calc_time_range = TimeRange(calc_t_from, calc_t_to)
                    
                    calculate_and_cache_resampled(
                        dataset_mode=dataset_mode,
                        export_mode=export_mode,
                        aggregation_mode=aggregation_mode,
                        resample_at_events_func=resample_function,
                        params=resample_params,
                        time_range=calc_time_range,
                        calculation_batch_days=1,  # Process each range as single batch
                        overwrite_cache=args.overwrite_cache
                    )
            
            print("  Successfully cached resampled data")
        except Exception as e:
            print(f"  Failed to cache resampled data: {e}")

if __name__ == "__main__":
    main()
