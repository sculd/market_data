import argparse
import datetime
import pandas as pd
import os
import multiprocessing
from pathlib import Path
from functools import partial

import setup_env # needed for env variables

from market_data.ingest.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.target.target import TargetParamsBatch, TargetParams
from market_data.target.cache_target import calculate_and_cache_targets
import market_data.util.cache.time
import market_data.util.cache.missing_data_finder
import market_data.util.cache.dataframe

def main():
    parser = argparse.ArgumentParser(
        description='Target data management tool',
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
    
    # Target parameters
    parser.add_argument('--forward_periods', type=str,
                        help='Comma-separated list of forward periods (e.g., "1,2,3")')
    
    parser.add_argument('--tps', type=str,
                        help='Comma-separated list of target price shifts (e.g., "0.001,0.002,0.003")')
    
    # Multiprocessing arguments
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing using multiprocessing')
    
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: number of CPU cores)')

    args = parser.parse_args()
    
    # Validate forward_periods and tps are specified together
    if (args.forward_periods is None) != (args.tps is None):
        parser.error("--forward_periods and --tps must be specified together")
    
    # Get enum values by name
    dataset_mode = getattr(DATASET_MODE, args.dataset_mode)
    export_mode = getattr(EXPORT_MODE, args.export_mode)
    aggregation_mode = getattr(AGGREGATION_MODE, args.aggregation_mode)
    
    # Create TimeRange object
    time_range = TimeRange(date_str_from=args.date_from, date_str_to=args.date_to)
    
    print(f"Processing with parameters:")
    print(f"  Action: {args.action}")
    print(f"  Dataset Mode: {str(dataset_mode)}")
    print(f"  Export Mode: {str(export_mode)}")
    print(f"  Aggregation Mode: {str(aggregation_mode)}")
    print(f"  Time Range: {args.date_from} to {args.date_to}")
    if args.forward_periods and args.tps:
        print(f"  Forward Periods: {args.forward_periods}")
        print(f"  Target Price Shifts: {args.tps}")

    target_params: TargetParamsBatch = None
    if args.forward_periods and args.tps:
        target_params = TargetParamsBatch(
            target_params_list=[TargetParams(forward_period=int(period), tp_value=float(tp), sl_value=float(tp)) 
                for period in args.forward_periods.split(',') 
                for tp in args.tps.split(',')]
        )

    if args.action == 'check':
        print("\nChecking target data")
        missing_ranges = market_data.util.cache.missing_data_finder.check_missing_target_data(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            target_params=target_params,
        )
        
        if not missing_ranges:
            print(f"  All target data is present in the cache.")
        else:
            # Group consecutive dates
            grouped_ranges = market_data.util.cache.time.group_consecutive_dates(missing_ranges)
            
            total_missing_days = len(missing_ranges)
            print(f"  Missing target data: {total_missing_days} day(s), grouped into {len(grouped_ranges)} range(s):")
            
            for i, (d_from, d_to) in enumerate(grouped_ranges):
                if d_from.date() == d_to.date() - datetime.timedelta(days=1):
                    # Single day range (common when using daily intervals)
                    print(f"    {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    print(f"    {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
            
            # Suggest command to cache target data
            suggest_cmd = f"python main_target_data.py --action cache --from {args.date_from} --to {args.date_to}"
            if args.forward_periods and args.tps:
                suggest_cmd += f" --forward_periods {args.forward_periods} --tps {args.tps}"
            print(f"\n  To cache target data, run:")
            print(f"    {suggest_cmd}")
    
    elif args.action == 'cache':
        print("\nCaching target data")
        try:
            missing_range_finder_func = partial(
                market_data.util.cache.missing_data_finder.check_missing_target_data,
                dataset_mode=dataset_mode,
                export_mode=export_mode,
                aggregation_mode=aggregation_mode,
                target_params=target_params,
                )

            calculation_ranges = market_data.util.cache.time.chop_missing_time_range(
                missing_range_finder_func=missing_range_finder_func,
                time_range=time_range,
                overwrite_cache=args.overwrite_cache,
                calculation_batch_days=args.calculation_batch_days
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
                    calculate_and_cache_targets,
                    dataset_mode=dataset_mode,
                    export_mode=export_mode,
                    aggregation_mode=aggregation_mode,
                    params=target_params,
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
                    
                    calculate_and_cache_targets(
                        dataset_mode=dataset_mode,
                        export_mode=export_mode,
                        aggregation_mode=aggregation_mode,
                        time_range=calc_time_range,
                        params=target_params,
                        calculation_batch_days=args.calculation_batch_days,
                        overwrite_cache=args.overwrite_cache,
                    )

            print("  Successfully cached target data")
        except Exception as e:
            print(f"  Failed to cache target data: {e}")

if __name__ == "__main__":
    main()
