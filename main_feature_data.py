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
from market_data.util.cache.time import split_t_range
from market_data.feature.registry import list_registered_features
from market_data.feature.cache_writer import cache_feature_cache
from market_data.feature.util import parse_feature_label_param
import market_data.util.cache.time
import market_data.util.cache.missing_data_finder
import market_data.util.cache.dataframe

import market_data.feature.impl  # Import to ensure all features are registered

def main():
    parser = argparse.ArgumentParser(
        description='Feature data management tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Action argument
    parser.add_argument('--action', type=str, default='cache',
                        choices=['list', 'check', 'cache'],
                        help='Action to perform: list features, check missing data, or cache feature data')
    
    # Feature to process
    parser.add_argument('--feature', type=str, default='all',
                        help='Specific feature label to process (required for check and cache actions). '
                        'Use "all" to process all available features. '
                        'Use "forex" or "crypto" or "stock" to process class specific features. '
                        'Or use a specific feature label such as "bollinger".'
                        )
    
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
    parser.add_argument('--from', dest='date_from', type=str,
                        help='Start date in YYYY-MM-DD format')
    
    parser.add_argument('--to', dest='date_to', type=str,
                        help='End date in YYYY-MM-DD format')
    
    # Optional arguments
    parser.add_argument('--calculation_batch_days', type=int, default=1,
                        help='Number of days to calculate features for in each batch')
    
    parser.add_argument('--warmup-days', type=int, default=None,
                        help='Warm up days. Auto-detection if not provided.')
    
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite existing cache files')
    
    # Multiprocessing arguments
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing using multiprocessing')
    
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: number of CPU cores)')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        # List all available features
        security_type = args.feature or 'all'
        features = list_registered_features(security_type=security_type)
        print(f"\nAvailable features ({len(features)}):")
        for i, feature in enumerate(sorted(features)):
            print(f"  {i+1}. {feature}")
        return
    
    # Check if feature is provided for non-list actions
    if args.action in ['check', 'cache'] and args.feature is None:
        parser.error("--feature is required for 'check' and 'cache' actions")
        
    # For check and cache actions, ensure we have a date range
    if args.action in ['check', 'cache'] and (args.date_from is None or args.date_to is None):
        parser.error("--from and --to arguments are required for 'check' and 'cache' actions")
    
    # Get enum values by name
    dataset_mode = getattr(DATASET_MODE, args.dataset_mode)
    export_mode = getattr(EXPORT_MODE, args.export_mode)
    aggregation_mode = getattr(AGGREGATION_MODE, args.aggregation_mode)
    
    # Create TimeRange object
    time_range = TimeRange(date_str_from=args.date_from, date_str_to=args.date_to)
    
    print(f"Processing with parameters:")
    print(f"  Action: {args.action}")
    print(f"  Feature: {args.feature}")
    print(f"  Dataset Mode: {str(dataset_mode)}")
    print(f"  Export Mode: {str(export_mode)}")
    print(f"  Aggregation Mode: {str(aggregation_mode)}")
    print(f"  Time Range: {args.date_from} to {args.date_to}")
    
    # Determine features to process
    features_to_process = []
    if args.feature in ["all", "forex", "crypto", "stock"]:
        features_to_process = list_registered_features(security_type=args.feature)
        print(f"\nProcessing all {len(features_to_process)} registered {args.feature} features")
    else:
        features_to_process = [args.feature]
    
    if args.action == 'check':
        # Process each feature
        for feature_label in features_to_process:
            print(f"\nChecking feature: {feature_label}")
            missing_ranges = market_data.util.cache.missing_data_finder.check_missing_feature_data(
                feature_label=feature_label,
                feature_params=None,
                dataset_mode=dataset_mode,
                export_mode=export_mode,
                aggregation_mode=aggregation_mode,
                time_range=time_range
            )
            
            if not missing_ranges:
                print(f"  All data for '{feature_label}' is present in the cache.")
            else:
                # Group consecutive dates
                grouped_ranges = market_data.util.cache.time.group_consecutive_dates(missing_ranges)
                
                total_missing_days = len(missing_ranges)
                print(f"  Missing data for '{feature_label}': {total_missing_days} day(s), grouped into {len(grouped_ranges)} range(s):")
                
                for i, (d_from, d_to) in enumerate(grouped_ranges):
                    if d_from.date() == d_to.date() - datetime.timedelta(days=1):
                        # Single day range (common when using daily intervals)
                        print(f"    {i+1}. {d_from.date()}")
                    else:
                        # Multi-day range
                        print(f"    {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
                
                # Suggest command to cache this feature
                suggest_cmd = f"python main_feature_data.py --action cache --feature {feature_label} --from {args.date_from} --to {args.date_to}"
                print(f"\n  To cache this feature, run:")
                print(f"    {suggest_cmd}")
    
    elif args.action == 'cache':
        # Process each feature
        successful_features = []
        failed_features = []
        
        for feature_label in features_to_process:
            print(f"\nCaching feature: {feature_label}")
            
            try:
                # Set up calculation parameters
                calculation_batch_days = args.calculation_batch_days
                if calculation_batch_days <= 0:
                    calculation_batch_days = 1

                missing_range_finder_func = partial(
                    market_data.util.cache.missing_data_finder.check_missing_feature_data,
                    dataset_mode=dataset_mode,
                    export_mode=export_mode,
                    aggregation_mode=aggregation_mode,
                    feature_label=feature_label,
                    feature_params=None,
                    )

                calculation_ranges = market_data.util.cache.time.chop_missing_time_range(
                    missing_range_finder_func=missing_range_finder_func,
                    time_range=time_range,
                    overwrite_cache=args.overwrite_cache,
                    calculation_batch_days=calculation_batch_days
                )
                
                # Process each calculation range for this feature
                if args.parallel:
                    # Parallel processing
                    if args.workers is None:
                        workers = multiprocessing.cpu_count()
                    else:
                        workers = args.workers
                    
                    print(f"  Using parallel processing with {workers} workers")

                    cache_func = partial(
                        cache_feature_cache,
                        feature_label_param=feature_label,
                        dataset_mode=dataset_mode,
                        export_mode=export_mode,
                        aggregation_mode=aggregation_mode,
                        calculation_batch_days=1,  # Process each range as single batch
                        warm_up_days=args.warmup_days,
                        overwrite_cache=args.overwrite_cache,
                    )

                    time_ranges = [TimeRange(t_from=t_from, t_to=t_to) for t_from, t_to in calculation_ranges]
                    successful_batches, failed_batches = market_data.util.cache.dataframe.cache_multiprocess(
                        cache_func=cache_func,
                        time_ranges=time_ranges,
                        workers=workers
                    )
                    feature_success = True
                    if failed_batches > 0:
                        feature_success = False
                        print(f"  Failed to process feature: {feature_label} with {successful_batches} successful batches and {failed_batches} failed batches")

                else:
                    # Sequential processing (original behavior)
                    feature_success = True
                    for i, calc_range in enumerate(calculation_ranges):
                        calc_t_from, calc_t_to = calc_range
                        print(f"  Processing batch {i+1}/{len(calculation_ranges)}: {calc_t_from.date()} to {calc_t_to.date()}")
                        
                        calc_time_range = TimeRange(calc_t_from, calc_t_to)
                        
                        success = cache_feature_cache(
                            feature_label_param=feature_label,
                            dataset_mode=dataset_mode,
                            export_mode=export_mode,
                            aggregation_mode=aggregation_mode,
                            time_range=calc_time_range,
                            calculation_batch_days=1,  # Process each range as single batch
                            warm_up_days=args.warmup_days,
                            overwrite_cache=args.overwrite_cache
                        )
                        
                        if not success:
                            feature_success = False
                            print(f"  Failed to process batch {i+1} for feature: {feature_label}")
                
                if feature_success:
                    successful_features.append(feature_label)
                    print(f"  Successfully cached feature: {feature_label}")
                else:
                    failed_features.append(feature_label)
                    print(f"  Failed to cache feature: {feature_label}")
                    
            except Exception as e:
                failed_features.append(feature_label)
                print(f"  Failed to cache feature {feature_label}: {e}")
        
        # Summary
        print("\nCaching summary:")
        print(f"  Successfully cached: {len(successful_features)} feature(s)")
        print(f"  Failed to cache: {len(failed_features)} feature(s)")
        
        if failed_features:
            print("\nFailed features:")
            for feature in failed_features:
                print(f"  - {feature}")

if __name__ == "__main__":
    main()
