import argparse
import datetime
import pandas as pd
import os
from pathlib import Path

import setup_env # needed for env variables

import main_util
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.util.cache.time import split_t_range
from market_data.feature.registry import list_registered_features
from market_data.feature.cache_writer import cache_feature_cache
from market_data.feature.util import parse_feature_label_param

import market_data.feature.impl  # Import to ensure all features are registered

def _check_missing_feature_data(
        feature_label: str,
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        time_range: TimeRange
) -> list:
    """
    Check which date ranges are missing from the feature cache.
    
    Returns a list of (start_date, end_date) tuples for missing days.
    """
    from market_data.feature.cache_feature import FEATURE_CACHE_BASE_PATH
    from market_data.util.cache.path import to_filename
    
    # Parse feature_label to get params
    feature_label, params = parse_feature_label_param(feature_label)
    params_dir = params.get_params_dir()
    
    t_from, t_to = time_range.to_datetime()
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Check if file exists in cache
        cache_path = f"{FEATURE_CACHE_BASE_PATH}/features"
        filename = to_filename(
            cache_path, 
            feature_label, 
            d_from, 
            d_to, 
            params_dir=params_dir,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode
        )
        
        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges

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
    parser.add_argument('--feature', type=str, default=None,
                        help='Specific feature label to process (required for check and cache actions)')
    
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
    
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite existing cache files')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        # List all available features
        features = list_registered_features()
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
    print(f"  Dataset Mode: {dataset_mode}")
    print(f"  Export Mode: {export_mode}")
    print(f"  Aggregation Mode: {aggregation_mode}")
    print(f"  Time Range: {args.date_from} to {args.date_to}")
    
    # For non-list actions, we now know the feature is specified
    features_to_process = [args.feature]
    
    if args.action == 'check':
        # Process each feature
        for feature_label in features_to_process:
            print(f"\nChecking feature: {feature_label}")
            missing_ranges = _check_missing_feature_data(
                feature_label=feature_label,
                dataset_mode=dataset_mode,
                export_mode=export_mode,
                aggregation_mode=aggregation_mode,
                time_range=time_range
            )
            
            if not missing_ranges:
                print(f"  All data for '{feature_label}' is present in the cache.")
            else:
                # Group consecutive dates
                grouped_ranges = main_util.group_consecutive_dates(missing_ranges)
                
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
            success = cache_feature_cache(
                feature_label_param=feature_label,
                dataset_mode=dataset_mode,
                export_mode=export_mode,
                aggregation_mode=aggregation_mode,
                time_range=time_range,
                calculation_batch_days=args.calculation_batch_days,
                warm_up_days=None,  # Use auto-detection
                overwrite_cache=args.overwrite_cache
            )
            
            if success:
                successful_features.append(feature_label)
                print(f"  Successfully cached feature: {feature_label}")
            else:
                failed_features.append(feature_label)
                print(f"  Failed to cache feature: {feature_label}")
        
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
