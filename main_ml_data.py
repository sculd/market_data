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
from market_data.target.target import TargetParamsBatch, TargetParams
from market_data.machine_learning.resample import ResampleParams
from market_data.feature.impl.common import SequentialFeatureParam
from market_data.feature.util import parse_feature_label_params
from market_data.feature.registry import list_registered_features
from market_data.machine_learning.cache_ml_data import (
    calculate_and_cache_ml_data,
    load_cached_ml_data,
    CACHE_BASE_PATH
)
import market_data.util.cache.missing_data_finder
from market_data.machine_learning.ml_data import prepare_ml_data, prepare_sequential_ml_data

def main():
    parser = argparse.ArgumentParser(
        description='ML data management tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Action argument
    parser.add_argument('--action', type=str, default='cache',
                        choices=['check', 'cache'],
                        help='Action to perform: check missing data or cache ML data')
    
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
    
    # Time range arguments
    parser.add_argument('--from', dest='date_from', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    
    parser.add_argument('--to', dest='date_to', type=str, required=True,
                        help='End date in YYYY-MM-DD format')
    
    # Feature handling
    parser.add_argument('--features', type=str, default="all",
                        help='Specific feature labels to include (comma separated). "forex", "stock", "crypto" or "all" are special values')
                        
    # Optional arguments
    parser.add_argument('--sequential', action='store_true',
                        help='Create sequential features for ML data')
    
    parser.add_argument('--sequence_window', type=int, default=30,
                        help='Window size for sequential features')
    
    # Target parameters
    parser.add_argument('--forward_periods', type=str,
                        help='Comma-separated list of forward periods (e.g., "1,2,3"). For forex, 10,30,60 are used')
    
    parser.add_argument('--tps', type=str,
                        help='Comma-separated list of target price shifts (e.g., "0.001,0.002,0.003"). For forex, 0.0025,0.005,0.01 are used')
    
    parser.add_argument('--resample_params', type=str, default=None,
                        help='Resampling parameters in format "price_col,threshold" (e.g., "close,0.07"). For forex, close,0.0025" "close,0.005" "close,0.01" are used')
                        
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite existing cache files')
    
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
    
    # Create parameter objects
    features_to_process = []
    if args.features == "all":
        features_to_process = list_registered_features()
        print(f"\nProcessing all {len(features_to_process)} registered features")
    elif args.features == "forex":
        features_to_process = list_registered_features(security_type="forex")
        print(f"\nProcessing all {len(features_to_process)} registered forex features")
    elif args.features == "crypto":
        features_to_process = list_registered_features(security_type="crypto")
        print(f"\nProcessing all {len(features_to_process)} registered crypto features")
    elif args.features == "stock":
        features_to_process = list_registered_features(security_type="stock")
        print(f"\nProcessing all {len(features_to_process)} registered stock features")
    else:
        features_to_process = [f.strip() for f in args.features.split(",")]
    
    feature_label_params = features_to_process  # with default parameters
    target_params_batch = TargetParamsBatch()
    if args.forward_periods and args.tps:
        target_params_batch = TargetParamsBatch(
            target_params_list=[TargetParams(forward_period=int(period), tp_value=float(tp), sl_value=float(tp)) 
                for period in args.forward_periods.split(',') 
                for tp in args.tps.split(',')]
        )
    
    # Parse resample parameters
    resample_params = main_util.parse_resample_params(args.resample_params)
    
    # Create sequential parameters if needed
    seq_params = None
    if args.sequential:
        seq_params = SequentialFeatureParam(sequence_window=args.sequence_window)
    
    print(f"Processing with parameters:")
    print(f"  Action: {args.action}")
    print(f"  Dataset Mode: {str(dataset_mode)}")
    print(f"  Export Mode: {str(export_mode)}")
    print(f"  Aggregation Mode: {str(aggregation_mode)}")
    print(f"  Time Range: {args.date_from} to {args.date_to}")
    print(f"  Features: {args.features if args.features else 'All registered features'}")
    if args.forward_periods and args.tps:
        print(f"  Forward Periods: {args.forward_periods}")
        print(f"  Target Price Shifts: {args.tps}")
    print(f"  Resample Params: price_col={resample_params.price_col}, threshold={resample_params.threshold}")
    print(f"  Sequential: {args.sequential}")
    if args.sequential:
        print(f"  Sequence Window: {args.sequence_window}")
    
    if args.action == 'check':
        print("\nChecking ml_data")
        # Check which date ranges are missing from the ML data cache
        missing_ranges = market_data.util.cache.missing_data_finder.check_missing_ml_data(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            feature_label_params=feature_label_params,
            target_params_batch=target_params_batch,
            resample_params=resample_params,
            seq_params=seq_params
        )
        
        if not missing_ranges:
            print("\nAll ML data for the specified range is present in the cache.")
        else:
            # Group consecutive dates
            grouped_ranges = market_data.util.cache.missing_data_finder.group_consecutive_dates(missing_ranges)
            
            total_missing_days = len(missing_ranges)
            print(f"\nMissing ML data: {total_missing_days} day(s), grouped into {len(grouped_ranges)} range(s):")
            
            for i, (d_from, d_to) in enumerate(grouped_ranges):
                if d_from.date() == d_to.date() - datetime.timedelta(days=1):
                    # Single day range
                    print(f"  {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    print(f"  {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
            
            # Suggest command to cache the ML data
            features_str = ""
            if args.features:
                features_str = f"--features {args.features}"
                
            sequential_str = "--sequential" if args.sequential else ""
            seq_window_str = f"--sequence_window {args.sequence_window}" if args.sequential else ""
            
            suggest_cmd = (f"python main_ml_data.py --action cache "
                          f"--dataset_mode {args.dataset_mode} "
                          f"--export_mode {args.export_mode} "
                          f"--aggregation_mode {args.aggregation_mode} "
                          f"--from {args.date_from} --to {args.date_to} "
                          f"--resample_params '{args.resample_params}' "
                          f"{features_str} {sequential_str} {seq_window_str}")
            
            print(f"\nTo cache the missing ML data, run:")
            print(f"  {suggest_cmd}")
    
    elif args.action == 'cache':
        print("\nCalculating and caching ML data...")
        
        try:
            calculate_and_cache_ml_data(
                dataset_mode=dataset_mode,
                export_mode=export_mode,
                aggregation_mode=aggregation_mode,
                time_range=time_range,
                feature_label_params=feature_label_params,
                target_params_batch=target_params_batch,
                resample_params=resample_params,
                seq_params=seq_params,
                overwrite_cache=args.overwrite_cache
            )
            
            print("\nSuccessfully cached ML data.")
            
            # Load a sample to show details
            print("\nLoading sample of cached data to verify...")
            ml_data = load_cached_ml_data(
                dataset_mode=dataset_mode,
                export_mode=export_mode,
                aggregation_mode=aggregation_mode,
                time_range=time_range,
                feature_label_params=feature_label_params,
                target_params_batch=target_params_batch,
                resample_params=resample_params,
                seq_params=seq_params
            )
            
            if ml_data is not None and not ml_data.empty:
                print(f"Successfully loaded ML data with {len(ml_data)} rows and {len(ml_data.columns)} columns")
                print(f"Data sample spans from {ml_data.index.min()} to {ml_data.index.max()}")
                print(f"Number of unique symbols: {ml_data['symbol'].nunique()}")
                print(f"Features included: {len(ml_data.columns) - 1} columns")  # -1 for symbol column
            else:
                print("Failed to load cached ML data. Please check the logs for details.")
                
        except Exception as e:
            print(f"\nError caching ML data: {e}")

if __name__ == "__main__":     
    main()


