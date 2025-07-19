import argparse
import datetime
import pandas as pd
import os
from pathlib import Path

import setup_env # needed for env variables

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.util.cache.time import split_t_range
from market_data.target.target import TargetParamsBatch, TargetParams
from market_data.machine_learning.resample import (
    get_resample_params_class,
    list_registered_resample_methods
)
from market_data.feature.impl.common import SequentialFeatureParam
from market_data.feature.util import parse_feature_label_params
from market_data.feature.registry import list_registered_features
from market_data.machine_learning.cache_ml_data import (
    calculate_and_cache_ml_data,
    load_cached_ml_data,
)
import market_data.util.cache.time
import market_data.util.cache.missing_data_finder

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
    
    parser.add_argument('--features', type=str, default='all',
                        help='Features to process: "all", "forex", "crypto", "stock", or comma-separated list')
    
    parser.add_argument('--sequential', action='store_true',
                        help='Use sequential features instead of regular features')
    
    parser.add_argument('--sequence_window', type=int, default=30,
                        help='Size of the sliding window for sequential features')

    # Target parameters
    parser.add_argument('--forward_periods', type=str,
                        help='Comma-separated list of forward periods (e.g., "1,2,3"). For forex, 10,30,60 are used')
    
    parser.add_argument('--tps', type=str,
                        help='Comma-separated list of target price shifts (e.g., "0.001,0.002,0.003"). For forex, 0.0025,0.005,0.01 are used')
    
    # Resample type and parameters
    available_methods = list_registered_resample_methods()
    parser.add_argument('--resample_type_label', type=str, default='cumsum',
                        choices=available_methods,
                        help=f'Resampling method to use. Available: {", ".join(available_methods)}')
    
    parser.add_argument('--resample_params', type=str, default=None,
                        help='Resampling parameters. Format depends on method: '
                             'cumsum: "price_col,threshold" (e.g., "close,0.07") '
                             'reversal: "price_col,threshold,threshold_reversal" (e.g., "close,0.1,0.03")')
                        
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
    
    # Get resample method components from registry
    resample_params_class = get_resample_params_class(args.resample_type_label)
    
    if resample_params_class is None:
        print(f"Error: Unknown resample type '{args.resample_type_label}'")
        print(f"Available methods: {', '.join(list_registered_resample_methods())}")
        return 1
    
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
    resample_params = resample_params_class.parse_resample_params(args.resample_params)
    
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
    print(f"  Resample Type: {args.resample_type_label}")
    if resample_params:
        # Create a display string for the parameters
        param_display = []
        for field_name, field_value in resample_params.__dict__.items():
            param_display.append(f"{field_name}={field_value}")
        print(f"  Resample Params: {', '.join(param_display)}")
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
            print("  All ml_data is present in the cache.")
        else:
            # Group consecutive dates
            grouped_ranges = market_data.util.cache.time.group_consecutive_dates(missing_ranges)
            
            total_missing_days = len(missing_ranges)
            print(f"\nMissing ML data: {total_missing_days} day(s), grouped into {len(grouped_ranges)} range(s):")
            
            for i, (d_from, d_to) in enumerate(grouped_ranges):
                if d_from.date() == d_to.date() - datetime.timedelta(days=1):
                    # Single day range
                    print(f"  {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    print(f"  {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
            
            # Suggest command to cache ML data
            resample_type_option = f" --resample_type_label {args.resample_type_label}" if args.resample_type_label != 'cumsum' else ""
            resample_param_option = f" --resample_params {args.resample_params}" if args.resample_params else ""
            target_options = ""
            if args.forward_periods and args.tps:
                target_options = f" --forward_periods {args.forward_periods} --tps {args.tps}"
            sequential_option = " --sequential" if args.sequential else ""
            
            suggest_cmd = f"python main_ml_data.py --action cache --features {args.features}{resample_type_option}{resample_param_option}{target_options}{sequential_option} --from {args.date_from} --to {args.date_to}"
            print(f"\n  To cache ML data, run:")
            print(f"    {suggest_cmd}")
    
    elif args.action == 'cache':
        print("\nCalculating and caching ML data...")
        
        try:
            # Set up calculation parameters
            calculation_batch_days = 1  # Use daily batches for ML data
            calculation_interval = datetime.timedelta(days=calculation_batch_days)
            
            # Determine which ranges need to be calculated
            if args.overwrite_cache:
                # If overwriting cache, process all ranges
                t_from, t_to = time_range.to_datetime()
                calculation_ranges = split_t_range(t_from, t_to, interval=calculation_interval)
                print(f"  Overwrite cache enabled - processing all {len(calculation_ranges)} ranges")
            else:
                # If not overwriting cache, only process missing ranges
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
                    print("  All ML data already cached - skipping calculation")
                    return
                
                # Group consecutive missing ranges and split into calculation batches
                grouped_ranges = market_data.util.cache.time.group_consecutive_dates(missing_ranges)
                calculation_ranges = []
                
                for grouped_start, grouped_end in grouped_ranges:
                    # Split each grouped range into calculation batches
                    batch_ranges = split_t_range(grouped_start, grouped_end, interval=calculation_interval)
                    calculation_ranges.extend(batch_ranges)
                
                print(f"  Found {len(missing_ranges)} missing days, grouped into {len(grouped_ranges)} ranges, "
                      f"split into {len(calculation_ranges)} calculation batches")
            
            # Process each calculation range
            for i, calc_range in enumerate(calculation_ranges):
                calc_t_from, calc_t_to = calc_range
                print(f"  Processing batch {i+1}/{len(calculation_ranges)}: {calc_t_from.date()} to {calc_t_to.date()}")
                
                calc_time_range = TimeRange(calc_t_from, calc_t_to)
                
                calculate_and_cache_ml_data(
                    dataset_mode=dataset_mode,
                    export_mode=export_mode,
                    aggregation_mode=aggregation_mode,
                    time_range=calc_time_range,
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


