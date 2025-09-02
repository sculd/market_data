import argparse
import datetime
import logging
import multiprocessing
from functools import partial

import setup_env  # needed for env variables

import market_data.ingest.missing_data_finder
import market_data.util.cache.parallel_processing
import market_data.util.cache.time
from market_data.feature.label import FeatureLabel, FeatureLabelCollection
from market_data.feature.param import SequentialFeatureParam
from market_data.feature.registry import list_registered_features
from market_data.ingest.common import AGGREGATION_MODE, DATASET_MODE, EXPORT_MODE, CacheContext
from market_data.machine_learning.ml_data.param import MlDataParam
from market_data.machine_learning.ml_data.cache import calculate_and_cache_ml_data, load_cached_ml_data
from market_data.machine_learning.resample import get_resample_params_class, list_registered_resample_methods
from market_data.target.param import TargetParams, TargetParamsBatch
from market_data.util.time import TimeRange

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    parser.add_argument('--aggregation_mode', type=str, default='TAKE_LATEST',
                        choices=[mode.name for mode in AGGREGATION_MODE],
                        help='Aggregation mode')
    
    # Time range arguments
    parser.add_argument('--from', dest='date_from', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    
    parser.add_argument('--to', dest='date_to', type=str, required=True,
                        help='End date in YYYY-MM-DD format')
    
    parser.add_argument('--features', type=str, default='all',
                        help='Features to process: "all", "forex", "crypto", "stock", "none", or comma-separated list')
    
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
    
    # Create cache context
    cache_context = CacheContext(dataset_mode, export_mode, aggregation_mode)
    
    # Create TimeRange object
    time_range = TimeRange(date_str_from=args.date_from, date_str_to=args.date_to)
    
    # Get resample method components from registry
    resample_params_class = get_resample_params_class(args.resample_type_label)
    
    if resample_params_class is None:
        logger.error(f"Unknown resample type '{args.resample_type_label}'")
        logger.error(f"Available methods: {', '.join(list_registered_resample_methods())}")
        return 1
    
    # Create parameter objects
    features_to_process = []
    if args.features == "all":
        features_to_process = list_registered_features()
        logger.info(f"Processing all {len(features_to_process)} registered features")
    elif args.features == "forex":
        features_to_process = list_registered_features(security_type="forex")
        logger.info(f"Processing all {len(features_to_process)} registered forex features")
    elif args.features == "crypto":
        features_to_process = list_registered_features(security_type="crypto")
        logger.info(f"Processing all {len(features_to_process)} registered crypto features")
    elif args.features == "stock":
        features_to_process = list_registered_features(security_type="stock")
        logger.info(f"Processing all {len(features_to_process)} registered stock features")
    elif args.features == "none":
        features_to_process = []
        logger.info(f"Processing none {len(features_to_process)} registered features")
    else:
        features_to_process = [f.strip() for f in args.features.split(",")]
    
    # Create FeatureLabelCollection from feature labels
    feature_collection = FeatureLabelCollection()
    for feature in features_to_process:
        feature_label = FeatureLabel(feature)
        feature_collection = feature_collection.with_feature_label(feature_label)

    target_params_batch = TargetParamsBatch()
    if args.forward_periods and args.tps:
        target_params_batch = TargetParamsBatch(
            target_params_list=[TargetParams(forward_period=int(period), tp_value=float(tp), sl_value=float(tp)) 
                for period in args.forward_periods.split(',') 
                for tp in args.tps.split(',')]
        )
    
    # Parse resample parameters
    resample_params = resample_params_class.from_str(args.resample_params)
    
    # Create sequential parameters if needed
    seq_param = None
    if args.sequential:
        seq_param = SequentialFeatureParam(sequence_window=args.sequence_window)

    ml_params = MlDataParam(
        feature_collection=feature_collection,
        target_params_batch=target_params_batch,
        resample_params=resample_params,
        seq_param=seq_param
    )

    logger.info("Processing with parameters:")
    logger.info(f"  Action: {args.action}")
    logger.info(f"  Dataset Mode: {str(dataset_mode)}")
    logger.info(f"  Export Mode: {str(export_mode)}")
    logger.info(f"  Aggregation Mode: {str(aggregation_mode)}")
    logger.info(f"  Time Range: {args.date_from} to {args.date_to}")
    logger.info(f"  Features: {args.features if args.features else 'All registered features'}")
    if args.forward_periods and args.tps:
        logger.info(f"  Forward Periods: {args.forward_periods}")
        logger.info(f"  Target Price Shifts: {args.tps}")
    logger.info(f"  Resample Type: {args.resample_type_label}")
    if resample_params:
        # Create a display string for the parameters
        param_display = []
        for field_name, field_value in resample_params.__dict__.items():
            param_display.append(f"{field_name}={field_value}")
        logger.info(f"  Resample Params: {', '.join(param_display)}")
    logger.info(f"  Sequential: {args.sequential}")
    if args.sequential:
        logger.info(f"  Sequence Window: {args.sequence_window}")
    
    if args.action == 'check':
        logger.info("Checking ml_data")
        # Check which date ranges are missing from the ML data cache
        missing_ranges = market_data.ingest.missing_data_finder.check_missing_ml_data(
            cache_context=cache_context,
            time_range=time_range,
            feature_collection=feature_collection,
            target_params_batch=target_params_batch,
            resample_params=resample_params,
            seq_param=seq_param
        )
        
        if not missing_ranges:
            logger.info("All ml_data is present in the cache.")
        else:
            # Group consecutive dates
            grouped_ranges = market_data.util.cache.time.group_consecutive_dates(missing_ranges)
            
            total_missing_days = len(missing_ranges)
            logger.info(f"Missing ML data: {total_missing_days} day(s), grouped into {len(grouped_ranges)} range(s):")
            
            for i, (d_from, d_to) in enumerate(grouped_ranges):
                if d_from.date() == d_to.date() - datetime.timedelta(days=1):
                    # Single day range
                    logger.info(f"  {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    logger.info(f"  {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
            
            # Suggest command to cache ML data
            resample_type_option = f" --resample_type_label {args.resample_type_label}" if args.resample_type_label != 'cumsum' else ""
            resample_param_option = f" --resample_params {args.resample_params}" if args.resample_params else ""
            target_options = ""
            if args.forward_periods and args.tps:
                target_options = f" --forward_periods {args.forward_periods} --tps {args.tps}"
            sequential_option = " --sequential" if args.sequential else ""
            
            suggest_cmd = f"python main_ml_data.py --action cache --features {args.features}{resample_type_option}{resample_param_option}{target_options}{sequential_option} --from {args.date_from} --to {args.date_to}"
            logger.info(f"To cache ML data, run:")
            logger.info(f"    {suggest_cmd}")
    
    elif args.action == 'cache':
        logger.info("Calculating and caching ML data...")
        
        try:
            # Set up calculation parameters
            calculation_batch_days = 1  # Use daily batches for ML data

            missing_range_finder_func = partial(
                market_data.ingest.missing_data_finder.check_missing_ml_data,
                cache_context=cache_context,
                feature_collection=feature_collection,
                target_params_batch=target_params_batch,
                resample_params=resample_params,
                seq_param=seq_param
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
                
                logger.info(f"  Using parallel processing with {workers} workers")

                cache_func = partial(
                    calculate_and_cache_ml_data,
                    cache_context=cache_context,
                    ml_params=ml_params,
                    overwrite_cache=args.overwrite_cache,
                )

                time_ranges = [TimeRange(t_from=t_from, t_to=t_to) for t_from, t_to in calculation_ranges]
                market_data.util.cache.parallel_processing.cache_multiprocess(
                    cache_func=cache_func,
                    time_ranges=time_ranges,
                    workers=workers
                )
                
            else:
                # Sequential processing (original behavior)
                for i, calc_range in enumerate(calculation_ranges):
                    calc_t_from, calc_t_to = calc_range
                    logger.info(f"  Processing batch {i+1}/{len(calculation_ranges)}: {calc_t_from.date()} to {calc_t_to.date()}")
                    
                    calc_time_range = TimeRange(calc_t_from, calc_t_to)
                    
                    calculate_and_cache_ml_data(
                        cache_context=cache_context,
                        time_range=calc_time_range,
                        ml_params=ml_params,
                        overwrite_cache=args.overwrite_cache
                    )
            
            print("\nSuccessfully cached ML data.")
            
            # Load a sample to show details
            print("\nLoading sample of cached data to verify...")
            ml_data = load_cached_ml_data(
                cache_context=cache_context,
                time_range=time_range,
                ml_params=ml_params,
            )
            
            if ml_data is not None and not ml_data.empty:
                logger.info(f"Successfully loaded ML data with {len(ml_data)} rows and {len(ml_data.columns)} columns")
                logger.info(f"Data sample spans from {ml_data.index.min()} to {ml_data.index.max()}")
                logger.info(f"Number of unique symbols: {ml_data['symbol'].nunique()}")
                logger.info(f"Features included: {len(ml_data.columns) - 1} columns")  # -1 for symbol column
            else:
                logger.error("Failed to load cached ML data. Please check the logs for details.")
                
        except Exception as e:
            logger.error(f"Error caching ML data: {e}")
            logger.error("--- traceback ---")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()


