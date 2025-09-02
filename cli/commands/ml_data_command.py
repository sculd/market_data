import datetime
import multiprocessing
from argparse import ArgumentParser, Namespace
from functools import partial

import market_data.ingest.missing_data_finder
import market_data.util.cache.parallel_processing
import market_data.util.cache.time
from cli.base import BaseCommand, handle_common_errors
from market_data.feature.label import FeatureLabel, FeatureLabelCollection
from market_data.feature.param import SequentialFeatureParam
from market_data.feature.registry import list_registered_features
from market_data.machine_learning.ml_data.cache import calculate_and_cache_ml_data, load_cached_ml_data
from market_data.machine_learning.ml_data.param import MlDataParam
from market_data.machine_learning.resample import get_resample_params_class, list_registered_resample_methods
from market_data.target.calc import TargetParams, TargetParamsBatch


class MLDataCommand(BaseCommand):
    name = "ml-data"
    help = "Manage ML data operations"
    
    def add_arguments(self, parser: ArgumentParser):
        subparsers = parser.add_subparsers(dest='action', required=True, help='ML data operations')
        
        # Check command  
        check_parser = subparsers.add_parser('check', help='Check missing ML data')
        self._add_ml_args(check_parser)
        
        # Cache command
        cache_parser = subparsers.add_parser('cache', help='Cache ML data')
        self._add_ml_args(cache_parser)
    
    def _add_ml_args(self, parser: ArgumentParser):
        """Add common ML data arguments"""
        self.add_common_args(parser)
        
        parser.add_argument('--features', type=str, default='all',
                           help='Features to process: "all", "forex", "crypto", "stock", "none", or comma-separated list')
        parser.add_argument('--sequential', action='store_true',
                           help='Use sequential features instead of regular features')
        parser.add_argument('--sequence-window', type=int, default=30,
                           help='Size of the sliding window for sequential features')
        parser.add_argument('--forward-periods', type=str,
                           help='Comma-separated list of forward periods (e.g., "1,2,3")')
        parser.add_argument('--tps', type=str,
                           help='Comma-separated list of target price shifts (e.g., "0.001,0.002,0.003")')
        
        # Resample parameters
        available_methods = list_registered_resample_methods()
        parser.add_argument('--resample-type-label', type=str, default='cumsum',
                           choices=available_methods,
                           help=f'Resampling method to use. Available: {", ".join(available_methods)}')
        parser.add_argument('--resample-params', type=str, default=None,
                           help='Resampling parameters. Format depends on method: '
                                'cumsum: "price_col,threshold" (e.g., "close,0.07") '
                                'reversal: "price_col,threshold,threshold_reversal" (e.g., "close,0.1,0.03")')
    
    @handle_common_errors
    def handle(self, args: Namespace) -> int:
        self.setup_logging(getattr(args, 'verbose', False))
        
        # Validate forward_periods and tps are specified together
        if (args.forward_periods is None) != (args.tps is None):
            print("❌ Error: --forward-periods and --tps must be specified together")
            return 1
        
        if args.action == 'check':
            return self._handle_check(args)
        elif args.action == 'cache':
            return self._handle_cache(args)
        
        return 1
    
    def _create_ml_params(self, args: Namespace):
        """Create ML data parameters from arguments"""
        # Features
        features_to_process = []
        if args.features == "all":
            features_to_process = list_registered_features()
        elif args.features == "forex":
            features_to_process = list_registered_features(security_type="forex")
        elif args.features == "crypto":
            features_to_process = list_registered_features(security_type="crypto")
        elif args.features == "stock":
            features_to_process = list_registered_features(security_type="stock")
        elif args.features == "none":
            features_to_process = []
        else:
            features_to_process = [f.strip() for f in args.features.split(",")]
        
        # Targets
        target_params_batch = TargetParamsBatch()
        if args.forward_periods and args.tps:
            target_params_batch = TargetParamsBatch(
                target_params_list=[
                    TargetParams(forward_period=int(period), tp_value=float(tp), sl_value=float(tp)) 
                    for period in args.forward_periods.split(',') 
                    for tp in args.tps.split(',')
                ]
            )
        
        # Resample parameters
        resample_params_class = get_resample_params_class(args.resample_type_label)
        resample_params = resample_params_class.from_str(args.resample_params)
        
        # Sequential parameters
        seq_param = None
        if args.sequential:
            seq_param = SequentialFeatureParam(sequence_window=args.sequence_window)
        
        return features_to_process, target_params_batch, resample_params, seq_param
    
    def _handle_check(self, args: Namespace) -> int:
        """Handle the check command"""
        self.print_processing_info(args)
        print(f"  Action: check")
        print(f"  Features: {args.features}")
        if args.forward_periods and args.tps:
            print(f"  Forward Periods: {args.forward_periods}")
            print(f"  Target Price Shifts: {args.tps}")
        print(f"  Resample Type: {args.resample_type_label}")
        if args.resample_params:
            print(f"  Resample Params: {args.resample_params}")
        print(f"  Sequential: {args.sequential}")
        if args.sequential:
            print(f"  Sequence Window: {args.sequence_window}")
        
        cache_context = self.create_cache_context(args)
        time_range = self.create_time_range(args)
        feature_label_params, target_params_batch, resample_params, seq_param = self._create_ml_params(args)
        
        print("\nChecking ml_data")
        missing_ranges = market_data.ingest.missing_data_finder.check_missing_ml_data(
            cache_context=cache_context,
            time_range=time_range,
            feature_label_params=feature_label_params,
            target_params_batch=target_params_batch,
            resample_params=resample_params,
            seq_param=seq_param
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
            suggest_cmd = f"python cli.py ml-data cache --features {args.features}"
            if args.resample_type_label != 'cumsum':
                suggest_cmd += f" --resample-type-label {args.resample_type_label}"
            if args.resample_params:
                suggest_cmd += f" --resample-params {args.resample_params}"
            if args.forward_periods and args.tps:
                suggest_cmd += f" --forward-periods {args.forward_periods} --tps {args.tps}"
            if args.sequential:
                suggest_cmd += " --sequential"
                if args.sequence_window != 30:
                    suggest_cmd += f" --sequence-window {args.sequence_window}"
            suggest_cmd += f" --from {args.date_from} --to {args.date_to}"
            
            print(f"\n  To cache ML data, run:")
            print(f"    {suggest_cmd}")
        
        return 0
    
    def _handle_cache(self, args: Namespace) -> int:
        """Handle the cache command"""
        self.print_processing_info(args)
        print(f"  Action: cache")
        print(f"  Features: {args.features}")
        if args.forward_periods and args.tps:
            print(f"  Forward Periods: {args.forward_periods}")
            print(f"  Target Price Shifts: {args.tps}")
        print(f"  Resample Type: {args.resample_type_label}")
        if args.resample_params:
            print(f"  Resample Params: {args.resample_params}")
        print(f"  Sequential: {args.sequential}")
        if args.sequential:
            print(f"  Sequence Window: {args.sequence_window}")
        
        cache_context = self.create_cache_context(args)
        time_range = self.create_time_range(args)
        feature_label_params, target_params_batch, resample_params, seq_param = self._create_ml_params(args)
            
        # Create FeatureLabelCollection from feature labels
        feature_collection = FeatureLabelCollection()
        for feature in feature_label_params:
            feature_label = FeatureLabel(feature)
            feature_collection = feature_collection.with_feature_label(feature_label)

        ml_params = MlDataParam(
            feature_collection=feature_collection,
            target_params_batch=target_params_batch,
            resample_params=resample_params,
            seq_param=seq_param
        )

        print("\nCalculating and caching ML data...")
        
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
                
                print(f"  Using parallel processing with {workers} workers")

                cache_func = partial(
                    calculate_and_cache_ml_data,
                    cache_context=cache_context,
                    ml_params=ml_params,
                    overwrite_cache=args.overwrite_cache,
                )

                time_ranges = [market_data.util.time.TimeRange(t_from=t_from, t_to=t_to) 
                              for t_from, t_to in calculation_ranges]
                market_data.util.cache.parallel_processing.cache_multiprocess(
                    cache_func=cache_func,
                    time_ranges=time_ranges,
                    workers=workers
                )
                
            else:
                # Sequential processing
                for i, calc_range in enumerate(calculation_ranges):
                    calc_t_from, calc_t_to = calc_range
                    print(f"  Processing batch {i+1}/{len(calculation_ranges)}: {calc_t_from.date()} to {calc_t_to.date()}")
                    
                    calc_time_range = market_data.util.time.TimeRange(calc_t_from, calc_t_to)
                    
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
                print(f"Successfully loaded ML data with {len(ml_data)} rows and {len(ml_data.columns)} columns")
                print(f"Data sample spans from {ml_data.index.min()} to {ml_data.index.max()}")
                print(f"Number of unique symbols: {ml_data['symbol'].nunique()}")
                print(f"Features included: {len(ml_data.columns) - 1} columns")  # -1 for symbol column
            else:
                print("Failed to load cached ML data. Please check the logs for details.")
                return 1
                
            return 0
            
        except Exception as e:
            print(f"\nError caching ML data: {e}")
            return 1