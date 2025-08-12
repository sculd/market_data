"""
Feature data management command
"""
import datetime
import multiprocessing
from argparse import ArgumentParser, Namespace
from functools import partial

import market_data.ingest.missing_data_finder
import market_data.util.cache.parallel_processing
import market_data.util.cache.time
from cli.base import BaseCommand, handle_common_errors
from market_data.feature.cache_writer import cache_feature_cache
from market_data.feature.label import FeatureLabel
from market_data.feature.registry import list_registered_features


class FeatureCommand(BaseCommand):
    name = "feature"
    help = "Manage feature data operations"
    
    def add_arguments(self, parser: ArgumentParser):
        subparsers = parser.add_subparsers(dest='action', required=True, help='Feature operations')
        
        # List command
        list_parser = subparsers.add_parser('list', help='List available features')
        list_parser.add_argument('--security-type', type=str, default='all',
                                help='Security type to filter: all, forex, crypto, stock')
        
        # Check command  
        check_parser = subparsers.add_parser('check', help='Check missing feature data')
        self.add_common_args(check_parser)
        check_parser.add_argument('--feature', type=str, required=True,
                                 help='Specific feature label to check')
        
        # Cache command
        cache_parser = subparsers.add_parser('cache', help='Cache feature data')
        self.add_common_args(cache_parser)
        cache_parser.add_argument('--feature', type=str, default='all',
                                 help='Feature to process: "all", "forex", "crypto", "stock", or specific feature label')
        cache_parser.add_argument('--calculation-batch-days', type=int, default=1,
                                 help='Number of days to calculate features for in each batch')
        cache_parser.add_argument('--warmup-days', type=int, default=None,
                                 help='Warm up days. Auto-detection if not provided.')
    
    @handle_common_errors
    def handle(self, args: Namespace) -> int:
        self.setup_logging(getattr(args, 'verbose', False))
        
        if args.action == 'list':
            return self._handle_list(args)
        elif args.action == 'check':
            return self._handle_check(args)
        elif args.action == 'cache':
            return self._handle_cache(args)
        
        return 1
    
    def _handle_list(self, args: Namespace) -> int:
        """Handle the list command"""
        security_type = getattr(args, 'security_type', 'all')
        features = list_registered_features(security_type=security_type)
        
        type_label = args.security_type if args.security_type != 'all' else 'all'
        print(f"\nAvailable {type_label} features ({len(features)}):")
        for i, feature in enumerate(sorted(features)):
            print(f"  {i+1}. {feature}")
        
        return 0
    
    def _handle_check(self, args: Namespace) -> int:
        """Handle the check command"""
        self.print_processing_info(args)
        print(f"  Action: check")
        print(f"  Feature: {args.feature}")
        
        cache_context = self.create_cache_context(args)
        time_range = self.create_time_range(args)
        
        print(f"\nChecking feature: {args.feature}")
        feature_label_obj = FeatureLabel(args.feature, None)
        missing_ranges = market_data.ingest.missing_data_finder.check_missing_feature_data(
            cache_context=cache_context,
            feature_label=feature_label_obj,
            time_range=time_range
        )
        
        if not missing_ranges:
            print(f"  All data for '{args.feature}' is present in the cache.")
        else:
            # Group consecutive dates
            grouped_ranges = market_data.util.cache.time.group_consecutive_dates(missing_ranges)
            
            total_missing_days = len(missing_ranges)
            print(f"  Missing data for '{args.feature}': {total_missing_days} day(s), grouped into {len(grouped_ranges)} range(s):")
            
            for i, (d_from, d_to) in enumerate(grouped_ranges):
                if d_from.date() == d_to.date() - datetime.timedelta(days=1):
                    # Single day range
                    print(f"    {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    print(f"    {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
            
            # Suggest command to cache this feature
            suggest_cmd = f"python cli.py feature cache --feature {args.feature} --from {args.date_from} --to {args.date_to}"
            print(f"\n  To cache this feature, run:")
            print(f"    {suggest_cmd}")
        
        return 0
    
    def _handle_cache(self, args: Namespace) -> int:
        """Handle the cache command"""
        self.print_processing_info(args)
        print(f"  Action: cache")
        print(f"  Feature: {args.feature}")
        print(f"  Calculation Batch Days: {args.calculation_batch_days}")
        if args.warmup_days:
            print(f"  Warmup Days: {args.warmup_days}")
        
        cache_context = self.create_cache_context(args)
        time_range = self.create_time_range(args)
        
        # Determine features to process
        features_to_process = []
        if args.feature in ["all", "forex", "crypto", "stock"]:
            security_type = args.feature if args.feature != 'all' else None
            features_to_process = list_registered_features(security_type=security_type)
            print(f"\nProcessing all {len(features_to_process)} registered {args.feature} features")
        else:
            features_to_process = [args.feature]
        
        # Process each feature
        successful_features = []
        failed_features = []
        
        for feature_label in features_to_process:
            print(f"\nCaching feature: {feature_label}")
            feature_label_obj = FeatureLabel(feature_label)
            
            try:
                # Set up calculation parameters
                calculation_batch_days = args.calculation_batch_days
                if calculation_batch_days <= 0:
                    calculation_batch_days = 1

                missing_range_finder_func = partial(
                    market_data.ingest.missing_data_finder.check_missing_feature_data,
                    cache_context=cache_context,
                    feature_label=feature_label_obj,
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
                        feature_label_obj=feature_label_obj,
                        cache_context=cache_context,
                        calculation_batch_days=1,  # Process each range as single batch
                        warm_up_days=args.warmup_days,
                        overwrite_cache=args.overwrite_cache,
                    )

                    time_ranges = [market_data.util.time.TimeRange(t_from=t_from, t_to=t_to) 
                                  for t_from, t_to in calculation_ranges]
                    successful_batches, failed_batches = market_data.util.cache.parallel_processing.cache_multiprocess(
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
                        
                        calc_time_range = market_data.util.time.TimeRange(calc_t_from, calc_t_to)
                        
                        success = cache_feature_cache(
                            feature_label_obj=feature_label_obj,
                            cache_context=cache_context,
                            time_range=calc_time_range,
                            calculation_batch_days=1,  # Process each range as single batch
                            warm_up_days=args.warmup_days,
                            overwrite_cache=args.overwrite_cache
                        )
                        
                        if not success:
                            feature_success = False
                            print(f"  Failed to process batch {i+1} for feature: {feature_label}")
                            break
                
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
            return 1
        
        return 0