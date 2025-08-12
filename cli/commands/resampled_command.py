"""
Resampled data management command
"""
import datetime
import multiprocessing
from argparse import ArgumentParser, Namespace
from functools import partial

import market_data.ingest.missing_data_finder
import market_data.util.cache.parallel_processing
import market_data.util.cache.time
from cli.base import BaseCommand, handle_common_errors
from market_data.machine_learning.resample import (
    get_resample_function, get_resample_params_class,
    list_registered_resample_methods)
from market_data.machine_learning.resample.cache import \
    calculate_and_cache_resampled


class ResampledCommand(BaseCommand):
    name = "resampled"
    help = "Manage resampled data operations"
    
    def add_arguments(self, parser: ArgumentParser):
        subparsers = parser.add_subparsers(dest='action', required=True, help='Resampled data operations')
        
        # Check command  
        check_parser = subparsers.add_parser('check', help='Check missing resampled data')
        self._add_resample_args(check_parser)
        
        # Cache command
        cache_parser = subparsers.add_parser('cache', help='Cache resampled data')
        self._add_resample_args(cache_parser)
        cache_parser.add_argument('--calculation-batch-days', type=int, default=1,
                                 help='Number of days to calculate resampled data for in each batch')
    
    def _add_resample_args(self, parser: ArgumentParser):
        """Add common resampling arguments"""
        self.add_common_args(parser)
        
        # Resample type and parameters
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
        
        if args.action == 'check':
            return self._handle_check(args)
        elif args.action == 'cache':
            return self._handle_cache(args)
        
        return 1
    
    def _create_resample_params(self, args: Namespace):
        """Create resample parameters from arguments"""
        resample_params_class = get_resample_params_class(args.resample_type_label)
        resample_function = get_resample_function(args.resample_type_label)
        
        if resample_params_class is None:
            raise ValueError(f"Unknown resample type '{args.resample_type_label}'. Available: {', '.join(list_registered_resample_methods())}")
        
        if resample_function is None:
            raise ValueError(f"No function registered for resample type '{args.resample_type_label}'")
        
        resample_params = None
        if args.resample_params:
            resample_params = resample_params_class.from_str(args.resample_params)
        
        return resample_params, resample_function
    
    def _handle_check(self, args: Namespace) -> int:
        """Handle the check command"""
        self.print_processing_info(args)
        print(f"  Action: check")
        print(f"  Resample Type: {args.resample_type_label}")
        if args.resample_params:
            print(f"  Resample Params: {args.resample_params}")
        
        cache_context = self.create_cache_context(args)
        time_range = self.create_time_range(args)
        
        try:
            resample_params, _ = self._create_resample_params(args)
        except ValueError as e:
            print(f"❌ Error: {e}")
            return 1
        
        print("\nChecking resampled data")
        missing_ranges = market_data.ingest.missing_data_finder.check_missing_resampled_data(
            cache_context=cache_context,
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
                    # Single day range
                    print(f"    {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    print(f"    {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
            
            # Suggest command to cache resampled data
            suggest_cmd = f"python cli.py resampled cache"
            if args.resample_type_label != 'cumsum':
                suggest_cmd += f" --resample-type-label {args.resample_type_label}"
            if args.resample_params:
                suggest_cmd += f" --resample-params {args.resample_params}"
            suggest_cmd += f" --from {args.date_from} --to {args.date_to}"
            
            print(f"\n  To cache resampled data, run:")
            print(f"    {suggest_cmd}")
        
        return 0
    
    def _handle_cache(self, args: Namespace) -> int:
        """Handle the cache command"""
        self.print_processing_info(args)
        print(f"  Action: cache")
        print(f"  Resample Type: {args.resample_type_label}")
        if args.resample_params:
            print(f"  Resample Params: {args.resample_params}")
        print(f"  Calculation Batch Days: {args.calculation_batch_days}")
        
        cache_context = self.create_cache_context(args)
        time_range = self.create_time_range(args)
        
        try:
            resample_params, resample_function = self._create_resample_params(args)
        except ValueError as e:
            print(f"❌ Error: {e}")
            return 1
        
        print("\nCaching resampled data")
        try:
            # Set up calculation parameters
            calculation_batch_days = args.calculation_batch_days
            if calculation_batch_days <= 0:
                calculation_batch_days = 1
            
            missing_range_finder_func = partial(
                market_data.ingest.missing_data_finder.check_missing_resampled_data,
                cache_context=cache_context,
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
                    cache_context=cache_context,
                    resample_at_events_func=resample_function,
                    params=resample_params,
                    calculation_batch_days=1,  # Process each range as single batch
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
                    
                    calculate_and_cache_resampled(
                        cache_context=cache_context,
                        resample_at_events_func=resample_function,
                        params=resample_params,
                        time_range=calc_time_range,
                        calculation_batch_days=1,  # Process each range as single batch
                        overwrite_cache=args.overwrite_cache
                    )
            
            print("  Successfully cached resampled data")
            return 0
            
        except Exception as e:
            print(f"  Failed to cache resampled data: {e}")
            return 1