"""
Target data management command
"""
import datetime
import multiprocessing
from argparse import ArgumentParser, Namespace
from functools import partial

import market_data.ingest.missing_data_finder
import market_data.util.cache.parallel_processing
import market_data.util.cache.time
from cli.base import BaseCommand, handle_common_errors
from market_data.target.cache import calculate_and_cache_targets
from market_data.target.calc import TargetParams, TargetParamsBatch


class TargetCommand(BaseCommand):
    name = "target"
    help = "Manage target data operations"
    
    def add_arguments(self, parser: ArgumentParser):
        subparsers = parser.add_subparsers(dest='action', required=True, help='Target operations')
        
        # Check command  
        check_parser = subparsers.add_parser('check', help='Check missing target data')
        self.add_common_args(check_parser)
        check_parser.add_argument('--forward-periods', type=str,
                                 help='Comma-separated list of forward periods (e.g., "1,2,3")')
        check_parser.add_argument('--tps', type=str,
                                 help='Comma-separated list of target price shifts (e.g., "0.001,0.002,0.003")')
        
        # Cache command
        cache_parser = subparsers.add_parser('cache', help='Cache target data')
        self.add_common_args(cache_parser)
        cache_parser.add_argument('--calculation-batch-days', type=int, default=1,
                                 help='Number of days to calculate targets for in each batch')
        cache_parser.add_argument('--forward-periods', type=str,
                                 help='Comma-separated list of forward periods (e.g., "1,2,3")')
        cache_parser.add_argument('--tps', type=str,
                                 help='Comma-separated list of target price shifts (e.g., "0.001,0.002,0.003")')
    
    @handle_common_errors
    def handle(self, args: Namespace) -> int:
        self.setup_logging(getattr(args, 'verbose', False))
        
        # Validate forward_periods and tps are specified together
        if hasattr(args, 'forward_periods') and hasattr(args, 'tps'):
            if (args.forward_periods is None) != (args.tps is None):
                print("❌ Error: --forward-periods and --tps must be specified together")
                return 1
        
        if args.action == 'check':
            return self._handle_check(args)
        elif args.action == 'cache':
            return self._handle_cache(args)
        
        return 1
    
    def _create_target_params(self, args: Namespace) -> TargetParamsBatch:
        """Create target parameters from arguments"""
        if hasattr(args, 'forward_periods') and args.forward_periods and args.tps:
            return TargetParamsBatch(
                target_params_list=[
                    TargetParams(forward_period=int(period), tp_value=float(tp), sl_value=float(tp)) 
                    for period in args.forward_periods.split(',') 
                    for tp in args.tps.split(',')
                ]
            )
        return TargetParamsBatch()  # Use default parameters
    
    def _handle_check(self, args: Namespace) -> int:
        """Handle the check command"""
        self.print_processing_info(args)
        print(f"  Action: check")
        if hasattr(args, 'forward_periods') and args.forward_periods and args.tps:
            print(f"  Forward Periods: {args.forward_periods}")
            print(f"  Target Price Shifts: {args.tps}")
        
        cache_context = self.create_cache_context(args)
        time_range = self.create_time_range(args)
        target_params = self._create_target_params(args)
        
        print("\nChecking target data")
        missing_ranges = market_data.ingest.missing_data_finder.check_missing_target_data(
            cache_context=cache_context,
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
                    # Single day range
                    print(f"    {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    print(f"    {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
            
            # Suggest command to cache target data
            suggest_cmd = f"python cli.py target cache --from {args.date_from} --to {args.date_to}"
            if hasattr(args, 'forward_periods') and args.forward_periods and args.tps:
                suggest_cmd += f" --forward-periods {args.forward_periods} --tps {args.tps}"
            print(f"\n  To cache target data, run:")
            print(f"    {suggest_cmd}")
        
        return 0
    
    def _handle_cache(self, args: Namespace) -> int:
        """Handle the cache command"""
        self.print_processing_info(args)
        print(f"  Action: cache")
        print(f"  Calculation Batch Days: {args.calculation_batch_days}")
        if hasattr(args, 'forward_periods') and args.forward_periods and args.tps:
            print(f"  Forward Periods: {args.forward_periods}")
            print(f"  Target Price Shifts: {args.tps}")
        
        cache_context = self.create_cache_context(args)
        time_range = self.create_time_range(args)
        target_params = self._create_target_params(args)
        
        print("\nCaching target data")
        try:
            missing_range_finder_func = partial(
                market_data.ingest.missing_data_finder.check_missing_target_data,
                cache_context=cache_context,
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
                    cache_context=cache_context,
                    params=target_params,
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
                    
                    calculate_and_cache_targets(
                        cache_context=cache_context,
                        time_range=calc_time_range,
                        params=target_params,
                        calculation_batch_days=args.calculation_batch_days,
                        overwrite_cache=args.overwrite_cache,
                    )

            print("  Successfully cached target data")
            return 0
            
        except Exception as e:
            print(f"  Failed to cache target data: {e}")
            return 1