import datetime
from argparse import ArgumentParser, Namespace

import market_data.ingest.gcs.cache
import market_data.ingest.missing_data_finder
import market_data.util.cache.time
from cli.base import BaseCommand, handle_common_errors


class RawCommand(BaseCommand):
    name = "raw"
    help = "Manage raw market data operations"
    
    def add_arguments(self, parser: ArgumentParser):
        subparsers = parser.add_subparsers(dest='action', required=True, help='Raw data operations')
        
        # Check command  
        check_parser = subparsers.add_parser('check', help='Check missing raw data')
        self.add_common_args(check_parser)
        
        # Cache command
        cache_parser = subparsers.add_parser('cache', help='Cache raw data')
        self.add_common_args(cache_parser)
        cache_parser.add_argument('--skip-first-day', action='store_true',
                                 help='Skip the first day in the range')
    
    @handle_common_errors
    def handle(self, args: Namespace) -> int:
        self.setup_logging(getattr(args, 'verbose', False))
        
        if args.action == 'check':
            return self._handle_check(args)
        elif args.action == 'cache':
            return self._handle_cache(args)
        
        return 1
    
    def _handle_check(self, args: Namespace) -> int:
        """Handle the check command"""
        self.print_processing_info(args)
        print(f"  Action: check")
        
        cache_context = self.create_cache_context(args)
        time_range = self.create_time_range(args)
        
        print("\nChecking raw data")
        # Check which date ranges are missing from the cache
        missing_ranges = market_data.ingest.missing_data_finder.check_missing_raw_data(
            cache_context=cache_context,
            time_range=time_range
        )
        
        if not missing_ranges:
            print("All data for the specified range is present in the cache.")
        else:
            # Group consecutive dates
            grouped_ranges = market_data.util.cache.time.group_consecutive_dates(missing_ranges)
            
            total_missing_days = len(missing_ranges)
            print(f"Missing data for {total_missing_days} day(s), grouped into {len(grouped_ranges)} range(s):")
            
            for i, (d_from, d_to) in enumerate(grouped_ranges):
                if d_from.date() == d_to.date() - datetime.timedelta(days=1):
                    # Single day range
                    print(f"  {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    print(f"  {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
                
            # Suggest command to cache the missing data
            suggest_cmd = f"python cli.py raw cache --dataset-mode {getattr(args, 'dataset_mode', 'OKX')} --export-mode {getattr(args, 'export_mode', 'BY_MINUTE')} --aggregation-mode {getattr(args, 'aggregation_mode', 'TAKE_LATEST')} --from {args.date_from} --to {args.date_to}"
            print(f"\nTo cache the missing data, run:")
            print(f"  {suggest_cmd}")
        
        return 0
    
    def _handle_cache(self, args: Namespace) -> int:
        """Handle the cache command"""
        self.print_processing_info(args)
        print(f"  Action: cache")
        print(f"  Overwrite Cache: {args.overwrite_cache}")
        if hasattr(args, 'skip_first_day'):
            print(f"  Skip First Day: {args.skip_first_day}")
        
        # Get enum values
        from market_data.ingest.common import (DATASET_MODE, EXPORT_MODE,
                                               CacheContext)
        dataset_mode = getattr(DATASET_MODE, getattr(args, 'dataset_mode', 'OKX'))
        export_mode = getattr(EXPORT_MODE, getattr(args, 'export_mode', 'BY_MINUTE'))
        time_range = self.create_time_range(args)
        
        print("\nQuerying and caching raw data...")
        
        try:
            # Query and cache data
            result_df = market_data.ingest.gcs.cache.cache(
                CacheContext(dataset_mode, export_mode),
                time_range=time_range,
                overwrite_cache=args.overwrite_cache,
                skip_first_day=getattr(args, 'skip_first_day', False),
            )
            
            # Print summary of results
            if result_df is not None and not result_df.empty:
                print(f"\nSuccessfully cached {len(result_df)} rows of data")
                print(f"Data spans from {result_df.index.min()} to {result_df.index.max()}")
                print(f"Columns: {result_df.columns.tolist()}")
                print(f"Number of unique symbols: {result_df['symbol'].nunique()}")
            else:
                print("\nNo data was returned or cached.")
                return 1
            
            return 0
            
        except Exception as e:
            print(f"\nFailed to cache raw data: {e}")
            return 1