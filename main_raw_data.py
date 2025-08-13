import argparse
import datetime
import logging

import market_data.ingest.common
import market_data.ingest.gcs.cache
import market_data.ingest.missing_data_finder
import market_data.util.cache.time
import setup_env  # needed for env variables
from market_data.ingest.common import (AGGREGATION_MODE, DATASET_MODE,
                                       EXPORT_MODE, CacheContext)
from market_data.util.time import TimeRange

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='Query and cache raw market data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Action argument
    parser.add_argument('--action', type=str, default='cache',
                        choices=['check', 'cache'],
                        help='Action to perform: check missing data or cache data')
    
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
    parser.add_argument('--from', dest='date_from', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    
    parser.add_argument('--to', dest='date_to', type=str, required=True,
                        help='End date in YYYY-MM-DD format')
    
    # Optional arguments
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite existing cache files')
    
    parser.add_argument('--skip_first_day', action='store_true',
                        help='Skip the first day in the range')
    
    args = parser.parse_args()
    
    # Get enum values by name
    dataset_mode = getattr(DATASET_MODE, args.dataset_mode)
    export_mode = getattr(EXPORT_MODE, args.export_mode)
    aggregation_mode = getattr(AGGREGATION_MODE, args.aggregation_mode)
    
    # Create cache context
    cache_context = CacheContext(dataset_mode, export_mode, aggregation_mode)
    
    # Create TimeRange object
    time_range = TimeRange(date_str_from=args.date_from, date_str_to=args.date_to)
    
    logger.info("Processing with parameters:")
    logger.info(f"  Action: {args.action}")
    logger.info(f"  Dataset Mode: {str(dataset_mode)}")
    logger.info(f"  Export Mode: {str(export_mode)}")
    logger.info(f"  Aggregation Mode: {str(aggregation_mode)}")
    logger.info(f"  Time Range: {args.date_from} to {args.date_to}")
    
    if args.action == 'check':
        # Check which date ranges are missing from the cache
        missing_ranges = market_data.ingest.missing_data_finder.check_missing_raw_data(
            cache_context=cache_context,
            time_range=time_range
        )
        
        if not missing_ranges:
            logger.info("All data for the specified range is present in the cache.")
        else:
            # Group consecutive dates
            grouped_ranges = market_data.util.cache.time.group_consecutive_dates(missing_ranges)
            
            total_missing_days = len(missing_ranges)
            logger.info(f"Missing data for {total_missing_days} day(s), grouped into {len(grouped_ranges)} range(s):")
            
            for i, (d_from, d_to) in enumerate(grouped_ranges):
                if d_from.date() == d_to.date() - datetime.timedelta(days=1):
                    # Single day range (common when using daily intervals)
                    logger.info(f"  {i+1}. {d_from.date()}")
                else:
                    # Multi-day range
                    logger.info(f"  {i+1}. {d_from.date()} to {(d_to.date() - datetime.timedelta(days=1))}")
                
            # Suggest command to cache the missing data
            suggest_cmd = f"python main_raw_data.py --action cache --dataset_mode {args.dataset_mode} --export_mode {args.export_mode} --aggregation_mode {args.aggregation_mode} --from {args.date_from} --to {args.date_to}"
            logger.info(f"To cache the missing data, run:")
            logger.info(f"  {suggest_cmd}")
    
    else:  # args.action == 'cache'
        logger.info(f"  Overwrite Cache: {args.overwrite_cache}")
        logger.info(f"  Skip First Day: {args.skip_first_day}")
        
        # Query and cache data
        market_data.ingest.gcs.cache.cache(
            cache_context=cache_context,
            time_range=time_range,
            overwrite_cache=args.overwrite_cache,
            skip_first_day=args.skip_first_day,
        )
        
        # Force garbage collection
        import gc
        gc.collect()

if __name__ == "__main__":
    #"""
    cache_context = CacheContext(DATASET_MODE.OKX, EXPORT_MODE.RAW, AGGREGATION_MODE.TAKE_LASTEST)
    market_data.ingest.gcs.cache.cache(
        cache_context,
        time_range=TimeRange(date_str_from="2025-07-01", date_str_to="2025-07-03"),
        overwrite_cache=False,
        skip_first_day=False,
    )
    #"""
    main()
