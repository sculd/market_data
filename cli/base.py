import logging
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Optional

from market_data.ingest.common import (AGGREGATION_MODE, DATASET_MODE,
                                       EXPORT_MODE, CacheContext)
from market_data.util.time import TimeRange


class BaseCommand(ABC):
    """Base class for all CLI commands"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Command name (e.g., 'feature', 'ml-data')"""
        pass
    
    @property
    @abstractmethod
    def help(self) -> str:
        """Help description for the command"""
        pass
    
    def add_common_args(self, parser: ArgumentParser, require_dates: bool = True):
        """Add common arguments used by most commands"""
        parser.add_argument('--dataset-mode', type=str, default='OKX',
                           choices=[mode.name for mode in DATASET_MODE],
                           help='Dataset mode')
        parser.add_argument('--export-mode', type=str, default='BY_MINUTE',
                           choices=[mode.name for mode in EXPORT_MODE],
                           help='Export mode')
        parser.add_argument('--aggregation-mode', type=str, default='TAKE_LATEST',
                           choices=[mode.name for mode in AGGREGATION_MODE],
                           help='Aggregation mode')
        
        if require_dates:
            parser.add_argument('--from', dest='date_from', type=str, required=True,
                               help='Start date in YYYY-MM-DD format')
            parser.add_argument('--to', dest='date_to', type=str, required=True,
                               help='End date in YYYY-MM-DD format')
        
        parser.add_argument('--overwrite-cache', action='store_true',
                           help='Overwrite existing cache files')
        parser.add_argument('--parallel', action='store_true',
                           help='Enable parallel processing using multiprocessing')
        parser.add_argument('--workers', type=int, default=None,
                           help='Number of worker processes (default: number of CPU cores)')
    
    def create_cache_context(self, args: Namespace) -> CacheContext:
        """Create CacheContext from common arguments"""
        dataset_mode = getattr(DATASET_MODE, getattr(args, 'dataset_mode', 'OKX'))
        export_mode = getattr(EXPORT_MODE, getattr(args, 'export_mode', 'BY_MINUTE'))
        aggregation_mode = getattr(AGGREGATION_MODE, getattr(args, 'aggregation_mode', 'TAKE_LATEST'))
        return CacheContext(dataset_mode, export_mode, aggregation_mode)
    
    def create_time_range(self, args: Namespace) -> Optional[TimeRange]:
        """Create TimeRange from common arguments"""
        if hasattr(args, 'date_from') and hasattr(args, 'date_to'):
            return TimeRange(date_str_from=args.date_from, date_str_to=args.date_to)
        return None
    
    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration"""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)-5.5s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    def print_processing_info(self, args: Namespace):
        """Print common processing information"""
        print(f"Processing with parameters:")
        print(f"  Dataset Mode: {getattr(args, 'dataset_mode', 'OKX')}")
        print(f"  Export Mode: {getattr(args, 'export_mode', 'BY_MINUTE')}")
        print(f"  Aggregation Mode: {getattr(args, 'aggregation_mode', 'TAKE_LATEST')}")
        if hasattr(args, 'date_from') and hasattr(args, 'date_to'):
            print(f"  Time Range: {args.date_from} to {args.date_to}")
        if hasattr(args, 'parallel') and args.parallel:
            workers = args.workers or 'auto'
            print(f"  Parallel Processing: enabled ({workers} workers)")
    
    @abstractmethod
    def add_arguments(self, parser: ArgumentParser):
        """Add command-specific arguments"""
        pass
    
    @abstractmethod
    def handle(self, args: Namespace) -> int:
        """Execute the command, return exit code"""
        pass


class ProgressTracker:
    """Simple progress tracking utility"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
    
    def update(self, step_name: str):
        """Update progress with step name"""
        self.current_step += 1
        print(f"[{self.current_step}/{self.total_steps}] {step_name}")
    
    def finish(self, message: str = "Complete"):
        """Mark as finished"""
        print(f"✅ {message}")


def handle_common_errors(func):
    """Decorator to handle common CLI errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
            return 1
        except Exception as e:
            print(f"❌ Error: {e}")
            logging.exception("Command failed with exception")
            return 1
    return wrapper