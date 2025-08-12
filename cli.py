#!/usr/bin/env python3
"""
Market Data Processing CLI

Unified command-line interface for all market data processing operations.
This replaces the individual main_*.py scripts with a single, consistent interface.

Usage:
    python cli.py feature list
    python cli.py feature cache --feature bollinger --from 2024-01-01 --to 2024-01-02
    python cli.py ml-data check --features all --from 2024-01-01 --to 2024-01-02
    python cli.py target cache --forward-periods "1,2,3" --tps "0.001,0.002" --from 2024-01-01 --to 2024-01-02
"""
import os
import sys
from argparse import (ArgumentDefaultsHelpFormatter, ArgumentParser,
                      RawDescriptionHelpFormatter)
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup environment variables
import setup_env
from cli.commands import (FeatureCommand, MLDataCommand, RawCommand,
                          ResampledCommand, TargetCommand)


class MarketDataCLI:
    """Main CLI application class"""
    
    def __init__(self):
        self.commands = {
            cmd.name: cmd() for cmd in [
                FeatureCommand,
                TargetCommand,
                MLDataCommand,
                ResampledCommand,
                RawCommand
            ]
        }
    
    def create_parser(self) -> ArgumentParser:
        """Create the main argument parser with all subcommands"""
        parser = ArgumentParser(
            prog='market-data-cli',
            description='Market Data Processing CLI - Unified interface for all data operations',
            epilog="""
Examples:
  List available features:
    python cli.py feature list
    
  Cache a specific feature:
    python cli.py feature cache --feature bollinger --from 2024-01-01 --to 2024-01-02
    
  Check missing ML data:
    python cli.py ml-data check --features all --from 2024-01-01 --to 2024-01-02
    
  Cache targets with specific parameters:
    python cli.py target cache --forward-periods "1,2,3" --tps "0.001,0.002" --from 2024-01-01 --to 2024-01-02
    
  Cache resampled data with custom parameters:
    python cli.py resampled cache --resample-type-label reversal --resample-params "close,0.1,0.03" --from 2024-01-01 --to 2024-01-02
            """,
            formatter_class=RawDescriptionHelpFormatter
        )
        
        # Global options
        parser.add_argument('--verbose', '-v', action='store_true',
                           help='Enable verbose logging')
        parser.add_argument('--version', action='version', version='Market Data CLI 1.0.0')
        
        # Subcommands
        subparsers = parser.add_subparsers(
            dest='command', 
            required=True,
            help='Available commands',
            metavar='COMMAND'
        )
        
        # Add each command's subparser
        for command in self.commands.values():
            cmd_parser = subparsers.add_parser(
                command.name, 
                help=command.help,
                formatter_class=ArgumentDefaultsHelpFormatter
            )
            command.add_arguments(cmd_parser)
        
        return parser
    
    def run(self, args=None) -> int:
        """Run the CLI application"""
        parser = self.create_parser()
        
        # Parse arguments
        try:
            parsed_args = parser.parse_args(args)
        except SystemExit as e:
            return e.code
        
        # Store verbose flag in args for commands to use
        parsed_args.verbose = getattr(parsed_args, 'verbose', False)
        
        # Execute the appropriate command
        if parsed_args.command in self.commands:
            try:
                return self.commands[parsed_args.command].handle(parsed_args)
            except KeyboardInterrupt:
                print("\n⚠️  Operation cancelled by user")
                return 130  # Standard exit code for SIGINT
            except Exception as e:
                if parsed_args.verbose:
                    import traceback
                    traceback.print_exc()
                else:
                    print(f"❌ Error: {e}")
                return 1
        
        # This shouldn't happen due to required=True, but just in case
        parser.print_help()
        return 1


def main() -> int:
    """Main entry point"""
    cli = MarketDataCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())