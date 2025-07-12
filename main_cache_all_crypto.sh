#!/bin/bash

# Default market data cache script
# This is a wrapper script that calls the common cache script with default configuration

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the common script with crypto configuration
exec "$SCRIPT_DIR/main_cache_common.sh" --config crypto "$@"

