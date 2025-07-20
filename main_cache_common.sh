#!/bin/bash

# Common cache script for market data pipeline
# Usage: ./main_cache_common.sh --config CONFIG_NAME --from DATE --to DATE
#
# CONFIG_NAME can be: stock, forex, default
# This script contains the common logic for checking and caching market data

# Function to display usage
usage() {
    echo "Usage: $0 --config CONFIG_NAME --from YYYY-MM-DD --to YYYY-MM-DD [--skip-check] [--datatypes TYPES] [--no-overwrite-cache]"
    echo "CONFIG_NAME: crypto, stock, forex, or default"
    echo "Options:"
    echo "  --skip-check           Skip the data checking phase and go directly to caching"
    echo "  --datatypes TYPES      Comma-separated list of data types to process:"
    echo "                         raw,feature,target,resample,feature_resample,ml_data (default: all)"
    echo "  --no-overwrite-cache   Disable cache overwriting (default: enabled)"
    echo "Examples:"
    echo "  $0 --config stock --from 2024-01-01 --to 2024-01-02"
    echo "  $0 --config stock --from 2024-01-01 --to 2024-01-02 --datatypes raw,feature"
    echo "  $0 --config stock --from 2024-01-01 --to 2024-01-02 --skip-check --datatypes target"
    echo "  $0 --config stock --from 2024-01-01 --to 2024-01-02 --no-overwrite-cache"
    exit 1
}

# Parse command line arguments
SKIP_CHECK=false
DATATYPES="raw,feature,target,resample,feature_resample,ml_data"  # Default to all
OVERWRITE_CACHE=true  # Default to overwrite cache
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --from)
      FROM_DATE="$2"
      shift 2
      ;;
    --to)
      TO_DATE="$2"
      shift 2
      ;;
    --skip-check)
      SKIP_CHECK=true
      shift
      ;;
    --datatypes)
      DATATYPES="$2"
      shift 2
      ;;
    --no-overwrite-cache)
      OVERWRITE_CACHE=false
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      usage
      ;;
  esac
done

# Check if required parameters are set
if [ -z "$CONFIG" ] || [ -z "$FROM_DATE" ] || [ -z "$TO_DATE" ]; then
  echo "Error: All parameters (--config, --from, --to) are required"
  usage
fi

# Helper function to check if a datatype is selected
is_datatype_selected() {
    local datatype=$1
    [[ ",$DATATYPES," == *",$datatype,"* ]]
}

# Set configuration based on config parameter
case $CONFIG in
    stock)
        dataset_aggregation_options="--dataset_mode STOCK_HIGH_VOLATILITY --aggregation_mode COLLECT_ALL_UPDATES "
        target_arg="--forward_periods=10,30,60 --tps=0.03,0.05"
        feature_type="stock"
        warm_param="--warmup-days=0"
        ;;
    forex)
        dataset_aggregation_options="--dataset_mode FOREX_IBKR --aggregation_mode COLLECT_ALL_UPDATES "
        target_arg="--forward_periods=10,30,60 --tps=0.0025,0.005,0.01"
        feature_type="forex"
        warm_param=""
        ;;
    crypto)
        dataset_aggregation_options=""
        target_arg="--forward_periods=5,10,30 --tps=0.015,0.03,0.05"
        feature_type="crypto"
        warm_param=""
        ;;
    default)
        dataset_aggregation_options=""
        target_arg=""
        feature_type="all"
        warm_param=""
        ;;
    *)
        echo "Error: Invalid config '$CONFIG'. Must be 'crypto', 'stock', 'forex', or 'default'"
        usage
        ;;
esac
parallel_param="--parallel

# Get all registered resample methods from Python
echo "Discovering resample methods and parameters..."
registered_methods=($(python -c "
from market_data.machine_learning.resample import list_registered_resample_methods
print(' '.join(list_registered_resample_methods()))
"))

# Create a temporary file to store method-parameter mappings
temp_file=$(mktemp)
trap "rm -f $temp_file" EXIT

# Build resample method configuration
for method in "${registered_methods[@]}"; do
    params_output=$(python -c "
from market_data.machine_learning.resample import get_resample_params_class
try:
    cls = get_resample_params_class('$method')
    if hasattr(cls, 'get_default_params_for_config'):
        params = cls.get_default_params_for_config('$CONFIG')
        for param in params:
            print('$method:' + param)
    else:
        print('# No default params for $method')
except Exception as e:
    print('# Error getting params for $method:', str(e))
" 2>/dev/null)
    
    # Only add methods that have valid parameters
    if [[ ! "$params_output" =~ ^"#" ]]; then
        echo "$params_output" >> "$temp_file"
    fi
done

echo "Running $CONFIG configuration from $FROM_DATE to $TO_DATE"
echo "Selected datatypes: $DATATYPES"
echo "Cache overwrite: $([ "$OVERWRITE_CACHE" = true ] && echo "enabled" || echo "disabled")"
echo "Discovered resample methods and parameters:"
while IFS=':' read -r method param; do
    echo "  $method: $param"
done < "$temp_file"

# Main function to process data (check or cache)
process_data() {
    local action=$1
    local action_display=$2
    
    # Set additional flags for cache operations
    local cache_flags=""
    if [ "$action" = "cache" ] && [ "$OVERWRITE_CACHE" = true ]; then
        cache_flags="--overwrite_cache"
    fi
    
    echo "=== ${action_display} DATA ==="

    if is_datatype_selected "raw" && [ "$CONFIG" != "stock" ]; then
        echo "${action_display} raw data..."
        cmd="python main_raw_data.py --action $action --from \"$FROM_DATE\" --to \"$TO_DATE\" ${dataset_aggregation_options} ${parallel_param}"
        echo "Running: $cmd"
        eval $cmd
    elif is_datatype_selected "raw" && [ "$CONFIG" = "stock" ]; then
        echo "Skipping raw data ${action,,} for stock (separate process)"
    fi

    if is_datatype_selected "feature"; then
        echo "${action_display} feature data..."
        cmd="python main_feature_data.py --action $action --feature $feature_type --from \"$FROM_DATE\" --to \"$TO_DATE\" ${dataset_aggregation_options} ${warm_param} ${parallel_param}"
        echo "Running: $cmd"
        eval $cmd
    fi

    if is_datatype_selected "target"; then
        if [ -n "$target_arg" ]; then
            echo "${action_display} target data..."
            cmd="python main_target_data.py --action $action $target_arg --from \"$FROM_DATE\" --to \"$TO_DATE\" ${cache_flags} ${dataset_aggregation_options} ${parallel_param}"
            echo "Running: $cmd"
            eval $cmd
        else
            echo "❌ ERROR: Target processing requested but not configured for '$CONFIG' configuration"
            echo "   Target processing is only available for: stock, forex, crypto"
            echo "   Current config '$CONFIG' has no target_arg defined (forward_periods, tps, etc.)"
            echo "   Either:"
            echo "   1. Use a different config: --config stock, --config forex, or --config crypto"
            echo "   2. Remove 'target' from --datatypes parameter"
            echo "   3. Modify the script to add target_arg for '$CONFIG' configuration"
        fi
    fi

    if is_datatype_selected "resample"; then
        echo "${action_display} resampled data..."
        while IFS=':' read -r method param; do
            echo "  Processing resample method: $method with params: $param"
            cmd="python main_resampled_data.py --action $action --resample_type_label $method --resample_params \"$param\" --from \"$FROM_DATE\" --to \"$TO_DATE\" ${cache_flags} ${dataset_aggregation_options} ${parallel_param}"
            echo "Running: $cmd"
            eval $cmd
        done < "$temp_file"
    fi

    if is_datatype_selected "feature_resample"; then
        echo "${action_display} feature_resample data..."
        while IFS=':' read -r method param; do
            echo "  Processing feature_resample for method: $method with params: $param"
            cmd="python main_feature_resampled_data.py --action $action --feature $feature_type --resample_type_label $method --resample_params \"$param\" --from \"$FROM_DATE\" --to \"$TO_DATE\" ${cache_flags} ${dataset_aggregation_options} ${parallel_param}"
            echo "Running: $cmd"
            eval $cmd
        done < "$temp_file"
    fi

    if is_datatype_selected "ml_data"; then
        echo "${action_display} ML data..."
        while IFS=':' read -r method param; do
            echo "  Processing ML data for method: $method with params: $param"
            cmd="python main_ml_data.py --action $action --features $feature_type $target_arg --resample_type_label $method --resample_params \"$param\" --from \"$FROM_DATE\" --to \"$TO_DATE\" ${cache_flags} ${dataset_aggregation_options} ${parallel_param}"
            echo "Running: $cmd"
            eval $cmd
        done < "$temp_file"
    fi
}

# Check phase (only if not skipped)
if [ "$SKIP_CHECK" = false ]; then
    process_data "check" "CHECKING"

    # Ask for confirmation before proceeding
    echo ""
    read -p "Do you want to proceed with caching data? (y/N): " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Caching canceled."
        exit 0
    fi
else
    echo "=== SKIPPING DATA CHECK (--skip-check enabled) ==="
    echo ""
    read -p "Proceed directly with caching data? (y/N): " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Caching canceled."
        exit 0
    fi
fi

# Cache phase
echo ""
process_data "cache" "CACHING"

echo ""
echo "✅ $CONFIG configuration caching completed successfully!" 