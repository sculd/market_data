#!/bin/bash

# Common cache script for market data pipeline
# Usage: ./main_cache_common.sh --config CONFIG_NAME --from DATE --to DATE
#
# CONFIG_NAME can be: stock, forex, default
# This script contains the common logic for checking and caching market data

# Function to display usage
usage() {
    echo "Usage: $0 --config CONFIG_NAME --from YYYY-MM-DD --to YYYY-MM-DD [--skip-check] [--datatypes TYPES]"
    echo "CONFIG_NAME: stock, forex, or default"
    echo "Options:"
    echo "  --skip-check           Skip the data checking phase and go directly to caching"
    echo "  --datatypes TYPES      Comma-separated list of data types to process:"
    echo "                         raw,feature,target,resample,ml_data (default: all)"
    echo "Examples:"
    echo "  $0 --config stock --from 2024-01-01 --to 2024-01-02"
    echo "  $0 --config stock --from 2024-01-01 --to 2024-01-02 --datatypes raw,feature"
    echo "  $0 --config stock --from 2024-01-01 --to 2024-01-02 --skip-check --datatypes target"
    exit 1
}

# Parse command line arguments
SKIP_CHECK=false
DATATYPES="raw,feature,target,resample,ml_data"  # Default to all
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
        resample_params_list=("close,0.03" "close,0.05" "close,0.07" "close,0.1" "close,0.15")
        warm_param="--warmup-days=0"
        ;;
    forex)
        dataset_aggregation_options="--dataset_mode FOREX_IBKR --aggregation_mode COLLECT_ALL_UPDATES "
        target_arg="--forward_periods=10,30,60 --tps=0.0025,0.005,0.01"
        feature_type="forex"
        resample_params_list=("close,0.0025" "close,0.005" "close,0.01")
        warm_param=""
        ;;
    crypto)
        dataset_aggregation_options=""
        target_arg="--forward_periods=5,10,30 --tps=0.015,0.03,0.05"
        feature_type="crypto"
        resample_params_list=("close,0.03" "close,0.05" "close,0.07" "close,0.1" "close,0.15")
        warm_param=""
        ;;
    default)
        dataset_aggregation_options=""
        target_arg=""
        feature_type="all"
        resample_params_list=("close,0.03" "close,0.05" "close,0.07" "close,0.1" "close,0.15")
        warm_param=""
        ;;
    *)
        echo "Error: Invalid config '$CONFIG'. Must be 'crypto', 'stock', 'forex', or 'default'"
        usage
        ;;
esac

echo "Running $CONFIG configuration from $FROM_DATE to $TO_DATE"
echo "Selected datatypes: $DATATYPES"

# Check phase (only if not skipped)
if [ "$SKIP_CHECK" = false ]; then
    echo "=== CHECKING DATA ==="

    if is_datatype_selected "raw" && [ "$CONFIG" != "stock" ]; then
        echo "Checking raw data..."
        python main_raw_data.py --action check --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
    elif is_datatype_selected "raw" && [ "$CONFIG" = "stock" ]; then
        echo "Skipping raw data check for stock (separate process)"
    fi

    if is_datatype_selected "feature"; then
        echo "Checking feature data..."
        python main_feature_data.py --action check --feature $feature_type --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options} ${warm_param}
    fi

    if is_datatype_selected "target" && [ -n "$target_arg" ]; then
        echo "Checking target data..."
        python main_target_data.py --action check $target_arg --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
    fi

    if is_datatype_selected "resample"; then
        echo "Checking resampled data..."
        for resample_params in "${resample_params_list[@]}"; do
            python main_resampled_data.py --action check --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
        done
    fi

    if is_datatype_selected "ml_data"; then
        echo "Checking ML data..."
        for resample_params in "${resample_params_list[@]}"; do
            if [ -n "$target_arg" ]; then
                python main_ml_data.py --action check --features $feature_type $target_arg --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
            else
                python main_ml_data.py --action check --features $feature_type --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
            fi
        done
    fi

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
echo "=== CACHING DATA ==="

if is_datatype_selected "raw" && [ "$CONFIG" != "stock" ]; then
    echo "Caching raw data..."
    python main_raw_data.py --action cache --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
elif is_datatype_selected "raw" && [ "$CONFIG" = "stock" ]; then
    echo "Skipping raw data caching for stock (separate process)"
fi

if is_datatype_selected "feature"; then
    echo "Caching feature data..."
    python main_feature_data.py --action cache --feature $feature_type --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options} ${warm_param}
fi

if is_datatype_selected "target" && [ -n "$target_arg" ]; then
    echo "Caching target data..."
    python main_target_data.py --action cache $target_arg --from "$FROM_DATE" --to "$TO_DATE" --overwrite_cache ${dataset_aggregation_options}
fi

if is_datatype_selected "resample"; then
    echo "Caching resampled data..."
    for resample_params in "${resample_params_list[@]}"; do
        python main_resampled_data.py --action cache --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" --overwrite_cache ${dataset_aggregation_options}
    done
fi

if is_datatype_selected "ml_data"; then
    echo "Caching ML data..."
    for resample_params in "${resample_params_list[@]}"; do
        python main_ml_data.py --action cache --features $feature_type $target_arg --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" --overwrite_cache ${dataset_aggregation_options}
    done
fi

echo ""
echo "✅ $CONFIG configuration caching completed successfully!" 