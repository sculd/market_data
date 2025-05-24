#!/bin/bash

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --from)
      FROM_DATE="$2"
      shift 2
      ;;
    --to)
      TO_DATE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Check if FROM_DATE and TO_DATE are set
if [ -z "$FROM_DATE" ] || [ -z "$TO_DATE" ]; then
  echo "Error: Both --from and --to parameters are required"
  echo "Usage: $0 --from YYYY-MM-DD --to YYYY-MM-DD"
  exit 1
fi

echo "Checking data from $FROM_DATE to $TO_DATE"
dataset_aggregation_options="--dataset_mode FOREX_IBKR --aggregation_mode COLLECT_ALL_UPDATES "
target_arg="--forward_periods=10,30,60 --tps=0.0025,0.005,0.01"
resample_params_list=("close,0.0025" "close,0.005" "close,0.01")

python main_target_data.py --action check $target_arg --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_raw_data.py --action check --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_feature_data.py --action check --feature forex --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
for resample_params in "${resample_params_list[@]}"; do
    python main_resampled_data.py --action check --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
done
for resample_params in "${resample_params_list[@]}"; do
    python main_ml_data.py --action check --features forex $target_arg --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
done

# Ask for confirmation before proceeding
read -p "Do you want to proceed with caching data? (y/N): " response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Caching canceled."
    exit 0
fi

echo "Caching data from $FROM_DATE to $TO_DATE"

python main_raw_data.py --action cache --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_feature_data.py --action cache --feature forex --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_target_data.py --action cache $target_arg --from "$FROM_DATE" --to "$TO_DATE" --overwrite_cache ${dataset_aggregation_options}
for resample_params in "${resample_params_list[@]}"; do
    python main_resampled_data.py --action cache --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" --overwrite_cache ${dataset_aggregation_options}
done
for resample_params in "${resample_params_list[@]}"; do
    python main_ml_data.py --action cache --features forex $target_arg --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" --overwrite_cache ${dataset_aggregation_options}
done
