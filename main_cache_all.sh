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

resample_params_list=("close,0.03" "close,0.05" "close,0.07")

echo "Checking data from $FROM_DATE to $TO_DATE"

python main_raw_data.py --action check --from "$FROM_DATE" --to "$TO_DATE"
python main_feature_data.py --action check --feature all --from "$FROM_DATE" --to "$TO_DATE"
python main_target_data.py --action check --from "$FROM_DATE" --to "$TO_DATE"
for resample_params in "${resample_params_list[@]}"; do
    python main_resampled_data.py --action check --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE"
done
for resample_params in "${resample_params_list[@]}"; do
    python main_ml_data.py --action check --features all --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE"
done

# Ask for confirmation before proceeding
read -p "Do you want to proceed with caching data? (y/N): " response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Caching canceled."
    exit 0
fi

echo "Caching data from $FROM_DATE to $TO_DATE"

python main_raw_data.py --action cache --from "$FROM_DATE" --to "$TO_DATE"
python main_feature_data.py --action cache --feature all --from "$FROM_DATE" --to "$TO_DATE"
python main_target_data.py --action cache --from "$FROM_DATE" --to "$TO_DATE"
for resample_params in "${resample_params_list[@]}"; do
    python main_resampled_data.py --action cache --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" --overwrite_cache
done
for resample_params in "${resample_params_list[@]}"; do
    python main_ml_data.py --action cache --features all --resample_params $resample_params --from "$FROM_DATE" --to "$TO_DATE" --overwrite_cache
done

