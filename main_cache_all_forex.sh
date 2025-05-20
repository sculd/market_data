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

python main_raw_data.py --action check --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_feature_data.py --action check --feature forex --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_resampled_target_data.py --action check --target --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_resampled_target_data.py --action check --resample --resample_params close,0.01 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_resampled_target_data.py --action check --resample --resample_params close,0.015 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_resampled_target_data.py --action check --resample --resample_params close,0.02 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_resampled_target_data.py --action check --resample --resample_params close,0.03 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_ml_data.py --action check --features forex --resample_params close,0.01 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_ml_data.py --action check --features forex --resample_params close,0.015 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_ml_data.py --action check --features forex --resample_params close,0.02 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_ml_data.py --action check --features forex --resample_params close,0.03 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}

# Ask for confirmation before proceeding
read -p "Do you want to proceed with caching data? (y/N): " response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Caching canceled."
    exit 0
fi

echo "Caching data from $FROM_DATE to $TO_DATE"

#python main_raw_data.py --action cache --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_feature_data.py --action cache --feature forex --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_resampled_target_data.py --action cache --target --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_resampled_target_data.py --action cache --resample --resample_params close,0.01 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_resampled_target_data.py --action cache --resample --resample_params close,0.015 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_resampled_target_data.py --action cache --resample --resample_params close,0.02 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_resampled_target_data.py --action cache --resample --resample_params close,0.03 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_ml_data.py --action cache --features forex --resample_params close,0.01 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_ml_data.py --action cache --features forex --resample_params close,0.015 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_ml_data.py --action cache --features forex --resample_params close,0.02 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
python main_ml_data.py --action cache --features forex --resample_params close,0.03 --from "$FROM_DATE" --to "$TO_DATE" ${dataset_aggregation_options}
