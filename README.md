# Market Data

A project for ingesting, processing, and analyzing market data using Google BigQuery.

## Setup

### Prerequisites

- Python 3.6+
- Google Cloud account with BigQuery access
- GCP credential file (credential.json)

### Environment Setup

1. Clone the repository:
   ```
   git clone [repository-url]
   cd market_data
   ```

2. Environment Variables:
   - Copy `.env.example` to create your own `.env` file:
     ```
     cp .env.example .env
     ```
   - Edit the `.env` file and add your actual Google Cloud Project ID:
     ```
     GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
     ```
   - Place your Google Cloud credential file as `credential.json` in the project root

## Usage

### Running the Sandbox

The `sandbox.py` file provides a testing environment for data operations:

```
python sandbox.py
```

This script loads environment variables from the `.env` file and configures Google Cloud credentials automatically.

### Feature Engineering

The project includes a comprehensive feature engineering module for market data analysis on minute-level data, providing various technical indicators:

```python
from feature import create_features

# Load your OHLCV dataframe with timestamp index (minute-level data)
df = load_data()  

# Generate all features
features_df = create_features(df)
```

#### Available Features

The feature engineering module calculates the following indicators:

**Price-Based Features:**
- **Return Metrics:**
  - Returns over multiple time horizons (`return_1m`, `return_5m`, `return_15m`, `return_30m`, `return_60m`, `return_120m`)

- **Moving Averages:**
  - Exponential moving averages over multiple time horizons (`ema_5m`, `ema_15m`, `ema_30m`, `ema_60m`, `ema_120m`, `ema_240m`)
  - EMA relative to current price - normalized ratios (`ema_rel_5m`, `ema_rel_15m`, etc.)

- **Bollinger Bands (20-period, 2 standard deviations):**
  - Relative position within bands (`bb_position`) - normalized between 0 and 1
  - Band width as percentage of middle band (`bb_width`)

- **Volatility Metrics:**
  - True Range (`true_range`)
  - High-Low range as percentage of price (`hl_range_pct`)

- **Price Ratios:**
  - Open to Close ratio (`open_close_ratio`)

- **Momentum Indicators:**
  - Relative Strength Index - 14 period (`rsi`)

- **Statistical Features:**
  - Price autocorrelation with lag 1 (`autocorr_lag1`)
  - Z-score normalized price - 20 period window (`close_zscore`)
  - Min-Max scaled price - 20 period window (`close_minmax`)

**Volume-Based Features:**
- **Volume Metrics:**
  - Volume ratio vs. 20-minute average (`volume_ratio_20m`)
  - On-Balance Volume (`obv`)
  - Z-score normalized OBV - 20 period window (`obv_zscore`)

**Market Correlation Features:**
- BTC-specific returns for all time horizons - added to non-BTC symbols (`btc_return_1m`, `btc_return_5m`, etc.)
- Note: The code identifies Bitcoin data based on symbols containing "BTC" in their name
  - Different exchanges use different symbols for Bitcoin:
    - OKCoin uses "BTC-USDT-SWAP"
    - Other exchanges may use "BTCUSDT", "BTC/USDT", etc.
  - Ensure your DataFrame includes the appropriate Bitcoin symbol for your exchange
  - The feature engineering will automatically find the Bitcoin data to calculate correlation features

**Feature Naming Convention:**
Features are named based on their calculation method and time horizon. For example, `return_30m` represents the 30-minute price return, and `ema_60m` represents the 60-minute exponential moving average.

### Target Engineering

The project now includes a target engineering module for machine learning purposes, creating prediction targets based on future price movements:

```python
from feature import create_targets

# Generate target variables for ML
targets_df = create_targets(df)
```

#### Available Targets

**Forward Returns:**
- Future price returns over various horizons (`label_forward_return_2m`, `label_forward_return_10m` by default)
  
**Classification Labels:**
- Take-profit/stop-loss labels for both long and short positions
- Configurable time horizons, take-profit thresholds, and stop-loss thresholds
- Labels are 1 (take-profit reached), -1 (stop-loss triggered), or 0 (neither within time horizon)

All target columns have the `label_` prefix to easily identify them as machine learning targets.

### Sequence Features

For sequence-based models (LSTMs, Transformers), the module can generate arrays of past values for each feature:

```python
from feature import create_sequence_features

# Generate sequence features with 60-point history for each feature
sequence_df = create_sequence_features(df, sequence_length=60)
```

### Combined Data Preparation

For convenience, the module provides functions to generate both features and targets in one step:

```python
from feature import create_features_with_targets, create_sequence_features_with_targets

# For regular ML models - get features and targets in one DataFrame
combined_df = create_features_with_targets(
    df,
    forward_periods=[10, 60],
    tp_values=[0.01, 0.03],
    sl_values=[0.01, 0.03]
)

# For sequence models - get both sequence features and targets in one DataFrame
combined_seq_df = create_sequence_features_with_targets(
    df,
    sequence_length=60,
    forward_periods=[10, 60]
)
```

### Event-Based Resampling for Machine Learning

For machine learning tasks, it's often beneficial to sample data based on significant market events rather than at fixed time intervals. The project includes event-based resampling functionality:

```python
from machine_learning import get_events_t_multi, create_resampled_dataset, create_resampled_seq_dataset

# Method 1: Manual approach - first identify events, then filter your data
# Identify timestamps where price moves significantly (1% threshold)
# Resets cumulative sums for each symbol and date
events_df = get_events_t_multi(df, 'close', threshold=0.01)
print(f"Found {len(events_df)} significant price movement events")

# Generate features and targets using the feature engineering system
combined_df = create_features_with_targets(
    df,
    forward_periods=[10, 30],  # Forward returns for 10 and 30 minutes
    tp_values=[0.02],  # Take-profit threshold of 2%
    sl_values=[0.01]   # Stop-loss threshold of 1%
)

# Filter the data to only include the timestamps from events_df
# First create multi-index DataFrames
events_multi = events_df.reset_index().set_index(['timestamp', 'symbol'])
combined_multi = combined_df.reset_index().set_index(['timestamp', 'symbol'])

# Join to get only the rows at event timestamps
ml_dataset = combined_multi.loc[events_multi.index].reset_index().set_index('timestamp')

# Method 2: Simplified approach using convenience functions
# For regular features - one function handles everything
ml_dataset = create_resampled_dataset(
    df,
    price_col='close',
    threshold=0.01,  # 1% price movement
    forward_periods=[10, 30],  # Forward returns for 10 and 30 minutes
    tp_values=[0.02],  # Take-profit threshold of 2%
    sl_values=[0.01]   # Stop-loss threshold of 1%
)
print(f"ML dataset has {len(ml_dataset)} samples")

# For sequence features - use the sequence-specific function
seq_ml_dataset = create_resampled_seq_dataset(
    df,
    price_col='close',
    threshold=0.01,
    sequence_length=60,
    forward_periods=[10, 30]
)
print(f"Sequence ML dataset has {len(seq_ml_dataset)} samples")
```

#### Resampling Methodology

The resampling functionality:

1. **Identifies Significant Price Movements**: Uses cumulative price changes to detect meaningful market events
2. **Provides Symbol Isolation**: Resets cumulative sums for each symbol, ensuring events are specific to each instrument
3. **Maintains Date Boundaries**: Resets at each new day to prevent cross-day trend dependencies
4. **Works with Feature Engineering**: Integrated with the feature engineering system through convenient functions

This approach helps create more informative training datasets by focusing on periods of market activity rather than arbitrary time intervals.

### Exporting ML Datasets

The project provides functions to easily export machine learning datasets, with each target variable in a separate file:

```python
from machine_learning import export_resampled_datasets, export_resampled_sequence_datasets

# Export regular feature datasets (one file per target)
exported_files = export_resampled_datasets(
    df,
    export_dir="ml_datasets",
    price_col='close',
    threshold=0.01,  # 1% price movement
    forward_periods=[10, 30],  # Forward returns for 10 and 30 minutes
    tp_values=[0.02],  # Take-profit threshold of 2%
    sl_values=[0.01],  # Stop-loss threshold of 1%
    forward_return_only=True,  # Only export forward return targets
    file_format='parquet'  # Export as Parquet files (default)
)

print(f"Exported {len(exported_files)} datasets to {', '.join(list(exported_files.values())[:2])}...")

# Export sequence datasets (one file per target)
seq_files = export_resampled_sequence_datasets(
    df, 
    export_dir="ml_sequence_datasets",
    threshold=0.01,
    sequence_length=60,
    file_format='pickle'  # For sequence data, pickle is recommended
)
```

The export functions:
1. Create event-based resampled datasets (regular or sequence features)
2. Split the data by target variable (e.g., separate files for 10m returns, 30m returns, etc.)
3. Filter out rows with NaN targets to ensure clean training data
4. Save to the specified format and location with informative filenames
5. **Regular features**: Exported as Parquet files by default to preserve timestamp precision
6. **Sequence features**: Automatically detected based on numpy array content and exported as pickle files
7. **Symbol independence**: The 'symbol' column is excluded from exported datasets, ensuring models focus on price patterns rather than specific cryptocurrencies

This approach simplifies the process of preparing data for training separate models on different target variables, while ensuring all datasets are properly resampled based on significant market events.

To try the feature engineering module with example data:

```
python -m feature.example
```

## Project Structure

- `ingest/`: Data ingestion modules
  - `bq/`: BigQuery-related functionality
    - `common.py`: Shared constants and utilities
    - `cache.py`: Caching mechanisms for BigQuery data
- `feature/`: Feature engineering modules
  - `feature.py`: Main feature engineering class and functions
  - `target.py`: Target engineering for machine learning
  - `data.py`: Combined data preparation functions
  - `example.py`: Example usage of the feature engineering module
- `machine_learning/`: Machine learning utilities
  - `resample.py`: Event-based resampling functionality for identifying significant price movements
  - `feature_target.py`: Combined features and targets preparation
  - `data.py`: Dataset export utilities for ML

## Environment Variables

The project uses python-dotenv to manage environment variables:

- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID (required for BigQuery operations)

Additional environment variables can be added to the `.env` file as needed.