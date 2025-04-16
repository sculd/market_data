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

### Documentation

The project uses Sphinx for generating API documentation. To generate the documentation:

1. Install the required documentation packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the documentation:
   ```bash
   ./build_docs.sh
   ```

3. View the documentation:
   - Open `docs/build/html/index.html` in your web browser
   - Or use a local server:
     ```bash
     cd docs/build/html
     python -m http.server 8000
     ```
     Then visit `http://localhost:8000` in your browser

The documentation includes:
- Complete API reference for all modules
- Type hints and parameter descriptions
- Code examples with copy buttons
- Search functionality
- Cross-referencing between modules

To automatically rebuild documentation when files change:
```bash
cd docs
make watch
```

Note: The documentation build directory (`docs/build/`) is ignored by Git. Each developer should build their own documentation locally.

### Target Engineering

The project now includes a target engineering module for machine learning purposes, creating prediction targets based on future price movements:

#### Available Targets

**Forward Returns:**
- Future price returns over various horizons (`label_forward_return_2m`, `label_forward_return_10m` by default)
  
**Classification Labels:**
- Take-profit/stop-loss labels for both long and short positions
- Configurable time horizons, take-profit thresholds, and stop-loss thresholds
- Labels are 1 (take-profit reached), -1 (stop-loss triggered), or 0 (neither within time horizon)

All target columns have the `label_` prefix to easily identify them as machine learning targets.

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