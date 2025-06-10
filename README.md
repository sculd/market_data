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
   - Create a `.env` file with your Google Cloud Project ID:
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

### Feature Engineering

Comprehensive feature engineering module for market data analysis on minute-level data:

**Price-Based Features:**
- Returns over multiple time horizons (`return_1m`, `return_5m`, `return_15m`, `return_30m`, `return_60m`, `return_120m`)
- Exponential moving averages and ratios (`ema_5m`, `ema_15m`, etc. and `ema_rel_*`)
- Bollinger Bands position and width (`bb_position`, `bb_width`)
- Volatility metrics (`true_range`, `hl_range_pct`)
- Technical indicators (`rsi`, `autocorr_lag1`, `close_zscore`, `close_minmax`)

**Volume-Based Features:**
- Volume ratios and On-Balance Volume (`volume_ratio_log`, `obv_pct_change_log`)
- Fractionally differenced volume indicators (`obv_ffd_zscore`)

**Fractional Difference Features:**
- Fractionally differenced and z-scored features for stationarity (`ffd_zscore_close`, `ffd_zscore_volume`)
- Reduces non-stationarity while preserving memory in time series

**Market Correlation Features:**
- BTC-specific returns for correlation analysis (`btc_return_1m`, `btc_return_5m`, etc.)

### Target Engineering

Creates prediction targets for machine learning:
- Forward returns (`label_forward_return_2m`, `label_forward_return_10m`)
- Take-profit/stop-loss classification labels (1=profit, -1=loss, 0=neither)

### Event-Based Resampling

Implements López de Prado's CUMSUM breakout filter for intelligent sampling:
- Identifies significant price movements using cumulative return thresholds
- Essential for building machine learnable dataset

### Command-line Tools

Main pipeline tools:
- `main_raw_data.py`: Raw market data operations and BigQuery caching
- `main_feature_data.py`: Feature calculation and caching
- `main_target_data.py`: Target label processing and caching
- `main_resampled_data.py`: Event-based resampling for significant price movements
- `main_ml_data.py`: ML dataset preparation combining features and targets
- `main_data.py`: General data operations and utilities

### Documentation

Generate API documentation with Sphinx:
```bash
pip install -r requirements.txt
./build_docs.sh
```
View at `docs/build/html/index.html` or serve locally with `python -m http.server 8000`.

## Project Structure

- `market_data/`: Main package
  - `ingest/`: Data ingestion (BigQuery, Polygon, GCS)
  - `feature/`: Feature engineering and caching
  - `target/`: Target engineering for ML
  - `machine_learning/`: ML utilities and resampling
  - `util/`: Common utilities and caching

## Environment Variables

- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID (required for BigQuery operations)