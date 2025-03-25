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

The feature engineering module calculates:

**Price Indicators:**
- Returns over various horizons (1, 5, 15, 30, 60, 120 minutes by default)
- Exponential moving averages (5, 15, 30, 60, 120, 240 minutes by default)
- Bollinger Bands (20-minute with 2 standard deviations)
- True Range
- Open/Close Ratio
- RSI (Relative Strength Index)
- Autocorrelation with lag 1
- Z-score Normalization
- Min-Max Scaling

**Volume Indicators:**
- Volume Ratio vs. 20-minute average
- On-Balance Volume (OBV)

**Market Correlation Features:**
- BTC returns over various horizons (added as features for all non-BTC symbols)

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
  - `example.py`: Example usage of the feature engineering module

## Environment Variables

The project uses python-dotenv to manage environment variables:

- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID (required for BigQuery operations)

Additional environment variables can be added to the `.env` file as needed.
