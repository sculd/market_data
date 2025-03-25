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

## Project Structure

- `ingest/`: Data ingestion modules
  - `bq/`: BigQuery-related functionality
    - `common.py`: Shared constants and utilities
    - `cache.py`: Caching mechanisms for BigQuery data

## Environment Variables

The project uses python-dotenv to manage environment variables:

- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID (required for BigQuery operations)

Additional environment variables can be added to the `.env` file as needed.
