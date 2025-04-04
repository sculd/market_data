Welcome to Market Data's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
========

This documentation covers the Market Data project, which provides tools for processing and analyzing market data. The project includes modules for:

* Data ingestion and caching
* Feature engineering
* Machine learning utilities
* Data resampling and preprocessing

Getting Started
===============

To get started with the Market Data project:

1. Install the package in development mode:

   .. code-block:: bash

      pip install -e .

2. Import the modules you need:

   .. code-block:: python

      from market_data.machine_learning import resample
      from market_data.feature import feature
      from market_data.ingest import bq

For more detailed information about specific modules, please refer to the API documentation below. 