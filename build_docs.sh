#!/bin/bash

# Set the Google credentials path
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/credential.json"

# Run make html in the docs directory
cd docs && make html 