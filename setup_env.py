#!/usr/bin/env python
import logging
import os, sys
from dotenv import load_dotenv

# OpenMP threading issue
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables from .env file
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'credential.json')

