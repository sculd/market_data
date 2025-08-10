from enum import Enum
import os
import market_data.util.cache.cache_common


class EXPORT_MODE(Enum):
    BY_MINUTE = 1
    RAW = 2
    # by update means the bar will be constructed each time
    # the exchange sends out an update.
    # by update is not yet implemented.
    BY_UPDATE = 3
    ORDERBOOK_LEVEL10 = 13
    ORDERBOOK = 14
    ORDERBOOK_LIQUIDITY_IMBALANCE = 15
    ORDERBOOK_LEVEL1 = 16

class DATASET_MODE(Enum):
    EQUITY = 1
    FOREX = 2
    BINANCE = 3
    GEMINI = 4
    OKCOIN = 5
    KRAKEN = 6
    OKX = 7
    CEX = 8
    BITHUMB = 9
    FOREX_IBKR = 10
    STOCK_HIGH_VOLATILITY = 11

class AGGREGATION_MODE(str, Enum):
    TAKE_LASTEST = "take_tatest"
    COLLECT_ALL_UPDATES = "collect_all_updates"


def get_label(dataset_mode: DATASET_MODE, export_mode: EXPORT_MODE):
    return os.path.join(dataset_mode.name.lower(), export_mode.name.lower())


class CacheContext:
    def __init__(self, dataset_mode: DATASET_MODE, export_mode: EXPORT_MODE, aggregation_mode: AGGREGATION_MODE):
        self.base_label = get_label(dataset_mode, export_mode)

    def get_folder_path(self, labels, params_dir=None):
        folder_path = os.path.join(market_data.util.cache.cache_common.cache_base_path, *labels, self.base_label)
        if params_dir is not None:
            folder_path = os.path.join(folder_path, params_dir)
        return folder_path

