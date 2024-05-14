import datetime, logging, os


from . import cache
from . import common
from .common import AGGREGATION_MODE
from ..util import time as util_time



def verify_data_cache(
    date_str_from: str,
    date_str_to: str,
    dataset_mode: common.DATASET_MODE,
    export_mode: common.EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
):
    logging.info(f"veryfing the data for {dataset_mode} {export_mode} {date_str_from=} and {date_str_to=}")
    t_id = common.get_full_table_id(dataset_mode, export_mode)
    t_from, t_to = util_time.to_t(
        date_str_from=date_str_from,
        date_str_to=date_str_to,
    )

    t_ranges = cache.split_t_range(t_from, t_to)
    for t_range in t_ranges:
        t_from, t_to = t_range[0], t_range[1]
        filename = cache.to_filename(t_id, aggregation_mode, t_from, t_to)
        if not os.path.exists(filename):
            logging.info(f"{filename=} does not exist.")
