import datetime, logging, sys, os


if os.path.exists('./credential.json'):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'credential.json')
    os.environ["GOOGLE_CLOUD_PROJECT"] = "trading-290017"
else:
    print('the credential.json file does not exist')


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

import ingest.bq.validate
import ingest.bq.common

if __name__ == '__main__':
    ingest.bq.validate.verify_data_cache(
        date_str_from='2024-01-01',
        date_str_to='2024-03-21',
        dataset_mode=ingest.bq.common.DATASET_MODE.OKX,
        export_mode=ingest.bq.common.EXPORT_MODE.BY_MINUTE,
        aggregation_mode=ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    )