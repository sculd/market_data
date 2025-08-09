import datetime
import os

from google.cloud import storage

_storage_client = storage.Client()
_gcs_bucket_name = "algo_cache"
_gcs_bucket = _storage_client.bucket(_gcs_bucket_name)


def if_blob_exist(blob_name: str) -> bool:
    return storage.Blob(bucket=_gcs_bucket, name=blob_name).exists(_storage_client)

def download_gcs_blob(source_blob_name, destination_file_name):
    bucket = _storage_client.bucket(_gcs_bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        f"Downloaded storage object {source_blob_name} from bucket {_gcs_bucket_name} to local file {destination_file_name}."
    )

def upload_file_to_gcs(local_filename, gcs_filename, rewrite=False) -> None:
    # uoload the file to gcs
    if storage.Blob(bucket=_gcs_bucket, name=gcs_filename).exists(_storage_client):
        if rewrite:
            blob = _gcs_bucket.blob(gcs_filename)
            blob.delete(if_generation_match=None)
        else:
            print(f'{gcs_filename} already present in the bucket {_gcs_bucket_name} thus not proceeding further for {property}')
            return

    blob = _gcs_bucket.blob(gcs_filename)
    generation_match_precondition = 0
    blob.upload_from_filename(local_filename, if_generation_match=generation_match_precondition)

    print(
        f"File {local_filename} uploaded to {_gcs_bucket_name}/{gcs_filename}."
    )
