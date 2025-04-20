"""This module gets the San Francisco bus data from the Bus Observatory:
https://api.busobservatory.org/"""

import os
import boto3
from constants import DATA_DIR
from botocore import UNSIGNED
from botocore.config import Config

# Use unsigned requests to avoid setting aws credentials
config = Config(
    signature_version=UNSIGNED,
)


def fetch(bucket, key, target):
    """Fetches a single file"""
    fname = key.split("/")[-1].split("_")[3] + ".parquet"
    path = os.path.join(target, fname)
    os.makedirs(target, exist_ok=True)
    bucket.download_file(key, path)


def download_bus_data():
    """Downloads San Francisco bus data"""
    s3 = boto3.resource("s3", config=config)
    bucket = s3.Bucket("busobservatory-lake")
    objects = bucket.objects.filter(Prefix="feeds/sf_muni/COMPACTED")
    bus_data_dir = os.path.join(DATA_DIR, "bus", "sf")
    for obj in objects:
        fetch(bucket, obj.key, bus_data_dir)


if __name__ == "__main__":
    download_bus_data()
