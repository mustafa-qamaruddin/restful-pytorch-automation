from google.cloud import storage
import os
import re


if not os.path.exists("DATA"):
    os.mkdir("DATA")


storage_client = storage.Client.from_service_account_json('API Project-4675d7ee66b1.json')

ls_itr = storage_client.list_buckets(prefix="scanvid--images--")
for b in ls_itr:
    if "scanvid--images--" not in b.name:
        continue
    results = re.search("scanvid--images--(\d+)", b.name)
    if not results:
        continue
    label = results.group(1)
    bucket = storage_client.get_bucket(b.name)
    blobs = bucket.list_blobs()
    idx = 0
    for blob in blobs:
        if not os.path.exists(os.path.join("DATA", label)):
            os.mkdir(os.path.join("DATA", label))
        dir = os.path.join("DATA", label)
        file = os.path.join(dir, "{}.png".format(str(idx).zfill(10)))
        data = bucket.blob(blob.name)
        data.download_to_filename(file)
        idx += 1
