from minio import Minio

client = Minio(
    "storage_address",
    access_key="XXXXXXXXXXXXXXXXX",
    secret_key="XXXXXXXXXXXXXXXXXXXXXXXXXXX",
)

for bucket in client.list_buckets():
    for item in client.list_objects(bucket.name,recursive=True):
        client.fget_object(bucket.name,item.object_name,item.object_name)