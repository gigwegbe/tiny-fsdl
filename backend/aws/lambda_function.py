import json
import urllib.parse
import urllib.request
import boto3
import base64
from codecs import encode


def lambda_handler(event, context):
    print(event)
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    s3 = boto3.client('s3')
    obj = s3.get_object(
        Bucket=bucket,
        Key=key
    )

    text = obj['Body'].read().decode()
    data = json.loads(text)
    img = data["message"]
    filename = data['filename']
    download_path = "converted"

    bytes_img = encode(img, 'utf-8')
    binary_img = base64.decodebytes(bytes_img)

    with open(f'/tmp/{filename}', 'wb') as fout:
        fout.write(binary_img)

    s3.upload_file(f'/tmp/{filename}', bucket, f'{download_path}/{filename}')
