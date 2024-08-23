import base64
import boto3
import json
import io
import numpy as np


def lambda_handler(event, context):
    base64_image = event["receipt_image"]
    base64_decoded_image = base64.b64decode(base64_image)

    np_array = np.frombuffer(base64_decoded_image, np.uint8)
    image_io = io.BytesIO()
    np.save(image_io, np_array, allow_pickle=True)
    payload = image_io.getvalue()

    
    runtime_sm_client = boto3.client(service_name="sagemaker-runtime", region_name="us-east-1")
    endpoint_name = "yolov8-pytorch-2024-08-21-20-32-14-863267"
        
    response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload,
        ContentType="application/x-npy"
        )
    
    infer = json.load(response["Body"])
    
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }