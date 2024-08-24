import base64
import boto3
import cv2
import json
import io
import os
import pytesseract
import numpy as np
import time


SM_ENDPOINT = os.getenv("SM_ENDPOINT")
runtime_sm_client = boto3.client(service_name="sagemaker-runtime", region_name="us-east-1")


def resize_and_encode(image: np.array, model_height: int  = 640, model_width: int = 320) -> tuple:
    """
    Resizes the given image to the specified dimensions and encodes it into JPEG format.
    Also returns the ratios of the original image dimensions to the model dimensions.

    :param image (np.array): The image to be processed as a NumPy array.
    :param model_height (int, optional): The desired height for the resized image. Defaults to 640.
    :param model_width (int, optional): The desired width for the resized image. Defaults to 320.
    :return: A tuple containing:
             - x_ratio (float): The ratio of the original image width to the model width.
             - y_ratio (float): The ratio of the original image height to the model height.
             - resized_image (np.array): The resized image as a NumPy array.
             - encoded_image (np.array): The JPEG-encoded image as a NumPy array.
    """
    image_height, image_width, _ = image.shape
    x_ratio = image_width / model_width
    y_ratio = image_height / model_height

    resized_image = cv2.resize(image, (model_width, model_height))
    _, encoded_image = cv2.imencode(".jpg", resized_image)
    
    return (x_ratio, y_ratio, resized_image, encoded_image)


def handler(event, context):
    base64_image = event["receipt_image"]
    base64_decoded_image = base64.b64decode(base64_image)

    np_array = np.frombuffer(base64_decoded_image, np.uint8)
    orig_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Dimensions of the original image in pixels
    orig_height, orig_width, _ = orig_image.shape
    print(f"Original dimensions: {orig_width}x{orig_height}")

    # Resize and encode the image
    x_ratio, y_ratio, resized_image, encoded_image = resize_and_encode(orig_image)

    image_io = io.BytesIO()
    np.save(image_io, encoded_image, allow_pickle=True)
    payload = image_io.getvalue()

    infer_start_time = time.time()
        
    response = runtime_sm_client.invoke_endpoint(
        EndpointName=SM_ENDPOINT,
        Body=payload,
        ContentType="application/x-npy"
        )
        
    result = json.load(response["Body"])

    infer_end_time = time.time()
    infer_time = infer_end_time - infer_start_time

    return {
        "statusCode": 200,
        "inferenceTime": infer_time,
        "body": json.dumps(result)
    }