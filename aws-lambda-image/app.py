import base64
import boto3
import cv2
import glob
import json
import io
import os
import pytesseract
import numpy as np
import re
import shutil
import tempfile
import time


TEMP_DIR = tempfile.mkdtemp()   # Use this isntead of CROPS_DIR
SM_ENDPOINT = os.getenv("SM_ENDPOINT")
runtime_sm_client = boto3.client(service_name="sagemaker-runtime", region_name="us-east-1")
classes_mapper = {0: "item_description", 1: "store_address", 2: "store_name", 3: "total_amount", 4: "transaction_datetime"}
counters = {"item_description": 0, "store_address": 0, "store_name": 0, "total_amount": 0, "transaction_datetime": 0}


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


def pytesseract_text_extraction(image_path: str, lang: str = "eng+spa") -> str:
    """
    Extracts text from an image using pytesseract and cleans it by removing unwanted characters.

    :param image_path (str): Path to the image file.
    :param lang (str): Language(s) to be used by pytesseract for OCR. Default is "eng+spa".
    :return cleaned_text (str): The extracted and cleaned text.
    """
    # Perform OCR to extract text
    text = pytesseract.image_to_string(image_path, lang=lang)

    # Clean the text by removing unwanted characters
    cleaned_text = re.sub(r"[\x0c\n]", "", text)

    return cleaned_text


def handler(event, context):
    try:
        base64_image = event["body"]
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
            
        infer = json.load(response["Body"])

        infer_end_time = time.time()
        infer_time = infer_end_time - infer_start_time

        if "boxes" in infer:
            for (x1, y1, x2, y2, conf, lbl) in infer["boxes"]:
                # If confidence below 0.5, don't save the crop
                if conf < 0.5:
                    continue

                # Map lbl with class name
                lbl_name = classes_mapper.get(lbl)

                # Choose a color for the bounding box
                if lbl_name in counters:
                    counters[lbl_name] += 1
                    counter = counters[lbl_name]
                                            
                # Scale bounding box coordinates back to the original image size
                x1, x2 = int(x_ratio * x1), int(x_ratio * x2)
                y1, y2 = int(y_ratio * y1), int(y_ratio * y2)

                # Crop the image
                cropped_image = orig_image[y1:y2, x1:x2]

                # Create directory if it does not exist
                os.makedirs(f"{TEMP_DIR}/{lbl_name}", exist_ok=True)

                # Define the output file path
                output_path = f"{TEMP_DIR}/{lbl_name}/detection_{counter}.jpg"

                # Save the cropped image to the file
                cv2.imwrite(output_path, cropped_image)

        # From here OCR starts. Encapsulate this in a function??
        class_folders = glob.glob(f"{TEMP_DIR}/*")

        receipt_data = {}

        total_amount_pattern = r"(\d+[.,]?\d+)"
        items_pattern = r"^(\d*)\s*([A-Za-z\s.,/-ÁÉÍÓÚÑáéíóúñ]+?)(\d+[.,]?\d*)$"

        for class_folder in class_folders:
            class_label = class_folder.split("/")[-1]
            crops_list = glob.glob(f"{class_folder}/*")

            if class_label != "item_description":
                crop_path = crops_list[0] # If there is more than one, takes just the first one
                cleaned_text = pytesseract_text_extraction(crop_path)

                if class_label == "total_amount":
                    match = re.search(total_amount_pattern, cleaned_text)
                    if match:
                        cleaned_text = float(match.group(1).replace(",", "."))

                receipt_data[class_label] = cleaned_text

            else:
                items = []

                for crop_path in crops_list:
                    cleaned_text = pytesseract_text_extraction(crop_path)
                    match = re.match(items_pattern, cleaned_text)

                    if match:
                        quantity = int(match.group(1)) if match.group(1) else 1 # quantity is optional. Criteria is 1 in case no match
                        description = match.group(2).strip()  # strip any leading/trailing whitespace
                        price = float(match.group(3).replace(",", ".")) if match.group(3) else None # price is optional
                    else:
                        continue

                    item_dict = {
                        "quantity": quantity,
                        "description": description,
                        "price_eur": price
                    }

                    items.append(item_dict)

                receipt_data["items"] = items

    finally:
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)


    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(receipt_data),
        "isBase64Encoded": False
    }