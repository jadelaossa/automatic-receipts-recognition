import base64
import glob
from io import BytesIO
import json
from PIL import Image
import os
import pytesseract
import re
import shutil
import streamlit as st
from ultralytics import YOLO


CROPS_DIR = "./st-crops"

def image_to_base64(image: Image) -> str:
    """
    Converts a PIL image to a base64 encoded string.

    :param image (Image): The PIL Image object to be encoded.
    :return img_str (str): The base64 encoded string of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

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

@st.cache_data
def load_model(model_dir: str) -> YOLO:
    """
    Loads a YOLO model from the specified directory.

    :param model_dir (str): Path to the directory containing the YOLO model.
    :return model (YOLO): The loaded YOLO model.
    """

    model = YOLO(model_dir)
    return model

# 1. Cargar el modelo
model = load_model("./runs/detect/train7/weights/best.pt")

def main():
    
    st.title("🧾 Readceipt")
    st.markdown('*"Turning your messy receipts into clean data, one item at a time!"*')
    st.markdown("<br>", unsafe_allow_html=True)  # Insert a line break

    # Introduction and instructions
    st.markdown("""
    ### Welcome to Readceipt!
    
    **Readceipt** is an app that leverages state-of-the-art machine learning models to automatically extract and organize information from your receipt images.
    
    **How it works:**
    - **Object Detection:** We use a pre-trained YOLOv8 model to detect key components on your receipt such as total amount, items, and more.
    - **Optical Character Recognition (OCR):** Once the key components are detected, we use Tesseract OCR to extract the text from these regions.
    - **Structured Output:** The extracted information is then organized into a structured format for easy viewing.

    **Instructions:**
    1. **Upload** your receipt image in JPG, JPEG, or PNG format.
    2. Click the **"Read!"** button to start the processing.
    3. After a few moments, the extracted data will be displayed in a structured format below.
    4. Use the **"Download Receipt Data as JSON"** button to save the results to your computer.

    **Note:** The app will automatically clean up any temporary files after the process is completed.
    """)

    st.markdown("---")
    
    # 2. Subir un imagen
    uploaded_file = st.file_uploader("Choose a receipt file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_str = image_to_base64(image)

        with st.expander("Show/Hide Uploaded Image"):
            # st.image(image, caption="Uploaded Image", use_column_width=False, width=400)
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <img src='data:image/png;base64,{img_str}' alt='Uploaded Image' style='width: 400px;'>
                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button("Read!"):
            with st.spinner("Processing..."):
                # 3. Aplicar inferencia y guardar resultados
                results = model(image)

                for result in results:
                    result_path = result.path.split("/")[-1]
                    result.save_crop(save_dir=f"./{CROPS_DIR}/{result_path}", file_name=f"detection")

                # 4. Aplicar OCR a los resultados
                classes_folders = glob.glob(f"{CROPS_DIR}/*")

                receipt_data = {}

                total_amount_pattern = r"(\d+[.,]?\d+)"
                items_pattern = r"^(\d*)\s*([A-Za-z\s.,/-ÁÉÍÓÚÑáéíóúñ]+?)(\d+[.,]?\d*)$"

                for class_folder in classes_folders:
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

                # 5. Devolver e imprimir por pantalla datos estructurados
                st.write(receipt_data)

                # 6. Descarga resultados en formato .json
                json_data = json.dumps(receipt_data, indent=4)
                json_filename = f"{uploaded_file.name.split('.')[0]}.json"

                st.download_button(
                    label="Download Receipt Data as JSON",
                    data=json_data,
                    file_name=json_filename,
                    mime="application/json"
                )

                # 7. Borrar directorio con crops después de finalizar OCR
                if os.path.exists(CROPS_DIR):
                    shutil.rmtree(CROPS_DIR)
   

if __name__ == "__main__":
    main()
