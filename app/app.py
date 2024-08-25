import base64
from io import BytesIO
import json
from PIL import Image
import requests
import streamlit as st


api_endpoint = "https://vv25y2lt0e.execute-api.us-east-1.amazonaws.com/dev/receipt"


def image_to_base64(image: Image) -> str:
    """
    Converts a PIL image to a base64 encoded string.

    :param image (Image): The PIL Image object to be encoded.
    :return base64_image (str): The base64 encoded string of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode()
    return base64_image


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
        base64_image = image_to_base64(image)

        with st.expander("Show/Hide Uploaded Image"):
            # st.image(image, caption="Uploaded Image", use_column_width=False, width=400)
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <img src='data:image/png;base64,{base64_image}' alt='Uploaded Image' style='width: 400px;'>
                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button("Read!"):
            with st.spinner("Processing..."):
                try:   
                    headers = {
                        "Content-Type": "application/json"
                    }

                    response = requests.post(api_endpoint, data=base64_image, headers=headers)
                    receipt_data = response.json()

                    print(f"Status Code: {response.status_code}")
                    print("Response Body: ")
                    print(response.json())

                except requests.RequestException as e:
                    st.error(f"An error occurred: {e}")

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
   

if __name__ == "__main__":
    main()