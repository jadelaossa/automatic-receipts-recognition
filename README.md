# Sistema de conversión de tickets de compra a texto

## Descripción

Este proyecto sirve como trabajo de fin de máster del máster en [Ciencia de Datos e Ingeniería de Datos en la Nube de la Universidad de Castilla-La Mancha](http://www.cidaen.es/). El objetivo consiste en desarrollar una aplicación que, mediante la imagen de un ticket de compra sea capaz de convertir la información útil en datos de texto estructurados. Para lograr esto, se combina un algoritmo de detección de objectos, junto con un algoritmo de reconocimiento óptico de caracteres (OCR).

El proyecto se divide en dos fases principales:

1. **Detección de Objetos con YOLOv8**: Se entrena el modelo YOLOv8 de Ultralytics utilizando un conjunto de datos de tickets de compra. El modelo está diseñado para identificar y extraer cinco clases esenciales: Nombre del comercio, dirección del comercio, fecha de facturación, artículos de compra, y la cantidad total.

2. **Reconocimiento Óptico de Caracteres (OCR) con Tesseract**: Una vez que los elementos clave han sido identificados y extraídos de la imagen, se utiliza Tesseract para realizar OCR y convertir los datos visuales en texto editable.

## Instalación

1. Clona este repositorio en tu máquina local:
   ```bash
   git clone https://github.com/jadelaossa/automatic-receipts-recognition.git
   ```

2. Navega al directorio del proyecto:
   ```bash
   cd automatic-receipts-recognition
   ```

3. Instala los paquetes necesarios:
   ```bash
   pip install -r requirements.txt
   ```

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## Contacto

Para cualquier pregunta, puedes contactarme a través de javdelfer@gmail.com.