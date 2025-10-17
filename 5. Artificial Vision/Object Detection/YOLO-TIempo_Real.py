# %% [markdown]
# Author:  
# Manuel Eugenio Morocho Cayamcela, PhD

# %% [markdown]
# # YOLO en Tiempo Real: Detección de Objetos
# 
# En este cuaderno de Jupyter exploraremos la tarea de la detección de objetos en tiempo real utilizando YOLO (You Only Look Once), una técnica avanzada de visión por computadora.
# 
# Aprenderemos a utilizar YOLO para identificar y localizar diversos objetos en imágenes y secuencias de video. Esta habilidad es esencial en una amplia gama de aplicaciones, desde la conducción autónoma hasta la vigilancia de seguridad, y representa uno de los avances más emocionantes en el campo de la inteligencia artificial y la visión por computadora.
# 
# A lo largo de este tutorial, explorarás conceptos clave como:
# 
# - Detección de objetos en tiempo real.
# - Configuración de modelos YOLO pre-entrenados.
# - Interpretación de resultados de detección.
# - Aplicaciones prácticas de la detección de objetos.

# %% [markdown]
# Utilizaremos el comando `!pip` para instalar dos paquetes de Python: `opencv-python` y `ultralytics`.
# 
# - `opencv-python`: Es un paquete para tareas de visión por computador en Python. Proporciona una variedad de funciones para el procesamiento de imágenes y videos, incluyendo detección de objetos, manipulación de imágenes y extracción de características.
# 
# - `ultralytics`: Es un paquete construido sobre PyTorch, que se enfoca principalmente en tareas de visión por computadora como detección de objetos y clasificación de imágenes. Proporciona interfaces fáciles de usar para entrenar y evaluar modelos de aprendizaje profundo para estas tareas.
# 
# Al ejecutar este comando, estás instalando estos paquetes en tu entorno de Python para que puedas usarlos en tu código.

# %%
# Instalamos las librerías necesarias
#%pip install opencv-python ultralytics

# %% [markdown]
# Importamos las siguientes bibliotecas en Python:
# 
# 1. `ultralytics`: Esta es una biblioteca que proporciona una interfaz para usar modelos de detección de objetos YOLO (You Only Look Once). Permite entrenar, evaluar y utilizar modelos de detección de objetos de manera eficiente.
# 
# 2. `cv2` (OpenCV): Esta es una biblioteca popular para el procesamiento de imágenes y videos en Python. Proporciona una amplia gama de funciones para trabajar con imágenes y videos, incluyendo cargar imágenes, realizar operaciones de procesamiento de imágenes, y mostrar imágenes en una ventana, entre otros.
# 
# 3. `math`: Este es un módulo estándar de Python que proporciona funciones matemáticas comunes, como funciones trigonométricas, logarítmicas y aritméticas.

# %%
# Importamos las librerías necesarias
from ultralytics import YOLO
import cv2
import math

# %% [markdown]
# Para este tutorial, usaremos un modelo de YOLO preentrenado en una base de datos que contiene imagenes con sus respectivas cordenadas de cuadros delimitadores y etiquetas por cada objeto. La base de datos con la que el modelo fue preentrenado es COCO de Microsoft.

# %%
# Cargamos el modelo YOLO pre-entrenado en COCO dataset
model = YOLO("yolo-Weights/yolov8n.pt")

# %%
# Revisamos las clases que puede detectar el modelo
model.names

# %%
# Definimos una lista de nombres con todas las clases para identificar objetos detectados
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ] # Aquí se enumeran todas las clases

# %%
# Configuramos la captura de video desde la cámara
captura = cv2.VideoCapture(0) # Se abre la cámara por defecto

# Establecemos el ancho y alto de la imagen
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Ancho de la imagen
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Alto de la imagen

# Iniciamos un bucle para procesar los fotogramas de la cámara
while True:
    success, img = captura.read() # Capturamos un fotograma

    # Realizamos la detección de objetos en la imagen capturada (usando el modelo de YOLO pre-entrenado que cargamos anteriormente)
    results = model(img, stream=True)

   # Procesamos los resultados de la detección
    for r in results:
        boxes = r.boxes

        # Iteramos sobre las cajas delimitadoras detectadas
        for box in boxes:
            # Obtenemos las coordenadas de la caja delimitadora
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convertimos a valores enteros

            # Dibujamos la caja delimitadora en la imagen
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

            # Obtenemos la confianza de la detección
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # Obtenemos el nombre de la clase detectada
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # Mostramos el nombre de la clase junto a la caja delimitadora
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0) # Color: Azul (formato BGR)
            thickness = 1
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Mostramos la imagen con las detecciones
    cv2.imshow('Webcam', img)

    # Salimos del bucle si se presiona la tecla 'q'
    #if cv2.waitKey(1) == ord('q'):
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberamos la cámara y cerramos todas las ventanas
captura.release()
cv2.destroyAllWindows()

# %% [markdown]
# # 🧠 Tarea: Despliegue del Sistema YOLOv8 en Jetson Nano 2GB
# 
# ## 📌 Objetivo
# 
# Implementar un sistema de detección en tiempo real utilizando YOLOv8 en una Jetson Nano 2GB conectada a la misma red local, con una cámara web USB.
# 
# ---
# 
# ## 🔐 Acceso a la Jetson Nano
# 
# Se les proporcionará:
# - Nombre de usuario
# - Dirección IP de la Jetson Nano
# - Contraseña
# 
# Usen la terminal de su computador para conectarse:
# 
# ```bash
# ssh usuario@ip_de_jetson

# %% [markdown]
# # 📂 Opción 1: Carrera de Matemática
# 
# Clonar su repositorio desde GitHub y ejecutar el sistema directamente en la Jetson Nano.
# 
# ✅ Pasos:
# 
# ### 1. Clonar el repositorio
# git clone https://github.com/usuario/repositorio.git
# 
# ### 2. Acceder al directorio del proyecto
# cd repositorio
# 
# ### 3. Crear entorno virtual e instalar dependencias
# python3 -m venv env
# source env/bin/activate
# pip install -r requirements.txt
# 
# ### 4. Ejecutar el sistema
# python YOLO-Tiempo-Real.py
# 

# %% [markdown]
# # 📦 Opción 2: Carrera de Ciencias Computacionales
# 
# Desplegar el sistema dentro de un contenedor Docker.
# 
# ✅ Crear un archivo Dockerfile en el repositorio:

# %% [markdown]
# ### Dockerfile
# FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.11-py3
# RUN apt-get update && apt-get install -y python3-pip libopencv-dev
# RUN pip3 install ultralytics opencv-python
# 
# COPY . /app
# WORKDIR /app
# 
# CMD ["python3", "YOLO-Tiempo-Real.py"]

# %% [markdown]
# ## 🐳 Pasos para construir y ejecutar:
# 
# ### 1. Clonar su repositorio
# git clone https://github.com/usuario/repositorio.git
# cd repositorio
# 
# ### 2. Construir la imagen Docker
# sudo docker build -t yolov8-detector .
# 
# ### 3. Ejecutar el contenedor
# sudo docker run --rm -it --net=host --device=/dev/video0 yolov8-detector

# %% [markdown]
# # 📝 Entregable
# 
# - Confirmación de funcionamiento con captura de pantalla o foto del sistema en ejecución.
# - Código actualizado y funcional en su repositorio de GitHub.


