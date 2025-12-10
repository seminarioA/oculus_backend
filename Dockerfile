# Usa una imagen base ligera de Python
FROM python:3.10-slim

# Evitar diálogos interactivos de apt
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# Instalación de dependencias del sistema
# build-essential: Para compilar librerías de Python/OpenCV
# libgl1/libglib2.0-0: Requerido por OpenCV y otras librerías de procesamiento de imágenes
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

# -----------------------------
# PyTorch CPU (Versión ligera compatible con MiDaS/YOLO)
# -----------------------------
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu

# -----------------------------
# Dependencias principales de la aplicación
# -----------------------------
RUN pip install --no-cache-dir \
    fastapi \
    # **CORRECCIÓN:** Forzar una versión estable conocida de uvicorn[standard]
    uvicorn[standard]==0.27.0 \
    # La versión 0.38.0 vista en su log puede tener problemas con la instalación de Python.
    opencv-python-headless \
    matplotlib \
    numpy==1.26.4 \
    ultralytics==8.3.34 \
    timm==0.9.16

# -----------------------------
# Clonar el repositorio de MiDaS
# torch.hub.load("intel-isl/MiDaS", ...) requiere el código fuente en el host.
# Se clona en un directorio temporal para que PyTorch lo encuentre.
# -----------------------------
RUN git clone https://github.com/isl-org/MiDaS.git /tmp/MiDaS && \
    cd /tmp/MiDaS && \
    # Se recomienda usar la rama principal o una etiqueta estable, 
    # pero si un commit específico es necesario, se debe asegurar que se puede acceder.
    # Usaremos el commit que proporcionaste (HEAD de master en el momento del commit)
    git checkout 454597711a62eabcbf7d1e89f3fb9f569051ac9b

# Copiar los archivos de la aplicación
# Asegúrate de que server.py y vision_processor.py estén en el mismo directorio.
COPY server.py vision_processor.py /app/

# Exponer el puerto de la API
EXPOSE 8000

# Comando para iniciar la aplicación Uvicorn/FastAPI
# El modo estándar de Uvicorn inicia un proceso único, lo cual es óptimo para el uso intensivo de CPU.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]