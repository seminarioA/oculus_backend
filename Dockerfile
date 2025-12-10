FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Dependencias de sistema para OpenCV y HEIF
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libheif1 libde265-0 libheif-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi uvicorn[standard] \
        opencv-python-headless \
        matplotlib \
        numpy==1.26.4 \
        ultralytics==8.3.34 \
        git+https://github.com/apple/ml-depth-pro.git

# Descargar checkpoint (1.8 GB)
RUN mkdir -p /app/checkpoints && \
    wget -c https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -O /app/checkpoints/depth_pro.pt

# Copia tu código
COPY server.py vision_processor.py /app/

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]