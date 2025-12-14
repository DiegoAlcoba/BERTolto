FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

# Dependencias del sistema mínimas (para wheels y certs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Paquetes Python necesarios para INFERENCIA
# - torch CPU, transformers, fastapi, uvicorn, pydantic v2, numpy estable, safetensors
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    safetensors==0.4.3 \
	--extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1+cpu

#Resto de deps
RUN pip install --no-cache-dir \
    transformers==4.44.2 \
    fastapi==0.115.0 \
    uvicorn==0.30.5 \
    pydantic==2.8.2

#Se copia solo el server, el modelo se monta como volumen
COPY app.py infer.py ./
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host","0.0.0.0","--port","8000"]
