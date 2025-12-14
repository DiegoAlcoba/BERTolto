# Dockerfile
FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.7.1

# Entorno útil
ENV HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Trabajaremos dentro del repo montado como volumen
WORKDIR /workspace/BERTolto

# Paquetes base útiles (tini para señalización limpia si se llega a usar)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

# Asegura pip moderno del venv preinstalado en la imagen ROCm
RUN /opt/venv/bin/python -m pip install --upgrade pip

# Copiamos sólo requirements para aprovechar cache de capas
COPY requirements.txt /tmp/requirements.txt
# (si se quiere usar -lock al final -> no se hace) COPY requirements-lock.txt /tmp/requirements-lock.txt

# IMPORTANTÍSIMO:
# - NO instalar torch/torchvision/torchaudio aquí
# - requirements.txt no debe incluirlos.
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt \
 && /opt/venv/bin/pip check || true

# Notebooks opcionales
RUN /opt/venv/bin/pip install jupyterlab ipykernel ipywidgets jupyterlab_widgets \
 && /opt/venv/bin/python -m ipykernel install --name bertolto --display-name "Python (bertolto)" --prefix=/usr/local

# Directorios útiles
RUN mkdir -p /checkpoints /datasets

# (Opcional) sanity check en build para ver torch+HIP (no falla el build si no hay GPU en build host)
RUN /opt/venv/bin/python - <<'PY' || true
import torch, sys
print("torch:", torch.__version__, "HIP:", getattr(torch.version, "hip", None))
print("Torch CUDA available? ", torch.cuda.is_available())
PY

# Si vas a usar Jupyter desde navegador local -> no
#EXPOSE 8888

# Arranca en primer plano con bash
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
