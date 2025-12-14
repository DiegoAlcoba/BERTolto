#!/usr/bin/env bash
set -euo pipefail

# === Permisos GPU ===
KFD_GID=$(stat -c %g /dev/kfd)
RENDER_GID=$(stat -c %g /dev/dri/renderD128)

# === Rutas proyecto/cachés (host) ===
PROJ="$HOME/BERTolto"
HF_HOME_HOST="$PROJ/.hf_home"
PIP_CACHE_HOST="$PROJ/.cache/pip"

# Asegura que existen
mkdir -p "$HF_HOME_HOST"/{hub,transformers,datasets}
mkdir -p "$PIP_CACHE_HOST"

# === Rutas dentro del contenedor ===
WS="/workspace"                 # raíz de trabajo en container
PROJ_IN="$WS/BERTolto"
HF_HOME_IN="$PROJ_IN/.hf_home"
PIP_CACHE_IN="$PROJ_IN/.cache/pip"

docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add ${KFD_GID} \
  --group-add ${RENDER_GID} \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -v "$HOME":"$WS" \
  -v "$PIP_CACHE_HOST":"$PIP_CACHE_IN" \
  -v "$HOME/BERTolto/checkpoints":/checkpoints \
  -w "$PROJ_IN" \
  -e HF_HOME="$HF_HOME_IN" \
  -e HUGGINGFACE_HUB_CACHE="$HF_HOME_IN/hub" \
  -e TRANSFORMERS_CACHE="$HF_HOME_IN/transformers" \
  -e HF_DATASETS_CACHE="$HF_HOME_IN/datasets" \
  -e PIP_CACHE_DIR="$PIP_CACHE_IN" \
  -e TOKENIZERS_PARALLELISM=false \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  bertolto:rocm7 bash

