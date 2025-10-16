KFD_GID=$(stat -c %g /dev/kfd)
RENDER_GID=$(stat -c %g /dev/dri/renderD128)

docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add ${KFD_GID} \
  --group-add ${RENDER_GID} \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -v $HOME:/workspace \
  -v $HOME/.cache/pip:/root/.cache/pip \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/BERTolto/checkpoints:/checkpoints \
  -e TOKENIZERS_PARALLELISM=false \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  -w /workspace/BERTolto \
  bertolto:rocm7 bash
