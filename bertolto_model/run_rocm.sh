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
  -v $HOME/BERTolto/checkpoints:/checkpoints \
  -w /workspace/BERTolto \
  -e HF_HOME=/workspace/BERTolto/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/workspace/BERTolto/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/BERTolto/.cache/huggingface/hub \
  -e TOKENIZERS_PARALLELISM=false \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  bertolto:rocm7 bash
