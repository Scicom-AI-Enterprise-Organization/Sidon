#!/usr/bin/env bash
# Bootstrap a RunPod H100 pod for the Sidon call-centre FE finetune. Runs as root.
# Everything under / (container disk), never /workspace. Targeted venv (the manual
# trainer needs only torch+transformers+peft+audio; no Lightning/flash-attn/dac).
set -euo pipefail
REPO=/Sidon
VENV=$REPO/.venv
export DEBIAN_FRONTEND=noninteractive

echo "===== [bootstrap_fe] apt ====="
apt-get update -y
apt-get install -y --no-install-recommends ffmpeg libsndfile1 unzip git rsync curl ca-certificates || true

echo "===== [bootstrap_fe] uv ====="
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "===== [bootstrap_fe] venv (py3.10, torch cu128) ====="
uv venv "$VENV" --python 3.10
uv pip install --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
uv pip install --python "$VENV/bin/python" \
    "transformers>=4.56.1" "peft>=0.16.0" \
    "datasets>=2.17" soundfile librosa numpy tqdm wandb \
    "huggingface_hub>=0.34" hf_transfer hf_xet

echo "[bootstrap_fe] import check:"
"$VENV/bin/python" - <<'PY'
import torch, torchaudio, transformers, peft, soundfile, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("transformers", transformers.__version__, "peft", peft.__version__)
PY

echo "===== [bootstrap_fe] prewarm w2v-bert-2.0 (24L) ====="
set -a; [ -f "$REPO/.env" ] && source "$REPO/.env"; set +a
HF_HOME=/hf_cache "$VENV/bin/python" - <<'PY' || echo "[bootstrap_fe] prewarm failed (downloads on demand)"
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", num_hidden_layers=24)
print("prewarm OK")
PY
echo "===== [bootstrap_fe] done ====="
