#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for the Sidon restoration dry-run finetune.
# Runs as root on the pod. Everything lives under / (the container disk),
# NEVER /workspace (the network volume). Idempotent enough to re-run.
#
# We build a TARGETED venv rather than `uv sync`: the full pyproject pulls
# flash-attn / mmdit / flow-matching / diffusers / ray, which are only needed
# for the GENESES *dialogue* path (out of scope here) and flash-attn's source
# build is slow/fragile. The single-speaker restoration pipeline (feature
# predictor + DAC vocoder) only needs the subset installed below.
set -euo pipefail

REPO=/Sidon
VENV=$REPO/.venv
export DEBIAN_FRONTEND=noninteractive

echo "===== [bootstrap] system packages ====="
apt-get update -y
apt-get install -y --no-install-recommends \
    ffmpeg unzip p7zip-full git rsync curl wget ca-certificates aria2 || true

# Static 7zz (matches the dataset READMEs; handles multi-volume zips reliably).
if ! command -v 7zz >/dev/null 2>&1; then
    echo "[bootstrap] installing 7zz static binary"
    cd /tmp
    wget -q https://www.7-zip.org/a/7z2301-linux-x64.tar.xz -O 7z.tar.xz && \
        tar -xf 7z.tar.xz 7zz && mv 7zz /usr/local/bin/ && chmod +x /usr/local/bin/7zz || \
        echo "[bootstrap] 7zz download failed; will fall back to p7zip's 7z"
fi

echo "===== [bootstrap] uv ====="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# Sidon venv. Install torch/torchaudio FIRST from the cu128 index (matches the
# base image's CUDA), THEN everything else. Doing torch in a separate step keeps
# the pytorch index from polluting the resolution of pure-PyPI packages. Python
# 3.10 to match the repo's .python-version.
# ---------------------------------------------------------------------------
echo "===== [bootstrap] Sidon venv ($VENV) ====="
uv venv "$VENV" --python 3.10
uv pip install --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchaudio==2.8.0

# descript-audio-codec brings `dac` + `audiotools` (descript-audiotools), which
# sidon.model imports unconditionally. transformers/peft/lightning drive the
# w2v-BERT + LoRA student and the Trainer. pyroomacoustics powers the online RIR
# degradation. The HF stack + soundfile/soxr/librosa power data prep + loading.
uv pip install --python "$VENV/bin/python" \
    "numpy<2" scipy einops tqdm \
    "descript-audio-codec>=1.0.0" \
    "transformers>=4.56.1" "peft>=0.16.0" "lightning>=2.5.5" \
    hydra-core omegaconf \
    "webdataset>=0.2.100" \
    pyroomacoustics \
    soundfile soxr librosa mutagen \
    "huggingface_hub>=0.34" hf_transfer hf_xet \
    wandb

echo "[bootstrap] torch/CUDA + key imports check:"
"$VENV/bin/python" - <<'PY'
import torch, torchaudio, transformers, peft, lightning, webdataset
import dac, audiotools, pyroomacoustics  # the imports sidon.model needs
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("transformers", transformers.__version__, "peft", peft.__version__, "lightning", lightning.__version__)
print("dac+audiotools+pyroomacoustics import OK")
PY

# Pre-warm the w2v-BERT 2.0 feature extractor + config into /hf_cache so the
# first training step doesn't pay the download mid-run (best-effort).
echo "===== [bootstrap] pre-warming facebook/w2v-bert-2.0 ====="
set -a; [ -f "$REPO/.env" ] && source "$REPO/.env"; set +a
HF_HOME=/hf_cache "$VENV/bin/python" - <<'PY' || echo "[bootstrap] w2v-bert prewarm failed (will download on demand)"
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", num_hidden_layers=8)
print("w2v-bert-2.0 prewarm OK")
PY

echo "===== [bootstrap] done ====="
