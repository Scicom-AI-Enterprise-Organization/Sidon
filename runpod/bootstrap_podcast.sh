#!/usr/bin/env bash
# One-shot setup for a podcast-prep CPU pod: 7z + ffmpeg + a venv with the DNSMOS
# stack. Idempotent-ish.
set -u
export DEBIAN_FRONTEND=noninteractive
apt-get update -yq >/dev/null 2>&1
apt-get install -yq p7zip-full ffmpeg libsndfile1 >/dev/null 2>&1
test -d /Sidon/.venv_dnsmos || python3 -m venv /Sidon/.venv_dnsmos
/Sidon/.venv_dnsmos/bin/pip install -q --upgrade pip >/dev/null 2>&1
/Sidon/.venv_dnsmos/bin/pip install -q numpy soundfile soxr speechmos onnxruntime librosa \
    huggingface_hub hf_transfer hf_xet 2>&1 | tail -2
/Sidon/.venv_dnsmos/bin/python -c "import soxr,onnxruntime,huggingface_hub; from speechmos import dnsmos; print('venv ok')"
echo "BOOTSTRAP_DONE 7z=$(which 7z) ffmpeg=$(which ffmpeg) nproc=$(nproc)"
