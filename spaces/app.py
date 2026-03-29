#!/usr/bin/env python3
"""DialogueSidon — two-speaker dialogue separation demo.

Loads exported torch.export components from sarulab-speech/DialogueSidon on
Hugging Face Hub and runs diffusion-based speaker separation.
Inputs up to 120 s are processed in one shot; longer inputs use chunked
streaming inference with crossfade stitching and speaker re-alignment.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

import numpy as np
import torch
import torchaudio
import gradio as gr
from diffusers import DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ID = "sarulab-speech/DialogueSidon"
MODEL_FILES = ["ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2", "metadata.json"]
SAMPLE_RATE_IN = 16_000
CHUNK_SECONDS = 120.0
OVERLAP_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Feature extraction (inline, no sidon src dependency)
# ---------------------------------------------------------------------------

def _pad_batch(
    features: list[torch.Tensor],
    pad_to_multiple_of: int = 2,
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_length = max(f.shape[0] for f in features)
    if pad_to_multiple_of:
        target_length = (
            (target_length + pad_to_multiple_of - 1)
            // pad_to_multiple_of
            * pad_to_multiple_of
        )
    batch_size = len(features)
    feature_dim = features[0].shape[1]
    device = features[0].device
    padded = torch.full(
        (batch_size, target_length, feature_dim),
        padding_value,
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros((batch_size, target_length), dtype=torch.int64, device=device)
    for i, feat in enumerate(features):
        padded[i, : feat.shape[0]] = feat
        mask[i, : feat.shape[0]] = 1
    return padded, mask


def extract_fbank_features(
    waveforms: list[torch.Tensor],
    device: torch.device,
    num_mel_bins: int = 80,
    stride: int = 2,
) -> dict[str, torch.Tensor]:
    features = []
    for wav in waveforms:
        if wav.ndim > 1:
            wav = wav[0]
        feat = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0),
            sample_frequency=SAMPLE_RATE_IN,
            num_mel_bins=num_mel_bins,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            preemphasis_coefficient=0.97,
            remove_dc_offset=True,
            window_type="povey",
            use_energy=False,
            energy_floor=1.192092955078125e-07,
        )
        mean = feat.mean(0, keepdim=True)
        var = feat.var(0, keepdim=True)
        feat = (feat - mean) / torch.sqrt(var + 1e-5)
        features.append(feat.to(device))

    input_features, attention_mask = _pad_batch(features)
    b, t, c = input_features.shape
    t = (t // stride) * stride
    input_features = input_features[:, :t, :]
    attention_mask = attention_mask[:, :t]
    input_features = input_features.reshape(b, t // stride, c * stride)
    attention_mask = attention_mask[:, 1::stride]
    return {"input_features": input_features, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

_cache: dict = {}


def load_models(device: torch.device) -> dict:
    cache_key = str(device)
    if cache_key in _cache:
        return _cache[cache_key]

    print(f"Downloading model files from {REPO_ID} ...")
    paths = {f: hf_hub_download(repo_id=REPO_ID, filename=f, token=HF_TOKEN) for f in MODEL_FILES}

    with open(paths["metadata.json"]) as fp:
        meta = json.load(fp)

    ssl_encoder = torch.export.load(paths["ssl_encoder.pt2"]).module().to(device)
    diffusion_head = torch.export.load(paths["diffusion_head.pt2"]).module().to(device)
    vae_decoder = torch.export.load(paths["vae_decoder.pt2"]).module().to(device)

    latent_norm_mean = torch.tensor(
        meta["latent_norm_mean"], dtype=torch.float32, device=device
    ).view(1, 1, -1)
    latent_norm_std = torch.tensor(
        meta["latent_norm_std"], dtype=torch.float32, device=device
    ).view(1, 1, -1)

    scheduler = DPMSolverMultistepScheduler.from_config(
        meta["ddpm_config"],
        algorithm_type="dpmsolver++",
        timestep_spacing="linspace",
    )

    models = {
        "ssl_encoder": ssl_encoder,
        "diffusion_head": diffusion_head,
        "vae_decoder": vae_decoder,
        "latent_norm_mean": latent_norm_mean,
        "latent_norm_std": latent_norm_std,
        "latent_norm_initialized": meta["latent_norm_initialized"],
        "scheduler": scheduler,
        "latent_dim": meta["latent_dim"],
        "sample_rate": meta["sample_rate"],
    }
    _cache[cache_key] = models
    return models


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _normalize(latents: torch.Tensor, models: dict) -> torch.Tensor:
    if not models["latent_norm_initialized"]:
        return latents
    return ((latents.float() - models["latent_norm_mean"]) / models["latent_norm_std"]).to(latents.dtype)


def _denormalize(latents: torch.Tensor, models: dict) -> torch.Tensor:
    if not models["latent_norm_initialized"]:
        return latents
    return (latents.float() * models["latent_norm_std"] + models["latent_norm_mean"]).to(latents.dtype)


@torch.inference_mode()
def _separate_chunk(
    wav: torch.Tensor,   # [1, T] at 16 kHz, already normalized
    num_steps: int,
    models: dict,
    device: torch.device,
) -> torch.Tensor:
    """Run separation on a single chunk. Returns [2, T_audio] at model sample rate."""
    latent_dim = models["latent_dim"]

    noisy_ssl = extract_fbank_features([wav.view(-1)], device)
    features, pred0, pred1 = models["ssl_encoder"](
        noisy_ssl["input_features"], noisy_ssl["attention_mask"]
    )

    predicted_latents = torch.cat([pred0, pred1], dim=-1)
    conditioning = torch.cat([_normalize(predicted_latents, models), features], dim=-1)

    seq_len = conditioning.shape[1]
    scheduler = models["scheduler"]
    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(
        (1, seq_len, latent_dim * 2), device=device, dtype=conditioning.dtype
    )
    for t in scheduler.timesteps:
        t_batch = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
        latents = scheduler.step(
            models["diffusion_head"](latents, t_batch, conditioning), t, latents
        ).prev_sample

    latents = _denormalize(latents, models)
    spk1 = models["vae_decoder"](latents[:, :, :latent_dim].transpose(1, 2)).squeeze(0)  # [1, T]
    spk2 = models["vae_decoder"](latents[:, :, latent_dim:].transpose(1, 2)).squeeze(0)
    return torch.cat([spk1, spk2], dim=0)  # [2, T]


def _channel_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    return float(torch.dot(a, b) / denom) if float(denom) > 1e-8 else 0.0


def _maybe_swap(
    prev_overlap: torch.Tensor, curr_chunk: torch.Tensor, overlap_samples: int
) -> tuple[torch.Tensor, bool]:
    if overlap_samples <= 0 or prev_overlap.shape[0] != 2 or curr_chunk.shape[0] != 2:
        return curr_chunk, False
    curr_ov = curr_chunk[:, :overlap_samples]
    direct = _channel_similarity(prev_overlap[0], curr_ov[0]) + _channel_similarity(prev_overlap[1], curr_ov[1])
    swapped = _channel_similarity(prev_overlap[0], curr_ov[1]) + _channel_similarity(prev_overlap[1], curr_ov[0])
    if swapped > direct:
        return curr_chunk[[1, 0], :], True
    return curr_chunk, False


def separate(
    wav: torch.Tensor,   # [1, T] at original sample_rate
    sample_rate: int,
    num_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    models = load_models(device)
    out_sr = models["sample_rate"]

    # resample to 16 kHz
    if sample_rate != SAMPLE_RATE_IN:
        wav_16k = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE_IN)
    else:
        wav_16k = wav
    wav_16k = wav_16k.to(device)

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE_IN)
    total_samples = wav_16k.shape[-1]

    if total_samples <= chunk_samples:
        # single-shot inference
        max_val = wav_16k.abs().max().clamp_min(1e-6)
        wav_norm = torch.nn.functional.pad(0.9 * wav_16k / max_val, (160, 160))
        separated = _separate_chunk(wav_norm, num_steps, models, device)
        return separated, out_sr

    # chunked streaming inference
    overlap_samples_in = int(OVERLAP_SECONDS * SAMPLE_RATE_IN)
    hop_samples = chunk_samples - overlap_samples_in
    starts = list(range(0, total_samples, hop_samples))

    stitched: torch.Tensor | None = None
    prev_end_in = 0

    for idx, start in enumerate(starts):
        end = min(start + chunk_samples, total_samples)
        chunk = wav_16k[:, start:end]
        max_val = chunk.abs().max().clamp_min(1e-6)
        chunk_norm = torch.nn.functional.pad(0.9 * chunk / max_val, (160, 160))

        pred = _separate_chunk(chunk_norm, num_steps, models, device)  # [2, T_out]

        # match output length to input length (resampling ratio)
        target_out = max(1, round((end - start) * out_sr / SAMPLE_RATE_IN))
        if pred.shape[-1] > target_out:
            pred = pred[:, :target_out]
        elif pred.shape[-1] < target_out:
            pad = torch.zeros(2, target_out - pred.shape[-1], device=device)
            pred = torch.cat([pred, pad], dim=-1)

        if stitched is None:
            stitched = pred
            prev_end_in = end
            continue

        overlap_in = max(0, prev_end_in - start)
        overlap_out = max(0, min(
            round(overlap_in * out_sr / SAMPLE_RATE_IN),
            stitched.shape[-1],
            pred.shape[-1],
        ))

        if overlap_out > 0:
            pred, _ = _maybe_swap(stitched[:, -overlap_out:], pred, overlap_out)
            fade = torch.linspace(0.0, 1.0, overlap_out, device=device).unsqueeze(0)
            blended = stitched[:, -overlap_out:] * (1 - fade) + pred[:, :overlap_out] * fade
            stitched = torch.cat([stitched[:, :-overlap_out], blended, pred[:, overlap_out:]], dim=-1)
        else:
            stitched = torch.cat([stitched, pred], dim=-1)

        prev_end_in = end

    return stitched, out_sr


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def extract_audio_from_video(video_path: str) -> tuple[torch.Tensor, int]:
    """Extract mono audio from a video file using ffmpeg. Returns (wav [1,T], sr)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(SAMPLE_RATE_IN),
         "-vn", tmp_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wav, sr = torchaudio.load(tmp_path)
    os.unlink(tmp_path)
    return wav, sr


def create_stereo_video(
    video_path: str,
    spk1: np.ndarray,
    spk2: np.ndarray,
    out_sr: int,
) -> str:
    """Mux separated speakers (L=spk1, R=spk2) back into a video file.

    Returns path to the output video (caller is responsible for cleanup).
    """
    # Write stereo audio to a temp wav
    stereo = np.stack([spk1, spk2], axis=0)  # [2, T]
    stereo_tensor = torch.from_numpy(stereo).float()
    # Normalise to avoid clipping
    peak = stereo_tensor.abs().max().clamp_min(1e-6)
    stereo_tensor = stereo_tensor / peak * 0.9

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        audio_path = tmp_audio.name
    torchaudio.save(audio_path, stereo_tensor, out_sr)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        out_path = tmp_video.name

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,       # original video (video stream)
            "-i", audio_path,       # new stereo audio
            "-c:v", "copy",         # copy video stream unchanged
            "-c:a", "aac",          # encode audio as AAC
            "-b:a", "192k",
            "-map", "0:v:0",        # take video from first input
            "-map", "1:a:0",        # take audio from second input
            "-shortest",
            out_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.unlink(audio_path)
    return out_path


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _wav_to_numpy_output(wav_tensor: torch.Tensor, sr: int) -> tuple[int, np.ndarray]:
    arr = wav_tensor.cpu().numpy()
    # Convert to int16 for Gradio audio output
    arr = np.clip(arr / max(np.abs(arr).max(), 1e-6) * 0.9, -1.0, 1.0)
    return sr, (arr * 32767).astype(np.int16)


def run_separation_audio(
    input_audio: tuple[int, np.ndarray] | None,
    num_steps: int,
) -> tuple[tuple[int, np.ndarray], tuple[int, np.ndarray]]:
    if input_audio is None:
        raise gr.Error("Please upload an audio file.")

    sr, audio_np = input_audio
    wav = torch.from_numpy(audio_np.copy()).float()

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 2:
        if wav.shape[1] <= 8:
            wav = wav.T
        wav = wav.mean(dim=0, keepdim=True)

    if audio_np.dtype in (np.int16, np.int32):
        wav = wav / float(np.iinfo(audio_np.dtype).max)

    device = get_device()
    separated, out_sr = separate(wav, sr, num_steps, device)

    return (
        _wav_to_numpy_output(separated[0], out_sr),
        _wav_to_numpy_output(separated[1], out_sr),
    )


def run_separation_video(
    video_path: str | None,
    num_steps: int,
) -> tuple[str | None, tuple[int, np.ndarray], tuple[int, np.ndarray]]:
    if video_path is None:
        raise gr.Error("Please upload a video file.")

    wav, sr = extract_audio_from_video(video_path)
    device = get_device()
    separated, out_sr = separate(wav, sr, num_steps, device)

    spk1_np = separated[0].cpu().numpy()
    spk2_np = separated[1].cpu().numpy()

    out_video = create_stereo_video(video_path, spk1_np, spk2_np, out_sr)

    return (
        out_video,
        _wav_to_numpy_output(separated[0], out_sr),
        _wav_to_numpy_output(separated[1], out_sr),
    )


if HAS_SPACES:
    run_separation_audio = spaces.GPU(run_separation_audio)
    run_separation_video = spaces.GPU(run_separation_video)


with gr.Blocks(title="DialogueSidon — Dialogue Separation") as demo:
    gr.Markdown(
        """
        # DialogueSidon — Two-Speaker Dialogue Separation & Restoration
        Upload a degraded or noisy audio/video recording of a two-speaker conversation.
        DialogueSidon jointly **separates** the two speakers and **restores** clean, high-quality speech
        from the mixture — handling background noise, reverberation, and channel degradation in one pass.

        Inputs up to 120 s are processed in one shot. Longer inputs are processed in 120 s chunks
        with 10 s overlap crossfade and automatic speaker re-alignment across chunks.

        **Model**: [sarulab-speech/DialogueSidon](https://huggingface.co/sarulab-speech/DialogueSidon)
        """
    )

    num_steps = gr.Slider(
        minimum=10, maximum=100, value=30, step=5,
        label="Diffusion steps (more = slower but potentially better)",
    )

    with gr.Tabs():
        # ------------------------------------------------------------------
        # Audio tab
        # ------------------------------------------------------------------
        with gr.Tab("Audio"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="Input mixture (mono or stereo, any sample rate)",
                        type="numpy",
                    )
                    audio_btn = gr.Button("Separate", variant="primary")

                with gr.Column():
                    audio_spk1 = gr.Audio(label="Speaker 1", type="numpy")
                    audio_spk2 = gr.Audio(label="Speaker 2", type="numpy")

            audio_btn.click(
                fn=run_separation_audio,
                inputs=[audio_input, num_steps],
                outputs=[audio_spk1, audio_spk2],
            )

        # ------------------------------------------------------------------
        # Video tab
        # ------------------------------------------------------------------
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(
                        label="Input video (audio will be extracted and separated)",
                    )
                    video_btn = gr.Button("Separate", variant="primary")

                with gr.Column():
                    video_output = gr.Video(
                        label="Output video (Speaker 1 = Left, Speaker 2 = Right)",
                    )
                    video_spk1 = gr.Audio(label="Speaker 1 (audio only)", type="numpy")
                    video_spk2 = gr.Audio(label="Speaker 2 (audio only)", type="numpy")

            video_btn.click(
                fn=run_separation_video,
                inputs=[video_input, num_steps],
                outputs=[video_output, video_spk1, video_spk2],
            )

    gr.Markdown("---\n**License**: CC-BY-NC 4.0 — non-commercial use only.")


if __name__ == "__main__":
    demo.launch()
