"""Dialogue Sidon inference with chunked processing and latent-based permutation solving."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import torch.nn.functional as F
import torchaudio

from sidon.model.dialogue_sidion.lightning_module import (
    DialogueSidonDiffusionLightningModule,
)
from sidon.model.dialogue_sidion.audio import extract_seamless_m4t_features

# ---------------------------------------------------------------------------
# Checkpoint / IO helpers
# ---------------------------------------------------------------------------

def resolve_checkpoint(path: Path) -> Path:
    if path.is_file() and path.suffix == ".ckpt":
        return path
    if path.is_dir():
        checkpoints = sorted(path.rglob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No .ckpt found under: {path}")
        return checkpoints[-1]
    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def collect_wavs(input_dir: Path) -> list[Path]:
    wavs = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in: {input_dir}")
    return wavs


def normalize_to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim != 2:
        raise ValueError("Expected waveform with shape (channels, time) or (time,)")
    wav = wav.mean(dim=0, keepdim=True)
    max_val = wav.abs().max()
    if max_val > 0:
        wav = wav / max_val * 0.9
    return wav


def mux_video_with_audio(input_video: Path, input_wav: Path, output_video: Path) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg is required for --output-video but was not found in PATH.")
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(input_video),
        "-i", str(input_wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(output_video),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-2000:]}")


# ---------------------------------------------------------------------------
# Model forward: latents only, and separate decode
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_latents(
    model: DialogueSidonDiffusionLightningModule,
    wav: torch.Tensor,
    sample_rate: int,
    num_steps: int,
) -> torch.Tensor:
    """Run one chunk through the model and return sampled latents [T_lat, D*2].

    No VAE decoding is performed here.
    """
    device = model.device
    wav = torch.as_tensor(wav, dtype=torch.float32, device=device)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 2 and wav.shape[0] <= 2 and wav.shape[1] > 2:
        wav = wav.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        wav_16k = torchaudio.functional.resample(wav, sample_rate, 16_000)
    else:
        wav_16k = wav

    def _normalize_and_pad(signal: torch.Tensor) -> torch.Tensor:
        max_val = signal.abs().max().clamp_min(1e-6)
        return F.pad(0.9 * signal / max_val, (160, 160))

    wav_list = [_normalize_and_pad(s.view(-1)) for s in wav_16k]
    noisy_ssl_inputs = extract_seamless_m4t_features(wav_list, device=str(device))

    student_features = model.student_ssl_model(
        **{k: v.clone() for k, v in noisy_ssl_inputs.items()}
    ).last_hidden_state

    predicted_0 = model.output_linear1(student_features)
    predicted_1 = model.output_linear2(student_features)
    predicted_latents = torch.cat([predicted_0, predicted_1], dim=-1)

    conditioning = torch.cat(
        [model._normalize_latents(predicted_latents), student_features], dim=-1
    )

    sampled_latents = model.sample_latents(conditioning, conditioning.shape[1], num_steps=num_steps)
    # [1, T_lat, D*2] → [T_lat, D*2]
    return sampled_latents[0]


@torch.inference_mode()
def decode_latents(
    model: DialogueSidonDiffusionLightningModule,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Decode full concatenated latents [T_lat_total, D*2] → waveforms [2, T]."""
    audio_signal = model._decode_latents(latents.unsqueeze(0), normalized=True)
    waveforms = audio_signal.audio_data  # [1, 2, T]
    if waveforms.shape[0] == 1:
        waveforms = waveforms[0]  # [2, T]
    return waveforms


# ---------------------------------------------------------------------------
# Permutation solving via cosine similarity on latents
# ---------------------------------------------------------------------------

def latent_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean-pooled cosine similarity between two [T, D] latent tensors."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = a.norm() * b.norm()
    if denom < 1e-8:
        return 0.0
    return float(torch.dot(a, b) / denom)


def resolve_permutation(
    prev_latents: torch.Tensor,
    curr_latents: torch.Tensor,
    overlap_latent_frames: int,
    latent_dim: int,
) -> bool:
    """Return True if the current chunk's speakers should be swapped.

    Uses cosine similarity between the overlapping region's latents
    of the previous and current chunk to decide the best alignment.

    prev_latents : [T_lat_prev, D*2]
    curr_latents : [T_lat_curr, D*2]
    """
    if overlap_latent_frames <= 0:
        return False

    prev_overlap = prev_latents[-overlap_latent_frames:]  # [T_ov, D*2]
    curr_overlap = curr_latents[:overlap_latent_frames]   # [T_ov, D*2]

    prev_spk1 = prev_overlap[:, :latent_dim]
    prev_spk2 = prev_overlap[:, latent_dim:]
    curr_spk1 = curr_overlap[:, :latent_dim]
    curr_spk2 = curr_overlap[:, latent_dim:]

    direct_score = (
        latent_cosine_similarity(prev_spk1, curr_spk1)
        + latent_cosine_similarity(prev_spk2, curr_spk2)
    )
    swap_score = (
        latent_cosine_similarity(prev_spk1, curr_spk2)
        + latent_cosine_similarity(prev_spk2, curr_spk1)
    )
    return swap_score > direct_score


# ---------------------------------------------------------------------------
# Chunked inference
# ---------------------------------------------------------------------------

def run_separation_chunked(
    model: DialogueSidonDiffusionLightningModule,
    wav: torch.Tensor,
    sample_rate: int,
    num_steps: int,
    chunk_seconds: float = 20.0,
    overlap_seconds: float = 5.0,
) -> tuple[torch.Tensor, int]:
    """Separate a long waveform in overlapping chunks.

    Permutation is resolved by cosine similarity of VAE latents in the
    overlapping region. Latents from all chunks are concatenated in latent
    space, then a single VAE decode produces the final waveform.

    Returns
    -------
    separated : Tensor [2, T_total]
    out_sr    : int
    """
    model_sr = model.cfg.sample_rate
    latent_dim = model.vae.bottleneck.cfg.latent_dim

    # Resample input to model sample rate once
    if sample_rate != model_sr:
        wav = torchaudio.functional.resample(wav, sample_rate, model_sr)

    total_samples = wav.shape[-1]
    chunk_samples = int(chunk_seconds * model_sr)
    overlap_samples = int(overlap_seconds * model_sr)
    hop_samples = chunk_samples - overlap_samples

    if hop_samples <= 0:
        raise ValueError("overlap_seconds must be less than chunk_seconds")

    # If audio fits in a single chunk, skip chunking
    if total_samples <= chunk_samples:
        latents = predict_latents(model, wav, model_sr, num_steps)
        return decode_latents(model, latents), model_sr

    # minimum chunk: 1 second (feature extractor needs ~400 samples at 16kHz)
    min_chunk_samples = model_sr

    starts = list(range(0, total_samples, hop_samples))
    num_chunks = len(starts)
    print(f"chunked inference: {num_chunks} chunks "
          f"(chunk={chunk_seconds}s, overlap={overlap_seconds}s, hop={chunk_seconds - overlap_seconds}s)")

    all_latents: torch.Tensor | None = None
    prev_latents: torch.Tensor | None = None
    prev_end_sample = 0

    for idx, start in enumerate(starts, start=1):
        end = min(start + chunk_samples, total_samples)
        chunk_input_samples = end - start
        if chunk_input_samples < min_chunk_samples:
            print(f"  chunk {idx}/{num_chunks}: skipping (too short: {chunk_input_samples} samples)")
            break
        chunk_wav = wav[:, start:end]

        latents = predict_latents(model, chunk_wav, model_sr, num_steps)
        # latents: [T_lat, D*2]

        # Resolve permutation by latent cosine similarity in the overlap region
        swapped = False
        if prev_latents is not None:
            overlap_input_samples = prev_end_sample - start
            if overlap_input_samples > 0:
                overlap_latent_frames = int(
                    overlap_input_samples * latents.shape[0] / chunk_input_samples
                )
                swapped = resolve_permutation(
                    prev_latents, latents, overlap_latent_frames, latent_dim
                )
                if swapped:
                    # swap speakers: exchange first and second half of latent dim
                    latents = torch.cat(
                        [latents[:, latent_dim:], latents[:, :latent_dim]], dim=-1
                    )

        # Concatenate in latent space, discarding the overlapping frames
        if all_latents is None:
            all_latents = latents
        else:
            overlap_input_samples = prev_end_sample - start
            overlap_latent_frames = int(
                overlap_input_samples * latents.shape[0] / chunk_input_samples
            ) if overlap_input_samples > 0 else 0
            all_latents = torch.cat([all_latents, latents[overlap_latent_frames:]], dim=0)

        prev_latents = latents
        prev_end_sample = end

        print(f"  chunk {idx}/{num_chunks}: input[{start}:{end}] "
              f"latents={latents.shape[0]} frames (swapped={swapped})")

    print(f"decoding {all_latents.shape[0]} total latent frames...")
    return decode_latents(model, all_latents), model_sr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DialogueSidon separation on wav folders or a single video/audio file."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=Path,
                             help="Directory containing input wav files (batch mode).")
    input_group.add_argument("--input-video", type=Path,
                             help="Input video/audio file path (single-file mode).")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint .ckpt file or directory containing checkpoints.")
    parser.add_argument("--output-dir", type=Path,
                        help="Directory where predicted wavs are written (batch mode).")
    parser.add_argument("--output-wav", type=Path,
                        help="Output separated wav path (single-file mode).")
    parser.add_argument("--output-video", type=Path,
                        help="Optional output video with replaced audio (single-file mode).")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-steps", type=int, default=30,
                        help="Number of diffusion sampling steps.")
    parser.add_argument("--chunk-seconds", type=float, default=20.0,
                        help="Chunk duration in seconds.")
    parser.add_argument("--overlap-seconds", type=float, default=5.0,
                        help="Overlap duration in seconds.")
    args = parser.parse_args()

    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
        if args.output_dir is None:
            parser.error("--output-dir is required with --input-dir")
        if args.output_wav is not None or args.output_video is not None:
            parser.error("--output-wav/--output-video can only be used with --input-video")
    else:
        if args.input_video is None or not args.input_video.is_file():
            raise FileNotFoundError(f"Input video not found: {args.input_video}")
        if args.output_wav is None:
            parser.error("--output-wav is required with --input-video")
        if args.output_dir is not None:
            parser.error("--output-dir can only be used with --input-dir")

    ckpt_path = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    model = DialogueSidonDiffusionLightningModule.load_from_checkpoint(
        str(ckpt_path), map_location=device
    ).to(device).eval()

    print(f"checkpoint: {ckpt_path}")
    print(f"device: {device}")
    print(f"chunk={args.chunk_seconds}s, overlap={args.overlap_seconds}s, steps={args.num_steps}")

    def _process(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        wav = normalize_to_mono(wav)
        return run_separation_chunked(
            model, wav, sr,
            num_steps=args.num_steps,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )

    if args.input_dir is not None:
        input_wavs = collect_wavs(args.input_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"num input wavs: {len(input_wavs)}")
        for input_wav in input_wavs:
            wav, sr = torchaudio.load(str(input_wav))
            predicted, out_sr = _process(wav, sr)
            output_wav = args.output_dir / input_wav.name
            torchaudio.save(str(output_wav), predicted.cpu(), sample_rate=out_sr)
            print(f"{input_wav} -> {output_wav}")
    else:
        wav, sr = torchaudio.load(str(args.input_video))
        predicted, out_sr = _process(wav, sr)
        args.output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(args.output_wav), predicted.cpu(), sample_rate=out_sr)
        print(f"{args.input_video} -> {args.output_wav}")

        if args.output_video is not None:
            args.output_video.parent.mkdir(parents=True, exist_ok=True)
            mux_video_with_audio(args.input_video, args.output_wav, args.output_video)
            print(f"{args.input_video} + {args.output_wav} -> {args.output_video}")


if __name__ == "__main__":
    main()
