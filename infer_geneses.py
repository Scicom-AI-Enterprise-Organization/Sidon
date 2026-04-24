#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sidon.model.geneses.lightning_module import GenesesLightningModule


def move_aux_modules_to_device(model: GenesesLightningModule, device: torch.device) -> None:
    if hasattr(model, "dacvae") and hasattr(model.dacvae, "to"):
        model.dacvae.to(device)


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


def run_separation(
    model: GenesesLightningModule,
    wav: torch.Tensor,
    sample_rate: int,
    num_steps: int,
) -> tuple[torch.Tensor, int]:
    with torch.inference_mode():
        predicted, out_sr = model.predict_separated(
            wav.to(model.device),
            sample_rate=sample_rate,
            num_steps=num_steps,
        )
    predicted = predicted.detach().cpu()
    if predicted.ndim == 1:
        predicted = predicted.unsqueeze(0)
    elif predicted.ndim == 3:
        predicted = predicted[0]
    return predicted, out_sr


def match_num_samples(audio: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    current_num_samples = audio.shape[-1]
    if current_num_samples == target_num_samples:
        return audio
    if current_num_samples > target_num_samples:
        return audio[..., :target_num_samples]
    pad = torch.zeros(
        (*audio.shape[:-1], target_num_samples - current_num_samples),
        dtype=audio.dtype,
    )
    return torch.cat([audio, pad], dim=-1)


def channel_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom) < 1e-8:
        return 0.0
    return float(torch.dot(a, b) / denom)


def maybe_swap_chunk_speakers(
    prev_overlap: torch.Tensor,
    curr_chunk: torch.Tensor,
    overlap_num_samples: int,
) -> tuple[torch.Tensor, bool]:
    if overlap_num_samples <= 0:
        return curr_chunk, False
    if prev_overlap.shape[0] != 2 or curr_chunk.shape[0] != 2:
        return curr_chunk, False

    curr_overlap = curr_chunk[:, :overlap_num_samples]
    direct_score = channel_similarity(prev_overlap[0], curr_overlap[0]) + channel_similarity(
        prev_overlap[1], curr_overlap[1]
    )
    swapped_score = channel_similarity(prev_overlap[0], curr_overlap[1]) + channel_similarity(
        prev_overlap[1], curr_overlap[0]
    )
    if swapped_score > direct_score:
        return curr_chunk[[1, 0], :], True
    return curr_chunk, False


def run_separation_chunked(
    model: GenesesLightningModule,
    wav: torch.Tensor,
    sample_rate: int,
    num_steps: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> tuple[torch.Tensor, int]:
    chunk_num_samples = int(round(chunk_seconds * sample_rate))
    if chunk_seconds <= 0 or chunk_num_samples <= 0 or wav.shape[-1] <= chunk_num_samples:
        return run_separation(model, wav, sample_rate, num_steps)

    overlap_num_samples = int(round(overlap_seconds * sample_rate))
    overlap_num_samples = max(0, min(overlap_num_samples, chunk_num_samples - 1))
    hop_num_samples = chunk_num_samples - overlap_num_samples
    if hop_num_samples <= 0:
        hop_num_samples = chunk_num_samples
        overlap_num_samples = 0

    total_num_samples = wav.shape[-1]
    starts = list(range(0, total_num_samples, hop_num_samples))
    num_chunks = len(starts)
    print(
        "chunked inference: "
        f"{num_chunks} chunks (chunk_seconds={chunk_seconds}, overlap_seconds={overlap_num_samples / sample_rate:.3f})"
    )

    stitched_output: torch.Tensor | None = None
    output_sample_rate: int | None = None
    previous_chunk_end = 0
    for index, start in enumerate(starts, start=1):
        end = min(start + chunk_num_samples, total_num_samples)
        wav_chunk = wav[:, start:end]
        predicted_chunk, out_sr = run_separation(
            model=model,
            wav=wav_chunk,
            sample_rate=sample_rate,
            num_steps=num_steps,
        )
        if output_sample_rate is None:
            output_sample_rate = out_sr
        elif out_sr != output_sample_rate:
            raise RuntimeError(
                f"Inconsistent output sample rates across chunks: {output_sample_rate} vs {out_sr}"
            )

        target_chunk_samples = int(
            round((end - start) * float(output_sample_rate) / float(sample_rate))
        )
        target_chunk_samples = max(1, target_chunk_samples)
        predicted_chunk = match_num_samples(predicted_chunk, target_chunk_samples)
        if stitched_output is None:
            stitched_output = predicted_chunk
            previous_chunk_end = end
            print(f"  chunk {index}/{num_chunks}: input[{start}:{end}] -> {target_chunk_samples} samples")
            continue

        overlap_in_samples = max(0, previous_chunk_end - start)
        overlap_out_samples = int(
            round(overlap_in_samples * float(output_sample_rate) / float(sample_rate))
        )
        overlap_out_samples = max(0, overlap_out_samples)
        overlap_out_samples = min(
            overlap_out_samples,
            stitched_output.shape[-1],
            predicted_chunk.shape[-1],
        )

        swapped = False
        if overlap_out_samples > 0:
            previous_overlap = stitched_output[:, -overlap_out_samples:]
            predicted_chunk, swapped = maybe_swap_chunk_speakers(
                prev_overlap=previous_overlap,
                curr_chunk=predicted_chunk,
                overlap_num_samples=overlap_out_samples,
            )
            fade = torch.linspace(
                0.0,
                1.0,
                steps=overlap_out_samples,
                dtype=stitched_output.dtype,
            ).unsqueeze(0)
            blended_overlap = previous_overlap * (1.0 - fade) + predicted_chunk[:, :overlap_out_samples] * fade
            stitched_output = torch.cat(
                [
                    stitched_output[:, :-overlap_out_samples],
                    blended_overlap,
                    predicted_chunk[:, overlap_out_samples:],
                ],
                dim=-1,
            )
        else:
            stitched_output = torch.cat([stitched_output, predicted_chunk], dim=-1)

        previous_chunk_end = end
        print(
            f"  chunk {index}/{num_chunks}: input[{start}:{end}] -> {target_chunk_samples} samples "
            f"(overlap={overlap_out_samples}, swapped={swapped})"
        )

    if output_sample_rate is None or stitched_output is None:
        raise RuntimeError("No output generated during chunked inference.")
    return stitched_output, output_sample_rate


def mux_video_with_audio(input_video: Path, input_wav: Path, output_video: Path) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg is required for --output-video but was not found in PATH.")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(input_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()[-2000:]
        raise RuntimeError(f"ffmpeg failed while writing {output_video}: {stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GenesesLightningModule separation on wav folders or a single video/audio file."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing input wav files (batch mode).",
    )
    input_group.add_argument(
        "--input-video",
        type=Path,
        help="Input video/audio file path (single-file mode).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint .ckpt file or directory containing checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where predicted wavs are written (batch mode).",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        help="Output separated wav path (single-file mode).",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Optional output video path with replaced separated audio (single-file mode).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of ODE solver steps. step_size = 1 / num_steps.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=20.0,
        help="Chunk duration (seconds) for long wav inference. Set <=0 to disable chunking.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=1.0,
        help="Chunk overlap (seconds) used for speaker alignment and crossfade stitching.",
    )
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
    model = GenesesLightningModule.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
    ).to(device).eval()
    move_aux_modules_to_device(model, device)

    print(f"checkpoint: {ckpt_path}")
    print(f"device: {device}")

    if args.input_dir is not None:
        input_wavs = collect_wavs(args.input_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"num input wavs: {len(input_wavs)}")
        print(f"output dir: {args.output_dir}")

        for input_wav in input_wavs:
            wav, sr = torchaudio.load(str(input_wav))
            wav = normalize_to_mono(wav)
            predicted, out_sr = run_separation_chunked(
                model=model,
                wav=wav,
                sample_rate=sr,
                num_steps=args.num_steps,
                chunk_seconds=args.chunk_seconds,
                overlap_seconds=args.overlap_seconds,
            )
            output_wav = args.output_dir / input_wav.name
            torchaudio.save(str(output_wav), predicted, sample_rate=out_sr)
            print(f"{input_wav} -> {output_wav}")
    else:
        wav, sr = torchaudio.load(str(args.input_video))
        wav = normalize_to_mono(wav)
        predicted, out_sr = run_separation_chunked(
            model=model,
            wav=wav,
            sample_rate=sr,
            num_steps=args.num_steps,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )
        args.output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(args.output_wav), predicted, sample_rate=out_sr)
        print(f"{args.input_video} -> {args.output_wav}")

        if args.output_video is not None:
            args.output_video.parent.mkdir(parents=True, exist_ok=True)
            mux_video_with_audio(
                input_video=args.input_video,
                input_wav=args.output_wav,
                output_video=args.output_video,
            )
            print(f"{args.input_video} + {args.output_wav} -> {args.output_video}")


if __name__ == "__main__":
    main()
