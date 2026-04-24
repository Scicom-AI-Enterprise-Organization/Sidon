#!/usr/bin/env python3
"""RTF (Real-Time Factor) evaluation for Geneses and DialogueSidonDiffusion.

RTF = wall_clock_inference_time / audio_duration

Usage:
  python eval_rtf.py \
    --geneses-checkpoint  sidon/<run_id>/checkpoints/last.ckpt \
    --dialogue-checkpoint sidon/<run_id>/checkpoints/last.ckpt \
    [--input-wav path/to/16k_mono.wav] \
    [--num-steps 30] \
    [--num-runs 10] \
    [--device cuda:0]

If --input-wav is omitted, a 20-second synthetic 16 kHz mono signal is used.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
import sys

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sidon.model.geneses.lightning_module import GenesesLightningModule
from sidon.model.dialogue_sidion.lightning_module import DialogueSidonDiffusionLightningModule


EVAL_SAMPLE_RATE = 16_000
EVAL_DURATION_SEC = 20.0


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file() and path.suffix == ".ckpt":
        return path
    if path.is_dir():
        checkpoints = sorted(path.rglob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No .ckpt found under: {path}")
        return checkpoints[-1]
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def make_synthetic_wav(device: torch.device) -> tuple[torch.Tensor, int]:
    num_samples = int(EVAL_DURATION_SEC * EVAL_SAMPLE_RATE)
    t = torch.linspace(0, EVAL_DURATION_SEC, num_samples, device=device)
    wav = 0.45 * torch.sin(2 * 3.14159 * 440 * t) + 0.45 * torch.sin(2 * 3.14159 * 880 * t)
    return wav.unsqueeze(0), EVAL_SAMPLE_RATE


def load_wav(path: Path, device: torch.device) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != EVAL_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, EVAL_SAMPLE_RATE)
    num_samples = int(EVAL_DURATION_SEC * EVAL_SAMPLE_RATE)
    wav = wav[:, :num_samples]
    if wav.shape[-1] < num_samples:
        pad = torch.zeros(1, num_samples - wav.shape[-1])
        wav = torch.cat([wav, pad], dim=-1)
    return wav.to(device), EVAL_SAMPLE_RATE


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_rtf(
    model: torch.nn.Module,
    wav: torch.Tensor,
    sample_rate: int,
    num_steps: int,
    num_runs: int,
    audio_duration: float,
    device: torch.device,
) -> dict[str, float]:
    # reset peak memory stats before measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # warmup
    with torch.inference_mode():
        model.predict_separated(wav, sample_rate=sample_rate, num_steps=num_steps)
    sync(device)

    peak_mem_bytes = 0
    if device.type == "cuda":
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)

    times: list[float] = []
    for _ in range(num_runs):
        sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.predict_separated(wav, sample_rate=sample_rate, num_steps=num_steps)
        sync(device)
        times.append(time.perf_counter() - t0)

    rtf_values = [t / audio_duration for t in times]
    return {
        "rtf_mean": statistics.mean(rtf_values),
        "rtf_std": statistics.stdev(rtf_values) if len(rtf_values) > 1 else 0.0,
        "rtf_median": statistics.median(rtf_values),
        "rtf_min": min(rtf_values),
        "rtf_max": max(rtf_values),
        "elapsed_mean_s": statistics.mean(times),
        "elapsed_std_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "elapsed_median_s": statistics.median(times),
        "elapsed_min_s": min(times),
        "elapsed_max_s": max(times),
        "peak_gpu_mem_mb": peak_mem_bytes / 1024 ** 2,
    }


def print_results(
    name: str,
    results: dict[str, float],
    num_runs: int,
    ckpt_path: Path,
    total_params: int,
    trainable_params: int,
    num_steps: int,
    audio_duration: float,
    sample_rate: int,
    device: torch.device,
) -> None:
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  Model              : {name}")
    print(f"  Checkpoint         : {ckpt_path.name}")
    print(f"  Checkpoint size    : {ckpt_path.stat().st_size / 1024**2:.1f} MB")
    print(f"  Total params       : {total_params:,}")
    print(f"  Trainable params   : {trainable_params:,}")
    print(f"  Inference steps    : {num_steps}")
    print(f"  Input duration     : {audio_duration:.1f} s @ {sample_rate} Hz")
    print(f"  Runs               : {num_runs}  (+ 1 warmup)")
    if device.type == "cuda":
        print(f"  Peak GPU memory    : {results['peak_gpu_mem_mb']:.1f} MB")
    print(f"  ---")
    print(f"  RTF  mean  ± std   : {results['rtf_mean']:.5f} ± {results['rtf_std']:.5f}")
    print(f"  RTF  median        : {results['rtf_median']:.5f}")
    print(f"  RTF  min / max     : {results['rtf_min']:.5f} / {results['rtf_max']:.5f}")
    print(f"  Time mean  ± std   : {results['elapsed_mean_s']:.3f} ± {results['elapsed_std_s']:.3f} s")
    print(f"  Time median        : {results['elapsed_median_s']:.3f} s")
    print(f"  Time min / max     : {results['elapsed_min_s']:.3f} / {results['elapsed_max_s']:.3f} s")
    print(sep)


def print_hardware(device: torch.device) -> None:
    print("\n--- Hardware ---")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        print(f"  GPU        : {torch.cuda.get_device_name(idx)}")
        props = torch.cuda.get_device_properties(idx)
        print(f"  VRAM total : {props.total_memory / 1024**2:.0f} MB")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  CUDA       : {torch.version.cuda}")
    print(f"  torchaudio : {torchaudio.__version__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RTF evaluation for Geneses and DialogueSidon.")
    parser.add_argument("--geneses-checkpoint", type=Path, default=None)
    parser.add_argument("--dialogue-checkpoint", type=Path, default=None)
    parser.add_argument("--input-wav", type=Path, default=None)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.geneses_checkpoint is None and args.dialogue_checkpoint is None:
        parser.error("Provide at least one of --geneses-checkpoint or --dialogue-checkpoint.")

    device = torch.device(args.device)

    if args.input_wav is not None:
        wav, sr = load_wav(args.input_wav, device)
        print(f"Input: {args.input_wav}  ({wav.shape[-1] / sr:.2f}s @ {sr} Hz)")
    else:
        wav, sr = make_synthetic_wav(device)
        print(f"Input: synthetic sine mixture ({EVAL_DURATION_SEC}s @ {sr} Hz)")

    audio_duration = wav.shape[-1] / sr

    print_hardware(device)

    if args.geneses_checkpoint is not None:
        ckpt = resolve_checkpoint(args.geneses_checkpoint)
        print(f"\nLoading Geneses: {ckpt}")
        geneses = GenesesLightningModule.load_from_checkpoint(
            str(ckpt), map_location=device
        ).to(device).eval()
        if hasattr(geneses, "dacvae"):
            geneses.dacvae.to(device)
        total_p, train_p = count_parameters(geneses)
        results = measure_rtf(geneses, wav, sr, args.num_steps, args.num_runs, audio_duration, device)
        print_results("Geneses", results, args.num_runs, ckpt, total_p, train_p,
                      args.num_steps, audio_duration, sr, device)
        del geneses

    if args.dialogue_checkpoint is not None:
        ckpt = resolve_checkpoint(args.dialogue_checkpoint)
        print(f"\nLoading DialogueSidon (vae32): {ckpt}")
        dialogue = DialogueSidonDiffusionLightningModule.load_from_checkpoint(
            str(ckpt), map_location=device
        ).to(device).eval()
        total_p, train_p = count_parameters(dialogue)
        results = measure_rtf(dialogue, wav, sr, args.num_steps, args.num_runs, audio_duration, device)
        print_results("DialogueSidon-vae32", results, args.num_runs, ckpt, total_p, train_p,
                      args.num_steps, audio_duration, sr, device)
        del dialogue


if __name__ == "__main__":
    main()
