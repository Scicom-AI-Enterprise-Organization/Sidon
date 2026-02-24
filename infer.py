import argparse
from pathlib import Path
import shutil
import subprocess

import torch
import torchaudio

from sidon.model.dialogue_sidion.lightning_module import (
    DialogueSidonDiffusionLightningModule,
)


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
    model: DialogueSidonDiffusionLightningModule,
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
        description="Run DialogueSidon separation on wav folders or a single video/audio file."
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
        default="cpu",
        help="Torch device (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=30,
        help="Number of diffusion sampling steps.",
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
    model = DialogueSidonDiffusionLightningModule.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
    ).to(device).eval()

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
            predicted, out_sr = run_separation(
                model=model,
                wav=wav,
                sample_rate=sr,
                num_steps=args.num_steps,
            )
            output_wav = args.output_dir / input_wav.name
            torchaudio.save(str(output_wav), predicted, sample_rate=out_sr)
            print(f"{input_wav} -> {output_wav}")
    else:
        wav, sr = torchaudio.load(str(args.input_video))
        wav = normalize_to_mono(wav)
        predicted, out_sr = run_separation(
            model=model,
            wav=wav,
            sample_rate=sr,
            num_steps=args.num_steps,
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
