import argparse
import subprocess
from pathlib import Path
from diffusers import DPMSolverMultistepScheduler

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


def save_video_with_audio(
    input_video: Path,
    predicted_wav: Path,
    output_video: Path,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(predicted_wav),
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
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DialogueSidon separation and write output video with predicted audio."
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=Path("debate.mp4"),
        help="Input video/audio file for inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint .ckpt file or directory containing checkpoints.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("debate_separated.mp4"),
        help="Output video path.",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=Path("debate_separated.wav"),
        help="Intermediate/final predicted wav path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (e.g. cpu, cuda:0).",
    )
    args = parser.parse_args()

    if not args.input_video.exists():
        raise FileNotFoundError(f"Input not found: {args.input_video}")

    ckpt_path = resolve_checkpoint(args.checkpoint)

    wav, sr = torchaudio.load(str(args.input_video))
    wav = wav.mean(dim=0, keepdim=True)

    model = DialogueSidonDiffusionLightningModule.load_from_checkpoint(str(ckpt_path))
    model = model.to(args.device).eval()
    model.ddpm_scheduler = DPMSolverMultistepScheduler.from_config(model.ddpm_scheduler.config,algorithm_type='dpmsolver++',timestep_spacing='linspace')

    with torch.inference_mode():
        predicted, out_sr = model.predict_separated(wav.to(model.device), sample_rate=sr,num_steps=30)

    predicted = predicted.detach().cpu()
    if predicted.ndim == 1:
        predicted = predicted.unsqueeze(0)
    elif predicted.ndim == 3:
        predicted = predicted[0]

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output_wav), predicted, sample_rate=out_sr)
    save_video_with_audio(args.input_video, args.output_wav, args.output_video)

    print(f"checkpoint: {ckpt_path}")
    print(f"wav: {args.output_wav}")
    print(f"video: {args.output_video}")


if __name__ == "__main__":
    main()
