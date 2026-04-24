#!/usr/bin/env python3
"""CLI utility to run DialogueSidonDiffusionLightningModule inference on a WAV file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torchaudio

from sidon.model.dialogue_sidion.lightning_module import (
    DialogueSidonDiffusionLightningModule,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DialogueSidonDiffusionLightningModule inference on a mono dialogue mixture.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained DialogueSidonDiffusionLightningModule checkpoint.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the noisy mono dialogue WAV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the restored stereo WAV file.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run inference on (default: auto-detect CUDA).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Optional override for diffusion sampling steps.",
    )
    return parser.parse_args()


@torch.inference_mode()
def _load_module(
    checkpoint: Path,
    device: torch.device,
) -> DialogueSidonDiffusionLightningModule:
    module = DialogueSidonDiffusionLightningModule.load_from_checkpoint(
        checkpoint,
        map_location=device,
    )
    return module.to(device).eval()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    module = _load_module(args.checkpoint, device=device)
    noisy_wav, sr = torchaudio.load(str(args.input))
    separated, target_sr = module.predict_separated(
        noisy_wav.to(device),
        int(sr),
        num_steps=args.num_steps,
    )
    separated = separated.detach().cpu().float()
    if separated.ndim == 1:
        separated = separated.unsqueeze(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), separated, sample_rate=target_sr)
    print(f"Wrote restored dialogue to {args.output} at {target_sr} Hz.")


if __name__ == "__main__":
    main()
