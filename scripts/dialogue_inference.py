#!/usr/bin/env python3
"""CLI utility to run DialogueSidonLightningModule inference on a WAV file."""

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

from sidon.model.dialogue_sidion.audio import extract_seamless_m4t_features
from sidon.model.dialogue_sidion.lightning_module import (
    DialogueFeaturePredictorLightningModule,
    DialogueSidonLightningModule,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DialogueSidonLightningModule inference on a mono dialogue mixture.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained DialogueSidonLightningModule checkpoint.",
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
        "--feature-predictor-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional override for the DialogueFeaturePredictor checkpoint. "
            "Useful if the DialogueSidon checkpoint was saved without an embedded predictor."
        ),
    )
    return parser.parse_args()


def _normalise_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Scale waveform to 0.9 peak to match training-time normalisation."""
    peak = waveform.abs().max()
    if peak > 0:
        waveform = 0.9 * waveform / peak
    return waveform


def _repeat_batch_encoding(enc: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    """Repeat SSL BatchEncoding tensors along the batch dimension."""
    out: dict[str, torch.Tensor] = {}
    for key, value in enc.items():
        if isinstance(value, torch.Tensor):
            out[key] = torch.cat([value] * repeats, dim=0)
        else:
            out[key] = value
    return out


@torch.inference_mode()
def _load_module(
    checkpoint: Path,
    device: torch.device,
    feature_predictor_ckpt: Path | None = None,
) -> DialogueSidonLightningModule:
    """Restore the DialogueSidonLightningModule and optionally swap the predictor checkpoint."""
    module = DialogueSidonLightningModule.load_from_checkpoint(
        checkpoint,
        map_location=device,
    )
    module = module.to(device).eval()
    if feature_predictor_ckpt is not None:
        predictor = DialogueFeaturePredictorLightningModule.load_from_checkpoint(
            feature_predictor_ckpt,
            map_location=device,
        )
        predictor = predictor.to(device).eval()
        module.feature_predictor = predictor
    module.feature_predictor.eval()
    module.decoder.eval()
    return module


@torch.inference_mode()
def _prepare_ssl_inputs(
    waveform: torch.Tensor,
    sr: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert a mono waveform into the Seamless M4T feature batch used by the predictor."""
    waveform = waveform.to(device)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mixture = torchaudio.functional.resample(waveform, sr, 16_000)
    mixture = _normalise_waveform(mixture)
    padded = torch.nn.functional.pad(mixture, (160, 160))
    features = extract_seamless_m4t_features(
        [padded.view(-1)],
        device=str(device),
    )
    return features


@torch.inference_mode()
def run_inference(
    module: DialogueSidonLightningModule,
    noisy_wav: torch.Tensor,
    noisy_sr: int,
) -> tuple[torch.Tensor, int]:
    """Predict a stereo waveform from a noisy mono dialogue recording."""
    device = next(module.parameters()).device
    ssl_inputs = _prepare_ssl_inputs(noisy_wav, noisy_sr, device)

    n_speakers = 2
    batch_size = ssl_inputs["input_features"].shape[0]
    conditioning = (
        torch.arange(n_speakers, device=device, dtype=torch.long)
        .unsqueeze(1)
        .expand(-1, batch_size)
        .reshape(-1)
    )
    repeated_inputs = _repeat_batch_encoding(ssl_inputs, n_speakers)
    predicted_features = module.feature_predictor.student_ssl_model(
        **repeated_inputs,
        conditioning=conditioning,
    ).last_hidden_state

    predicted_features = predicted_features.view(
        n_speakers,
        batch_size,
        *predicted_features.shape[1:],
    )
    decoder_input = torch.cat(
        [predicted_features[i].transpose(1, 2) for i in range(n_speakers)],
        dim=1,
    )
    restored = module.decoder(decoder_input)
    stereo = restored[0].detach().cpu().float()
    sample_rate = int(module.cfg.sample_rate)
    return stereo, sample_rate


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    module = _load_module(
        args.checkpoint,
        device=device,
        feature_predictor_ckpt=args.feature_predictor_checkpoint,
    )
    noisy_wav, sr = torchaudio.load(str(args.input))
    stereo, target_sr = run_inference(module, noisy_wav, sr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(args.output),
        stereo,
        sample_rate=target_sr,
    )
    print(f"Wrote restored dialogue to {args.output} at {target_sr} Hz.")


if __name__ == "__main__":
    main()
