#!/usr/bin/env python3
"""Export DialogueSidonDiffusion model components via torch.export.

Four artifacts are produced:
  <out_dir>/ssl_encoder.pt2        -- w2v-BERT + output linears
  <out_dir>/diffusion_head.pt2     -- DiffusionTransformerHead (single denoising step)
  <out_dir>/vae_decoder.pt2        -- DAC VAE decoder
  <out_dir>/metadata.json          -- latent norm stats, scheduler config, dims

The diffusion sampling loop (scheduler + iterating over timesteps) is NOT exported
because it involves Python control flow that torch.export cannot trace.
Wrap these three exported components with a thin Python loop for full inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.export

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sidon.model.dialogue_sidion.lightning_module import DialogueSidonDiffusionLightningModule


class SSLEncoderExport(torch.nn.Module):
    """w2v-BERT backbone + two latent projection heads."""

    def __init__(self, model: DialogueSidonDiffusionLightningModule) -> None:
        super().__init__()
        self.ssl_model = model.student_ssl_model
        self.linear1 = model.output_linear1
        self.linear2 = model.output_linear2

    def forward(
        self,
        input_features: torch.Tensor,   # [B, T, 160]
        attention_mask: torch.Tensor,    # [B, T]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.ssl_model(
            input_features=input_features,
            attention_mask=attention_mask,
        ).last_hidden_state              # [B, T', hidden]
        pred0 = self.linear1(features)  # [B, T', latent_dim]
        pred1 = self.linear2(features)  # [B, T', latent_dim]
        return features, pred0, pred1


class VaeDecoderExport(torch.nn.Module):
    """DAC VAE decoder: latents -> audio."""

    def __init__(self, model: DialogueSidonDiffusionLightningModule) -> None:
        super().__init__()
        self.decoder = model.vae.decoder

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [B, latent_dim, T_latent]
        return self.decoder(latents)    # [B, 1, T_audio]


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file() and path.suffix == ".ckpt":
        return path
    if path.is_dir():
        checkpoints = sorted(path.rglob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No .ckpt found under: {path}")
        return checkpoints[-1]
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def export_ssl_encoder(
    model: DialogueSidonDiffusionLightningModule,
    out_path: Path,
    device: torch.device,
    batch_size: int = 1,
    ssl_seq_len: int = 500,
) -> None:
    wrapper = SSLEncoderExport(model).to(device).eval()

    example_features = torch.zeros(batch_size, ssl_seq_len, 160, device=device)
    example_mask = torch.ones(batch_size, ssl_seq_len, dtype=torch.long, device=device)

    seq_dim = torch.export.Dim("ssl_seq")
    dynamic_shapes = {
        "input_features": {1: seq_dim},
        "attention_mask": {1: seq_dim},
    }

    print("Exporting ssl_encoder ...")
    ep = torch.export.export(
        wrapper,
        args=(example_features, example_mask),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    torch.export.save(ep, str(out_path))
    print(f"  -> {out_path}")


def export_diffusion_head(
    model: DialogueSidonDiffusionLightningModule,
    out_path: Path,
    device: torch.device,
    batch_size: int = 1,
    latent_seq_len: int = 500,
) -> None:
    head = model.diffusion_head.to(device).eval()

    latent_size = model.vae.bottleneck.cfg.latent_dim * 2
    cond_size = model.student_ssl_model.config.hidden_size + latent_size

    example_noisy = torch.zeros(batch_size, latent_seq_len, latent_size, device=device)
    example_t = torch.zeros(batch_size, dtype=torch.long, device=device)
    example_cond = torch.zeros(batch_size, latent_seq_len, cond_size, device=device)

    seq_dim = torch.export.Dim("latent_seq")
    dynamic_shapes = {
        "noisy_latents": {1: seq_dim},
        "timesteps": {},
        "conditioning": {1: seq_dim},
    }

    print("Exporting diffusion_head ...")
    ep = torch.export.export(
        head,
        args=(example_noisy, example_t, example_cond),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    torch.export.save(ep, str(out_path))
    print(f"  -> {out_path}")


def export_vae_decoder(
    model: DialogueSidonDiffusionLightningModule,
    out_path: Path,
    device: torch.device,
    batch_size: int = 1,
    latent_seq_len: int = 500,
) -> None:
    wrapper = VaeDecoderExport(model).to(device).eval()

    latent_dim = model.vae.bottleneck.cfg.latent_dim
    example_latents = torch.zeros(batch_size, latent_dim, latent_seq_len, device=device)

    seq_dim = torch.export.Dim("latent_seq")
    dynamic_shapes = {
        "latents": {2: seq_dim},
    }

    print("Exporting vae_decoder ...")
    ep = torch.export.export(
        wrapper,
        args=(example_latents,),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    torch.export.save(ep, str(out_path))
    print(f"  -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DialogueSidonDiffusion components via torch.export."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    model = DialogueSidonDiffusionLightningModule.load_from_checkpoint(
        str(ckpt_path), map_location=device
    ).to(device).eval()

    export_ssl_encoder(model, args.output_dir / "ssl_encoder.pt2", device)
    export_diffusion_head(model, args.output_dir / "diffusion_head.pt2", device)
    export_vae_decoder(model, args.output_dir / "vae_decoder.pt2", device)
    save_metadata(model, args.output_dir / "metadata.json")

    print("Done.")


def save_metadata(
    model: DialogueSidonDiffusionLightningModule,
    out_path: Path,
) -> None:
    latent_dim = model.vae.bottleneck.cfg.latent_dim
    metadata = {
        "latent_dim": latent_dim,
        "sample_rate": model.cfg.sample_rate,
        "ssl_hidden_size": model.student_ssl_model.config.hidden_size,
        "latent_norm_initialized": bool(model.latent_norm_initialized.item()),
        "latent_norm_mean": model.latent_norm_mean.squeeze().tolist(),
        "latent_norm_std": model.latent_norm_std.squeeze().tolist(),
        "ddpm_config": dict(model.ddpm_scheduler.config),
    }
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
