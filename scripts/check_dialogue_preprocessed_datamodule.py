#!/usr/bin/env python3
"""
Fetch and print one batch from PreprocessedDialogueDataModule.

Usage:
  python scripts/check_dialogue_preprocessed_datamodule.py \
    --config config/data/dialogue_preprocessed.yaml \
    [--train-urls /path/to/shards] [--val-urls /path/to/shards] \
    [--train-num-workers 0] [--val-num-workers 0]

This prints key shapes from a single train batch to verify the pipeline.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any


def _add_src_to_path() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_add_src_to_path()

from omegaconf import OmegaConf  # noqa: E402
import hydra  # noqa: E402
import os  # noqa: E402
import torchaudio  # noqa: E402


def describe(value: Any) -> str:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return f"Tensor{tuple(value.shape)} {value.dtype}"
    except Exception:
        pass
    if hasattr(value, "keys") and not hasattr(value, "shape"):
        try:
            return f"Mapping[{', '.join(sorted(list(value.keys())))}]"
        except Exception:
            return f"{type(value).__name__}"
    if hasattr(value, "shape"):
        return f"Array{tuple(value.shape)}"
    return f"{type(value).__name__}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check one batch from PreprocessedDialogueDataModule"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("config/data/dialogue_preprocessed.yaml"),
        help="Hydra YAML containing datamodule=_target_: PreprocessedDialogueDataModule",
    )
    parser.add_argument(
        "--train-urls",
        type=str,
        default=None,
        help="Override datamodule.train_urls (directory or list file)",
    )
    parser.add_argument(
        "--val-urls",
        type=str,
        default=None,
        help="Override datamodule.val_urls (directory or list file)",
    )
    parser.add_argument(
        "--train-num-workers",
        type=int,
        default=None,
        help="Override datamodule.train_num_workers",
    )
    parser.add_argument(
        "--val-num-workers",
        type=int,
        default=None,
        help="Override datamodule.val_num_workers",
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("check_outputs"),
        help="Directory to save sample audio from the fetched batch",
    )
    parser.add_argument(
        "--num-save",
        type=int,
        default=2,
        help="Number of items per key to save (if available)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(str(args.config))
    if "datamodule" not in cfg:
        print(f"[ERROR] No 'datamodule' section in {args.config}")
        return 1
    dm_cfg = cfg.datamodule

    # Minimal overrides for quick checking
    if args.train_urls is not None:
        dm_cfg.train_urls = [args.train_urls]
    if args.val_urls is not None:
        dm_cfg.val_urls = [args.val_urls]
    if args.train_num_workers is not None:
        dm_cfg.train_num_workers = int(args.train_num_workers)
    if args.val_num_workers is not None:
        dm_cfg.val_num_workers = int(args.val_num_workers)

    # Instantiate the datamodule from Hydra-style config
    datamodule = hydra.utils.instantiate(dm_cfg)

    # Build datasets and get a loader
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))

    print("[OK] Fetched one train batch")
    print(f"keys: {sorted(list(batch.keys()))}")
    # Print a concise description of common entries
    for k in [
        "input_wav",
        "input_wav_lens",
        "sr",
        "noisy_input_wav16k",
        "noisy_input_wav",
        "clean_mixture",
        "clean_16k_mixture",
        "noisy_mixture",
        "noisy_16k_mixture",
    ]:
        if k in batch:
            print(f"  {k}: {describe(batch[k])}")

    # Optionally save a few audio samples with torchaudio.save
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    def _save_key(key: str, default_sr: int) -> None:
        if key not in batch:
            return
        tensor = batch[key]
        names = batch.get("names", None)
        try:
            sr = int(batch["sr"]) if "16k" not in key else int(16000)
        except Exception:
            sr = default_sr
        # Expect shapes: (B, C, T) or (B, T)
        if tensor.dim() == 3:
            B, C, T = tensor.shape
        elif tensor.dim() == 2:
            B, C, T = tensor.shape[0], 1, tensor.shape[1]
        else:
            return
        limit = min(args.num_save, B)
        for i in range(limit):
            wav = tensor[i]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # (1, T)
            filename = f"{key}_{i}.wav"
            if isinstance(names, list) and i < len(names):
                stem = str(names[i]).replace(os.sep, "_")
                filename = f"{stem}_{key}.wav"
            path = outdir / filename
            torchaudio.save(str(path), wav.cpu().float(), sample_rate=sr)

    _save_key("input_wav", default_sr=48000)
    _save_key("noisy_input_wav", default_sr=48000)
    _save_key("noisy_input_wav16k", default_sr=16000)
    _save_key("clean_mixture", default_sr=int(batch.get("sr", 48000)))
    _save_key("clean_16k_mixture", default_sr=16000)
    _save_key("noisy_mixture", default_sr=int(batch.get("sr", 48000)))
    _save_key("noisy_16k_mixture", default_sr=16000)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
