#!/usr/bin/env python3
"""
Quick checker for DialogueDatasetDataModule.

Loads default datamodule settings from the repo config, overrides the
train/val WebDataset shard paths to a provided speech dataset directory,
and attempts to iterate one batch from the train and val loaders while
printing key tensor shapes.

Usage:
  python scripts/check_dialogue_datamodule.py \
    --speech_dir /home/acc12576tt/datasets/shard \
    [--config config/data/webdataset_preprocess_48k.yaml] \
    [--max_batches 1] [--train-num-workers 0] [--val-num-workers 0]

Note:
- This script requires the same dependencies as the datamodule itself
  (torch, torchaudio, webdataset, transformers, etc.).
- The Hugging Face feature extractor will be loaded per the config; network
  access or a local cache is required.
"""

from __future__ import annotations

import argparse
import sys
import pathlib
import traceback
from typing import Any, Dict, List


def _add_src_to_path() -> None:
    """Ensure the local `src/` is on sys.path for `sidon` imports."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_add_src_to_path()

from omegaconf import OmegaConf  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402


def load_datamodule_defaults(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load datamodule defaults from a Hydra/OmegaConf YAML file.

    Expects a structure like:
      datamodule:
        _target_: sidon.data.preprocess.WebDatasetDataModule
        train_wds_patterns: [...]
        val_wds_patterns: [...]
        ...
    """
    cfg = OmegaConf.load(str(config_path))
    if "datamodule" not in cfg:
        raise ValueError(f"No 'datamodule' section in {config_path}")
    # Convert OmegaConf to a plain dict
    dm_cfg = OmegaConf.to_container(cfg.datamodule, resolve=True)  # type: ignore[arg-type]
    assert isinstance(dm_cfg, dict)
    # Remove Hydra's _target_ if present; we'll instantiate directly
    dm_cfg.pop("_target_", None)
    return dm_cfg


def maybe_load_env_module(module_name: str = "gcc") -> bool:
    """Attempt to load an environment module (e.g., gcc) into this process.

    Uses `modulecmd python load <module>` to emit Python code that adjusts
    os.environ, then execs it so the current process sees LD_LIBRARY_PATH, etc.

    Returns True on success, False otherwise.
    """
    candidates = []
    # Prefer explicit MODULESHOME when available
    modules_home = os.environ.get("MODULESHOME")
    if modules_home:
        candidates.append(str(pathlib.Path(modules_home) / "bin" / "modulecmd"))
    # Common default install path
    candidates.append("/usr/share/Modules/bin/modulecmd")
    # Fall back to PATH lookup
    candidates.append("modulecmd")

    last_err = None
    for cmd in candidates:
        exe = shutil.which(cmd) if os.path.sep not in cmd else (cmd if os.path.exists(cmd) else None)
        if not exe:
            continue
        try:
            out = subprocess.check_output([exe, "python", "load", module_name], text=True)
            exec(out, {"os": os})  # apply env changes to current process
            print(f"[INFO] Loaded module via {exe}: {module_name}")
            return True
        except Exception as e:
            last_err = e
            continue
    if last_err:
        print(f"[WARN] Could not load module '{module_name}' via environment modules: {last_err}")
    else:
        print(f"[WARN] No 'modulecmd' found to load module '{module_name}'")
    return False


def build_dm_kwargs(defaults: Dict[str, Any], speech_dir: pathlib.Path,
                    train_num_workers: int | None, val_num_workers: int | None) -> Dict[str, Any]:
    """Prepare kwargs for DialogueDatasetDataModule with path overrides."""
    # Make a shallow copy we can mutate safely
    kwargs = dict(defaults)
    # Always override shard paths with the provided speech_dir for both splits
    kwargs["train_wds_patterns"] = [str(speech_dir)]
    kwargs["val_wds_patterns"] = [str(speech_dir)]

    # Optionally override workers for a lightweight check run
    if train_num_workers is not None:
        kwargs["train_num_workers"] = int(train_num_workers)
    if val_num_workers is not None:
        kwargs["val_num_workers"] = int(val_num_workers)

    return kwargs


def describe_batch(tag: str, batch: Dict[str, Any]) -> List[str]:
    """Format a compact description of key tensors in a batch."""
    lines: List[str] = [f"[{tag}] keys: {sorted(list(batch.keys()))}"]
    def shape(x: Any) -> str:
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return str(tuple(x.shape))
        except Exception:
            pass
        try:
            return str(x.shape)
        except Exception:
            return f"type={type(x).__name__}"

    for k in [
        "input_wav", "input_wav_lens", "sr",
        "clean_ssl_inputs_0", "clean_ssl_inputs_1",
        "clean_mixture_ssl_inputs",
        "noisy_ssl_inputs_0", "noisy_ssl_inputs_1",
        "noisy_mixture_ssl_inputs",
        "noisy_input_wav16k", "noisy_input_wav",
    ]:
        if k in batch:
            v = batch[k]
            try:
                # HF processors return BatchEncoding-like objects; show top-level keys
                if hasattr(v, "keys") and not hasattr(v, "shape"):
                    lines.append(f"  {k}: {sorted(list(v.keys()))}")
                else:
                    lines.append(f"  {k}: {shape(v)}")
            except Exception:
                lines.append(f"  {k}: {type(v).__name__}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Check DialogueDatasetDataModule setup and iteration")
    parser.add_argument("--speech_dir", type=pathlib.Path,
                        default=pathlib.Path("/home/acc12576tt/datasets/shard"),
                        help="Directory containing WebDataset shards (.tar/.tar.gz)")
    parser.add_argument("--config", type=pathlib.Path,
                        default=pathlib.Path("config/data/webdataset_preprocess_48k.yaml"),
                        help="Config file to read default datamodule settings from")
    parser.add_argument("--max_batches", type=int, default=1,
                        help="Max batches to fetch from each loader for checking")
    parser.add_argument("--train-num-workers", type=int, default=0,
                        help="Override train dataloader workers (default uses config)")
    parser.add_argument("--val-num-workers", type=int, default=0,
                        help="Override val dataloader workers (default uses config)")
    args = parser.parse_args()

    # Basic validation of the shard directory
    if not args.speech_dir.exists():
        print(f"[WARN] speech_dir does not exist: {args.speech_dir}")
    shard_glob = list(args.speech_dir.rglob("*.tar")) + list(args.speech_dir.rglob("*.tar.gz"))
    if not shard_glob:
        print(f"[WARN] No shards found under {args.speech_dir} (*.tar or *.tar.gz)")

    # Import after setting sys.path; avoids import failures
    # Try to load 'gcc' module to satisfy libstdc++ for pyroomacoustics
    maybe_load_env_module("gcc")
    from sidon.data.preprocess.dialogue_datamodule import DialogueDatasetDataModule  # noqa: E402

    print(f"[INFO] Loading defaults from: {args.config}")
    defaults = load_datamodule_defaults(args.config)
    dm_kwargs = build_dm_kwargs(defaults, args.speech_dir, args.train_num_workers, args.val_num_workers)

    # Summarize key kwargs
    summary_keys = [
        "sampling_rate", "batch_size", "ssl_model_name", "speaker_ssl_model_name",
        "use_noise", "max_duration", "merge_samples",
        "train_num_workers", "val_num_workers", "n_repeats", "split_by_worker",
    ]
    print("[INFO] Datamodule kwargs (subset):")
    for k in summary_keys:
        if k in dm_kwargs:
            print(f"  - {k}: {dm_kwargs[k]}")
    print(f"  - train_wds_patterns: {dm_kwargs.get('train_wds_patterns')}")
    print(f"  - val_wds_patterns:   {dm_kwargs.get('val_wds_patterns')}")

    # Instantiate and setup
    try:
        dm = DialogueDatasetDataModule(**dm_kwargs)
    except Exception as e:
        print("[ERROR] Failed to instantiate DialogueDatasetDataModule:")
        traceback.print_exc()
        return 1

    try:
        dm.setup("fit")
        print("[OK] setup('fit') completed")
    except Exception:
        print("[ERROR] dm.setup('fit') raised an exception:")
        traceback.print_exc()
        return 1

    # Train loader check
    try:
        train_loader = dm.train_dataloader()
        print("[OK] train_dataloader() constructed")
        for i, batch in enumerate(train_loader):
            print(f"[OK] Got train batch {i}")
            for line in describe_batch("train", batch):
                print(line)
            if i + 1 >= args.max_batches:
                break
    except Exception:
        print("[ERROR] Iterating train_dataloader raised an exception:")
        traceback.print_exc()
        return 1

    # Val loader check
    try:
        val_loader = dm.val_dataloader()
        print("[OK] val_dataloader() constructed")
        for i, batch in enumerate(val_loader):
            print(f"[OK] Got val batch {i}")
            for line in describe_batch("val", batch):
                print(line)
            if i + 1 >= args.max_batches:
                break
    except Exception:
        print("[ERROR] Iterating val_dataloader raised an exception:")
        traceback.print_exc()
        return 1

    print("[DONE] DialogueDatasetDataModule check completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
