#!/usr/bin/env python3
"""Turn the released sarulab-speech/sidon-v0.1 *inference* weights into Lightning
checkpoints the Sidon training code can finetune from.

The HF repo ships inference artifacts, NOT Lightning `.ckpt` files:
  feature_extractor_{cpu,cuda}.pt  ~795 MB  -> w2v-BERT(8 layers) + LoRA student
  decoder_{cpu,cuda}.pt            ~210 MB  -> dac.model.dac.Decoder (48 kHz)

So a true optimizer-state *resume* is impossible; this is a weight-init
finetune ("load the weights, then continue training"). We:

  1. download the two .pt files,
  2. robustly extract a state_dict from each (plain state_dict / pickled module /
     TorchScript all handled),
  3. load the FE weights into a fresh `FeaturePredictorLightningModule`
     (`export.py` shows the released FE == this module's student_ssl_model, just
     wrapped by peft's `get_peft_model` so keys carry a `base_model.model.`
     prefix we strip), and the decoder weights into a fresh
     `SidonLightningModule(pretraining=True).decoder`,
  4. save two Lightning `.ckpt`s:
       feature_predictor.ckpt  (-> stage-3 cfg.ssl_model_name)
       vocoder_pretrain.ckpt   (-> stage-3 cfg.pretrain_path; decoder = released,
                                 GAN discriminator freshly initialised)

  stage-3 `sidon_vocoder_finetune` then loads both via its normal
  `load_from_checkpoint` paths — no training-code changes required.

Usage (on the pod, with PYTHONPATH=/Sidon/src and HF_HOME=/hf_cache):
  python build_ckpt_from_hf.py --out-dir /Sidon/ckpt_v0.1 [--device-variant cpu]
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ID = "sarulab-speech/sidon-v0.1"
SSL_MODEL = "facebook/w2v-bert-2.0"


def log(m: str) -> None:
    print(m, flush=True)


def _tensor_dict(d: dict) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, v) for k, v in d.items() if isinstance(v, torch.Tensor))


def extract_state_dict(path: str) -> "OrderedDict[str, torch.Tensor]":
    """Return a name->tensor mapping from any of the common save formats."""
    # 1) torch.load (plain state_dict, dict-wrapped, or a pickled nn.Module)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        log(f"[extract] torch.load failed ({e}); trying torch.jit.load")
        obj = None
    if obj is not None:
        if isinstance(obj, dict):
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return _tensor_dict(obj["state_dict"])
            td = _tensor_dict(obj)
            if td:
                return td
            # dict-of-dicts: pick the first nested tensor-dict
            for v in obj.values():
                if isinstance(v, dict):
                    td = _tensor_dict(v)
                    if td:
                        return td
        if hasattr(obj, "state_dict"):
            return _tensor_dict(dict(obj.state_dict()))
    # 2) TorchScript archive
    sm = torch.jit.load(path, map_location="cpu")
    return _tensor_dict(dict(sm.state_dict()))


def strip_prefixes(sd: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    """Normalise FE keys to match `student_ssl_model.state_dict()` naming.

    Released FE was saved from peft's get_peft_model -> keys look like
    `base_model.model.<...>`; the training module injects adapters in-place so
    its keys are just `<...>`. Some dumps also keep the `student_ssl_model.`
    Lightning prefix. Strip both, leading-only.
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in sd.items():
        nk = k
        for pref in ("student_ssl_model.", "base_model.model."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        # the peft wrapper can leave a doubled prefix; sweep once more
        if nk.startswith("base_model.model."):
            nk = nk[len("base_model.model."):]
        out[nk] = v
    return out


def report_load(name: str, module: torch.nn.Module,
                 sd: "OrderedDict[str, torch.Tensor]") -> int:
    target = module.state_dict()
    matched = [k for k in sd if k in target and target[k].shape == sd[k].shape]
    missing = [k for k in target if k not in sd]
    unexpected = [k for k in sd if k not in target]
    log(f"[{name}] released keys={len(sd)} | target keys={len(target)} | "
        f"matched={len(matched)} | missing={len(missing)} | unexpected={len(unexpected)}")
    for k in unexpected[:8]:
        log(f"[{name}]   unexpected: {k}")
    for k in missing[:8]:
        log(f"[{name}]   missing:    {k}")
    res = module.load_state_dict(sd, strict=False)
    if not matched:
        raise SystemExit(
            f"[{name}] 0 keys matched — remapping is wrong, refusing to write a "
            f"checkpoint that is effectively random init. Inspect the keys above."
        )
    return len(matched)


def save_lightning_ckpt(module: torch.nn.Module, cfg, out_path: Path) -> None:
    try:
        import lightning
        ver = getattr(lightning, "__version__", "2.5")
    except Exception:  # noqa: BLE001
        ver = "2.5"
    ckpt = {
        "state_dict": module.state_dict(),
        # __init__(self, cfg) -> load_from_checkpoint calls cls(cfg=<this>).
        "hyper_parameters": {"cfg": cfg},
        "pytorch-lightning_version": ver,
        "epoch": 0,
        "global_step": 0,
        "loops": {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    log(f"[save] {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/Sidon/ckpt_v0.1")
    ap.add_argument("--device-variant", choices=["cpu", "cuda"], default="cpu",
                    help="which released .pt variant to download (weights are identical)")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download

    from sidon.model import FeaturePredictorLightningModule, SidonLightningModule

    out_dir = Path(a.out_dir)
    v = a.device_variant
    log(f"[dl] downloading {REPO_ID} feature_extractor_{v}.pt + decoder_{v}.pt …")
    fe_path = hf_hub_download(REPO_ID, f"feature_extractor_{v}.pt")
    dec_path = hf_hub_download(REPO_ID, f"decoder_{v}.pt")

    # ---- feature predictor -------------------------------------------------
    log("\n===== feature predictor =====")
    cfg_fp = OmegaConf.create({
        "ssl_model_name": SSL_MODEL,
        "optim": {"lr": 2e-5, "weight_decay": 0.01},
    })
    lm_fp = FeaturePredictorLightningModule(cfg_fp)
    fe_sd = strip_prefixes(extract_state_dict(fe_path))
    report_load("FE", lm_fp.student_ssl_model, fe_sd)
    save_lightning_ckpt(lm_fp, cfg_fp, out_dir / "feature_predictor.ckpt")

    # ---- vocoder (decoder) -------------------------------------------------
    log("\n===== vocoder decoder =====")
    cfg_voc = OmegaConf.create({
        "ssl_model_name": SSL_MODEL,
        "sample_rate": 48000,
        "pretraining": True,
        "dac_loss": {
            "stft_loss": {"window_lengths": [2048, 512]},
            "mel_loss": {
                "n_mels": [5, 10, 20, 40, 80, 160, 320],
                "window_lengths": [32, 64, 128, 256, 512, 1024, 2048],
                "mel_fmin": [0, 0, 0, 0, 0, 0, 0],
                "mel_fmax": [None, None, None, None, None, None, None],
                "pow": 1.0, "clamp_eps": 1.0e-5, "mag_weight": 0.0,
            },
        },
        "loss": {"loss_weight": {
            "regression_loss": 15.0, "discriminator_loss": 1.0, "ssl_loss": 1.0,
            "adv_gen": 2.0, "adv_feature": 1.0,
        }},
        "optim": {"lr": 0.0001, "weight_decay": 0.01},
        "scheduler": {
            "generator": {"_target_": "torch.optim.lr_scheduler.ExponentialLR", "gamma": 0.999996},
            "discriminator": {"_target_": "torch.optim.lr_scheduler.ExponentialLR", "gamma": 0.999996},
        },
    })
    lm_voc = SidonLightningModule(cfg_voc)
    dec_sd = extract_state_dict(dec_path)
    dec_sd = OrderedDict(
        (k[len("decoder."):] if k.startswith("decoder.") else k, v) for k, v in dec_sd.items()
    )
    report_load("DEC", lm_voc.decoder, dec_sd)
    log("[DEC] note: GAN discriminator is freshly initialised (no weights in the release)")
    save_lightning_ckpt(lm_voc, cfg_voc, out_dir / "vocoder_pretrain.ckpt")

    log("\n[build_ckpt_from_hf] complete.")
    log(f"  feature_predictor.ckpt -> stage-3 model.cfg.ssl_model_name")
    log(f"  vocoder_pretrain.ckpt  -> stage-3 model.cfg.pretrain_path")


if __name__ == "__main__":
    main()
