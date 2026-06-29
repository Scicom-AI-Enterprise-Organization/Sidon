#!/usr/bin/env python3
"""Dry-run finetune that CONTINUES FROM sidon-v0.1's feature extractor.

Why this shape: the released sidon-v0.1 weights ship as *frozen TorchScript*
(`state_dict`/`named_parameters`/`named_buffers` all empty — weights are inlined
into the forward graph as constants), so they cannot be loaded back into the
trainable Sidon `nn.Module`s for a true weight-resume. What we CAN do — and what
stage-3 finetuning does conceptually (FE frozen, decoder trained) — is keep
v0.1's feature extractor FROZEN and train a fresh DAC decoder + GAN on top of its
denoised SSL features.

  noisy fbank --(frozen v0.1 FE, TorchScript)--> SSL features
              --(TRAINABLE DAC decoder)--------> 48 kHz waveform
  loss = multi-resolution mel (DACLoss) + adversarial/feature-matching (GANLoss)
         vs the clean target waveform.

A plain manual loop (no Lightning) keeps the dry run free of the cluster-specific
Trainer config (ABCI MPI plugin, DDP, manual-opt grad-clip rules).

Usage (on the pod, PYTHONPATH=/Sidon/src, HF_HOME=/hf_cache):
  python dryrun_v01_decoder.py --steps 20 --batch 2 --max-duration 10
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

import audiotools
import dac

from sidon.model.losses import DACLoss, GANLoss
from sidon.data.preprocess import WebDatasetDataModule

REPO = "sarulab-speech/sidon-v0.1"
SSL_MODEL = "facebook/w2v-bert-2.0"
HIDDEN = 1024  # w2v-bert-2.0 hidden size == DAC decoder input_channel
SR = 48_000

DAC_LOSS_CFG = OmegaConf.create({
    "stft_loss": {"window_lengths": [2048, 512]},
    "mel_loss": {
        "n_mels": [5, 10, 20, 40, 80, 160, 320],
        "window_lengths": [32, 64, 128, 256, 512, 1024, 2048],
        "mel_fmin": [0, 0, 0, 0, 0, 0, 0],
        "mel_fmax": [None, None, None, None, None, None, None],
        "pow": 1.0, "clamp_eps": 1.0e-5, "mag_weight": 0.0,
    },
})
W = {"regression_loss": 15.0, "adv_gen": 2.0, "adv_feature": 1.0}


def log(m: str) -> None:
    print(m, flush=True)


def fe_features(fe, input_features: torch.Tensor) -> torch.Tensor:
    """Call the frozen TorchScript FE and return SSL hidden states [B,T,HIDDEN]."""
    out = fe(input_features)
    if isinstance(out, dict):
        if "last_hidden_state" in out:
            return out["last_hidden_state"]
        for v in out.values():  # first 3-D, HIDDEN-wide tensor
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[-1] == HIDDEN:
                return v
        return next(iter(out.values()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-duration", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save", default="/Sidon/ckpt_v0.1/dryrun_decoder.pt")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")

    # 1) frozen v0.1 feature extractor (TorchScript)
    log(f"[load] {REPO} feature_extractor (frozen TorchScript) …")
    fe = torch.jit.load(hf_hub_download(REPO, "feature_extractor_cuda.pt"), map_location=dev).eval()

    # 2) fresh trainable DAC decoder + discriminator (same arch as Sidon)
    decoder = dac.model.dac.Decoder(
        input_channel=HIDDEN, channels=1536, rates=[8, 5, 4, 3, 2]
    ).to(dev).train()
    disc = GANLoss(dac.model.discriminator.Discriminator(sample_rate=SR)).to(dev).train()
    reg_loss = DACLoss(DAC_LOSS_CFG)
    n_params = sum(p.numel() for p in decoder.parameters())
    log(f"[model] trainable DAC decoder params: {n_params/1e6:.1f}M  (FE is frozen v0.1)")

    opt_g = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.8, 0.98))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.8, 0.98))

    # 3) data: online degradation from the packed sg-podcast shards
    log("[data] building WebDatasetDataModule (sg_wds, online degradation) …")
    dm = WebDatasetDataModule(
        train_wds_patterns=["/data/sg_wds/train"], val_wds_patterns=["/data/sg_wds/valid"],
        batch_size=a.batch, sampling_rate=SR, ssl_model_name=SSL_MODEL,
        speaker_ssl_model_name=None, use_noise=True, max_duration=a.max_duration,
        merge_samples=True, train_num_workers=2, val_num_workers=1,
        noise_path=["/data/noise_wds"], use_pra=True, n_repeats=1,
        split_by_worker=False, add_squim_sdr=False,
    )
    dm.setup()
    loader = dm.train_dataloader()

    log(f"[train] dry-run: {a.steps} steps, batch={a.batch}, {a.max_duration}s @ {SR} Hz")
    step = 0
    t0 = time.time()
    for batch in loader:
        input_features = batch["noisy_ssl_inputs"]["input_features"].to(dev)
        with torch.no_grad():
            feats = fe_features(fe, input_features).detach()          # [B,T,HIDDEN]
        pred = decoder(feats.transpose(1, 2))                         # [B,1,T']
        target = batch["input_wav"].to(dev)
        ml = min(pred.shape[-1], target.shape[-1])
        bsz = target.shape[0]
        pred_sig = audiotools.AudioSignal(pred[..., :ml].reshape(bsz, 1, -1), sample_rate=SR)
        tgt_sig = audiotools.AudioSignal(target[:, :ml].reshape(bsz, 1, -1), sample_rate=SR)

        mel = reg_loss(tgt_sig, pred_sig)["mel_loss"]
        d_loss = disc.discriminator_loss(pred_sig.audio_data.detach(), tgt_sig.audio_data)
        opt_d.zero_grad(); d_loss.backward(); torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0); opt_d.step()

        adv_gen, adv_feat = disc.generator_loss(pred_sig.audio_data, tgt_sig.audio_data)
        g_loss = W["regression_loss"] * mel + W["adv_gen"] * adv_gen + W["adv_feature"] * adv_feat
        opt_g.zero_grad(); g_loss.backward(); torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0); opt_g.step()

        step += 1
        log(f"[step {step:>3}/{a.steps}] mel={mel.item():.4f} adv_gen={adv_gen.item():.4f} "
            f"adv_feat={adv_feat.item():.4f} d={d_loss.item():.4f} g={g_loss.item():.4f} "
            f"| feats={tuple(feats.shape)} pred={tuple(pred.shape)} {(time.time()-t0)/step:.2f}s/step")
        if step >= a.steps:
            break

    import os
    os.makedirs(os.path.dirname(a.save), exist_ok=True)
    torch.save({"decoder": decoder.state_dict()}, a.save)
    log(f"[done] dry-run finetune complete; trainable decoder saved -> {a.save}")


if __name__ == "__main__":
    main()
