#!/usr/bin/env python3
"""Inspect the sidon-v0.1 .pt files to learn how the weights are stored.

The conversion showed feature_extractor_cpu.pt is a TorchScript archive whose
state_dict() is empty (weights inlined as graph constants). This probes every
plausible extraction surface so we can pick a working one.
"""
import torch
from huggingface_hub import hf_hub_download

REPO = "sarulab-speech/sidon-v0.1"


def probe(fname: str) -> None:
    print(f"\n==================== {fname} ====================", flush=True)
    p = hf_hub_download(REPO, fname)
    # 1) TorchScript?
    try:
        m = torch.jit.load(p, map_location="cpu")
        print(f"[jit] loaded type={type(m).__name__}")
        sd = dict(m.state_dict())
        nps = dict(m.named_parameters())
        nbs = dict(m.named_buffers())
        print(f"[jit] state_dict={len(sd)} named_parameters={len(nps)} named_buffers={len(nbs)}")
        print(f"[jit] state_dict sample: {list(sd)[:10]}")
        print(f"[jit] named_parameters sample: {list(nps)[:10]}")
        print(f"[jit] named_buffers sample: {list(nbs)[:10]}")
        mods = [n for n, _ in m.named_modules()][:25]
        print(f"[jit] named_modules sample: {mods}")
        # frozen modules expose constants as attributes on _c
        try:
            attrs = [a for a in dir(m) if not a.startswith("__")][:40]
            print(f"[jit] dir sample: {attrs}")
        except Exception as e:  # noqa: BLE001
            print(f"[jit] dir failed: {e}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"[jit] load failed: {e}")
    # 2) torch.load fallback
    try:
        o = torch.load(p, map_location="cpu", weights_only=False)
        print(f"[torch.load] type={type(o)}")
        if isinstance(o, dict):
            print(f"[torch.load] dict keys sample: {list(o)[:10]}")
    except Exception as e:  # noqa: BLE001
        print(f"[torch.load] failed: {e}")


if __name__ == "__main__":
    probe("feature_extractor_cpu.pt")
    probe("decoder_cpu.pt")
    print("\n[inspect_v01] done.", flush=True)
