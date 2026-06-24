#!/usr/bin/env python3
"""Probe sidon-v0.1 TorchScript forward signatures + graph-constant inventory.

The weights are frozen into the graph (state_dict empty). To (a) decide if order-
based constant extraction is even plausible and (b) learn how to *call* the
modules so we can use the frozen feature extractor as-is, dump:
  - forward schema + code head
  - count & shapes of tensor constants in the forward graph
"""
import torch
from collections import Counter
from huggingface_hub import hf_hub_download

REPO = "sarulab-speech/sidon-v0.1"


def probe(fname: str) -> None:
    print(f"\n==================== {fname} ====================", flush=True)
    p = hf_hub_download(REPO, fname)
    m = torch.jit.load(p, map_location="cpu").eval()
    try:
        print("[schema]", m.forward.schema)
    except Exception as e:  # noqa: BLE001
        print("[schema] n/a:", e)
    try:
        code = m.code
        print("[code head]\n" + "\n".join(code.splitlines()[:25]))
    except Exception as e:  # noqa: BLE001
        print("[code] n/a:", e)
    # graph constant tensor inventory
    try:
        g = m.graph
        shapes = []
        for node in g.nodes():
            if node.kind() == "prim::Constant":
                try:
                    val = node.output().toIValue()
                except Exception:
                    val = None
                if isinstance(val, torch.Tensor):
                    shapes.append(tuple(val.shape))
        print(f"[constants] tensor constants in top-level graph: {len(shapes)}")
        c = Counter(shapes)
        print(f"[constants] distinct shapes: {len(c)}; most common: {c.most_common(6)}")
    except Exception as e:  # noqa: BLE001
        print("[constants] n/a:", e)


if __name__ == "__main__":
    probe("feature_extractor_cpu.pt")
    probe("decoder_cpu.pt")
    print("\n[inspect_v01b] done.", flush=True)
