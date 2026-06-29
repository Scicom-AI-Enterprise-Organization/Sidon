"""On the pod: report versions, extract a COMPLETE-but-slim FE adapter (LoRA +
the lora_only output_dense biases) and a decoder-only checkpoint, with self-checks."""
import torch, transformers, peft
from transformers import Wav2Vec2BertModel
from peft import LoraConfig, inject_adapter_in_model

print("transformers", transformers.__version__, "| peft", peft.__version__, flush=True)

# ---- FE: full student dict -> complete slim adapter -------------------------
fe = torch.load("/Sidon/fe_callcentre/last.pt", map_location="cpu")
student = fe["student"]
print("FE step", fe.get("step"), "| student tensors", len(student), flush=True)
l0 = [k for k in student if "layers.0." in k and "output_dense" in k]
print("layer0 output_dense keys:", l0, flush=True)

# trained params = LoRA + the (lora_only) biases of adapted output_dense layers
adapter = {k: v for k, v in student.items()
           if ("lora_" in k) or (("output_dense" in k) and k.endswith(".bias"))}
n_lora = sum("lora_" in k for k in adapter)
n_bias = sum(k.endswith(".bias") for k in adapter)
print(f"complete adapter: {len(adapter)} tensors (lora={n_lora}, bias={n_bias})", flush=True)

# is the trained bias actually different from a fresh base? (justifies including it)
base = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", num_hidden_layers=24, layerdrop=0.0)
bm = inject_adapter_in_model(LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64,
                             bias="lora_only", target_modules=["output_dense"]), base)
bsd = bm.state_dict()
bias_keys = [k for k in adapter if k.endswith(".bias")]
if bias_keys:
    import math
    diffs = [float((adapter[k] - bsd[k]).abs().max()) for k in bias_keys if k in bsd]
    print(f"bias delta vs base: max|Δ|={max(diffs):.4e} mean|Δ|={sum(diffs)/len(diffs):.4e} "
          f"(n={len(diffs)})", flush=True)
# self-check: loading complete adapter into the peft model -> 0 unexpected
res = bm.load_state_dict(adapter, strict=False)
print(f"load complete adapter: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
      flush=True)

torch.save({"step": fe.get("step"), "lora_alpha": 16, "r": 64, "dropout": 0.1,
            "target_modules": ["output_dense"], "layers": 24, "adapter": adapter,
            "note": "w2v-bert-2.0 24L + LoRA r64 (bias=lora_only). Build base + inject "
                    "LoRA, then load_state_dict(ckpt['adapter'], strict=False)."},
           "/Sidon/fe_callcentre/fe_adapter_full.pt")
print("saved /Sidon/fe_callcentre/fe_adapter_full.pt", flush=True)

# ---- decoder: drop optimizer/disc state -> decoder-only ----------------------
dc = torch.load("/Sidon/decoder_callcentre/last.pt", map_location="cpu")
print("decoder step", dc.get("step"), "| dec_channels", dc.get("dec_channels"), flush=True)
torch.save({"step": dc.get("step"), "dec_channels": dc.get("dec_channels", 3072),
            "decoder": dc["decoder"]}, "/Sidon/decoder_callcentre/decoder_only.pt")
print("saved /Sidon/decoder_callcentre/decoder_only.pt", flush=True)
print("DONE", flush=True)
