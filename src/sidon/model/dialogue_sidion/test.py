import torch
import torch.nn as nn
from transformers import Wav2Vec2BertModel

# ---------- 1) Conditional Adapter ----------
class CondAdapter(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, d_cond: int):
        super().__init__()
        self.down = nn.Linear(d_model, d_hidden)
        self.up   = nn.Linear(d_hidden, d_model)
        self.film = nn.Sequential(
            nn.Linear(d_cond, 2 * d_hidden),
            nn.SiLU(),
            nn.Linear(2 * d_hidden, 2 * d_hidden),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]; c: [B, Dc] (global) or [B, T, Dc] (frame-aligned)
        if c is None:
            return x
        if c.dim() == 2:
            c = c[:, None, :].expand(x.size(0), x.size(1), -1)
        h = torch.nn.functional.gelu(self.down(x))
        g, b = self.film(c).chunk(2, dim=-1)
        h = h * (1 + g) + b
        return x + self.up(h)

# ---------- 2) Locate encoder & layers for Wav2Vec2Bert (and friends) ----------
def _get_encoder_and_layers(model: nn.Module):
    # Wav2Vec2BertModel exposes `encoder` directly
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return model.encoder, model.encoder.layers

    # Fallbacks for other audio models (kept for portability)
    for attr in ["wav2vec2", "hubert", "wavlm", "data2vec_audio"]:
        if hasattr(model, attr) and hasattr(getattr(model, attr), "encoder"):
            enc = getattr(model, attr).encoder
            if hasattr(enc, "layers"):
                return enc, enc.layers
    raise ValueError("Could not find an Encoder with .layers in this model.")

# ---------- 3) Patch Wav2Vec2Bert with conditional adapters ----------
def add_cond_adapters_to_wav2vec2bert(
    model: Wav2Vec2BertModel,
    d_cond: int,
    *,
    hidden: int = 256,
    last_k_layers: int = 6,
):
    """
    Modifies the model in-place:
      - Adds a CondAdapter after the FFN of the last K encoder layers.
      - Lets you pass `cond=...` in the wrapped model's forward().
    """
    enc, layers = _get_encoder_and_layers(model)
    d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "hidden_dim")
    if not isinstance(d_model, int):
        raise ValueError("Couldn't infer hidden size from model.config")

    n = len(layers)
    target_idxs = range(max(0, n - last_k_layers), n)

    for i in target_idxs:
        layer = layers[i]
        # Attach adapter once
        if not hasattr(layer, "cond_adapter"):
            layer.register_module("cond_adapter", CondAdapter(d_model, hidden, d_cond))

        # Save original forward
        if not hasattr(layer, "_orig_forward"):
            layer._orig_forward = layer.forward

        def _wrapped_forward(self, *args, **kwargs):
            """
            For Wav2Vec2BertEncoderLayer:
              - original returns hidden_states (or tuple if output_attentions=True)
              - We post-apply the adapter to hidden_states.
            """
            out = self._orig_forward(*args, **kwargs)
            if isinstance(out, tuple):
                hs, *rest = out
            else:
                hs, rest = out, []
            cond = getattr(self, "_current_cond", None)
            if cond is not None:
                hs = self.cond_adapter(hs, cond)
            return (hs, *rest) if rest else hs

        layer.forward = _wrapped_forward.__get__(layer, layer.__class__)  # bind

    # Wrapper so you can pass `cond` into the forward call
    class WithCondition(nn.Module):
        def __init__(self, base, enc, target_idxs):
            super().__init__()
            self.base = base
            self._enc = enc
            self._idxs = list(target_idxs)

        @property
        def config(self):
            return self.base.config

        def forward(self, *args, cond: torch.Tensor = None, **kwargs):
            # Provide cond to patched layers for this call
            for i in self._idxs:
                getattr(self._enc.layers, str(i))._current_cond = cond
            out = self.base(*args, **kwargs)
            # Clean up
            for i in self._idxs:
                getattr(self._enc.layers, str(i))._current_cond = None
            return out

    return WithCondition(model, enc, target_idxs)

# ---------- 4) Minimal usage ----------
if __name__ == "__main__":
    base = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")  # or your checkpoint id
    model = add_cond_adapters_to_wav2vec2bert(base, d_cond=256, hidden=256, last_k_layers=6)

    B, T = 2, 16000
    wav = torch.randn(B, 100,160)

    # Example: global condition (e.g., language/speaker/SNR)
    cond_global = torch.randn(B, 256)

    out = model(wav, cond=cond_global)
    print(out.last_hidden_state.shape)  # [B, T', D]
    print(model)