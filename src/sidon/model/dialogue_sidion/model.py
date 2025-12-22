import torch
import torch.nn as nn
from transformers import Wav2Vec2BertModel


class FiLM(nn.Module):
    """
    FiLM layer: Feature-wise Linear Modulation

    x:    (B, C, ...)   – features to modulate
    cond: (B, D)        – conditioning vector

    Applies: y = gamma(cond) * x + beta(cond),
    where gamma, beta are per-channel.
    """
    def __init__(self, in_channels: int, cond_dim: int):
        super().__init__()
        # One linear to predict both gamma and beta
        self.film = nn.Linear(cond_dim, 2 * in_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, *spatial dims*)
        cond: (B, cond_dim)
        """
        B, C = x.size(0), x.size(1)

        # (B, 2C)
        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, C)

        # reshape to broadcast over spatial dims
        # -> (B, C, 1, 1, ..., 1)
        broadcast_shape = [B, C] + [1] * (x.dim() - 2)
        gamma = gamma.view(broadcast_shape)
        beta = beta.view(broadcast_shape)

        return gamma * x + beta

# --- adapter block (same as before) ---
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
    def forward(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        if c is None:  # allow condition-drop
            return x
        if c.dim() == 2:  # [B,Dc] -> [B,T,Dc]
            c = c[:, None, :].expand(x.size(0), x.size(1), -1)
        h = torch.nn.functional.gelu(self.down(x))
        g, b = self.film(c).chunk(2, dim=-1)
        h = h * (1 + g) + b
        return x + self.up(h)

def _get_encoder_and_layers(model: nn.Module):
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "layers"):
        return enc, enc.layers
    raise ValueError("Expected Wav2Vec2BertModel with .encoder.layers")

def add_cond_adapters_all_layers(
    model: Wav2Vec2BertModel,
    d_cond: int,
    *,
    hidden: int = 256,
    skip_first: int = 0,
    skip_last: int = 0,
):
    """
    Inserts a post-FFN CondAdapter into *all* encoder layers (minus any skipped at head/tail).
    You can then call the returned wrapper with forward(..., cond=...).
    """
    enc, layers = _get_encoder_and_layers(model)
    d_model = int(model.config.hidden_size)
    n = len(layers)

    start = max(0, skip_first)
    end   = n - max(0, skip_last)
    target_idxs = range(start, end)

    for i in target_idxs:
        layer = layers[i]
        if not hasattr(layer, "cond_adapter"):
            layer.register_module("cond_adapter", CondAdapter(d_model, hidden, d_cond))
        if not hasattr(layer, "_orig_forward"):
            layer._orig_forward = layer.forward

        def _wrapped_forward(self, *args, **kwargs):
            out = self._orig_forward(*args, **kwargs)
            if isinstance(out, tuple):
                hs, *rest = out
            else:
                hs, rest = out, []
            cond = getattr(self, "_current_cond", None)
            if cond is not None:
                hs = self.cond_adapter(hs, cond)  # <-- after FFN (post residual)
            return (hs, *rest) if rest else hs

        layer.forward = _wrapped_forward.__get__(layer, layer.__class__)  # bind

        class WithCondition(nn.Module):
            def __init__(self, base, enc, idxs):
                super().__init__()
                self.base = base
                self._enc = enc
                self._idxs = list(idxs)

                # <<< make legacy attribute access work >>>
                self.encoder = enc  # so .encoder.layers works (Wav2Vec2BertModel-style)

            @property
            def config(self):
                return self.base.config

            # Delegate everything else to the underlying HF model
            def __getattr__(self, name):
                # avoid recursion for our own fields
                if name in {"base", "_enc", "_idxs", "encoder"}:
                    return super().__getattr__(name)
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.base, name)

            def forward(self, *args, cond=None, **kwargs):
                for i in self._idxs:
                    getattr(self._enc.layers, str(i))._current_cond = cond
                out = self.base(*args, **kwargs)
                for i in self._idxs:
                    getattr(self._enc.layers, str(i))._current_cond = None
                return out

    return WithCondition(model, enc, target_idxs)

# ---- usage ----
if __name__ == "__main__":
    base = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    model = add_cond_adapters_all_layers(base, d_cond=128, hidden=256)
    wav = torch.randn(2, 16000)
    cond_global = torch.randn(2, 128)  # or frame-aligned [B, T_enc, 128]
    out = model(input_values=wav, cond=cond_global)
    print(out.last_hidden_state.shape)