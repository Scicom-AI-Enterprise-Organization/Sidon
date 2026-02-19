"""Lightning modules for Dialogue sidon models."""

from __future__ import annotations

from typing import Tuple
import math

import audiotools
import random
import dac
import hydra
import torch
import torch.nn.functional as F
import transformers
import itertools
import torch.distributed as dist
from lightning import LightningModule
import torchaudio
import torch.nn as nn
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from lightning.pytorch import loggers
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from peft import LoraConfig, inject_adapter_in_model
from torchmetrics.audio import PermutationInvariantTraining
from peft import get_peft_model,LoraConfig
import diffusers
from .model import add_cond_adapters_all_layers, FiLM 
from .audio import extract_seamless_m4t_features
from typing import Optional, Union
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertBaseModelOutput
from sidon.model.flow_dialogue_sidon.lightning_module import SSLVAE

from sidon.model.losses import DACLoss, GANLoss


class TransformerFeatureDiscriminator(nn.Module):
    """Transformer encoder that scores SSL feature sequences as real or fake."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(hidden_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * ff_mult),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        x = self.input_proj(features)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        else:
            x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

class Wav2VecBertWithConditioning(transformers.Wav2Vec2BertModel):
    def setup_conditioning_layer(self,cond_emb_size,n_conditions):
        self.cond_emb = torch.nn.Embedding(n_conditions, cond_emb_size)
        self.film = FiLM(in_channels=self.config.hidden_size,cond_dim=cond_emb_size)
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        conditioning: Optional[torch.Tensor] = None
    ) -> Union[tuple, Wav2Vec2BertBaseModelOutput]:
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        """
        if conditioning is None:
            raise ValueError('Conditioning is required for Wav2VecBERTWithConditioning')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, extract_features = self.feature_projection(input_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )
        cond = self.cond_emb.forward(conditioning)
        hidden_states = self.film.forward(hidden_states.transpose(1,2),cond=cond).transpose(1,2)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.intermediate_ffn:
            expanded_hidden_states = self.intermediate_ffn(hidden_states)
            hidden_states = hidden_states + 0.5 * expanded_hidden_states

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BertBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            transformers.activations.ACT2FN["silu"],
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_proj = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.act_fn = transformers.activations.ACT2FN["silu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = self.act_fn(gate)
        return self.down_proj(gate * up)


class HeadLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        cond_dim: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.norm = RMSNorm(embed_dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(
            transformers.activations.ACT2FN["silu"],
            nn.Linear(cond_dim, 3 * embed_dim, bias=False),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate_ffn * self.ffn(_modulate(self.norm(x), shift_ffn, scale_ffn))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, cond_size: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Sequential(
            transformers.activations.ACT2FN["silu"],
            nn.Linear(cond_size, 2 * hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class VibeVoiceDiffusionHead(nn.Module):
    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        cond_size: int,
        head_layers: int = 6,
        head_ffn_ratio: float = 4.0,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.cond_dim = hidden_size
        self.noisy_images_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.cond_proj = nn.Linear(cond_size, self.cond_dim, bias=False)
        self.t_embedder = TimestepEmbedder(self.cond_dim)
        ffn_dim = int(hidden_size * head_ffn_ratio)
        self.layers = nn.ModuleList(
            [
                HeadLayer(
                    embed_dim=hidden_size,
                    ffn_dim=ffn_dim,
                    cond_dim=self.cond_dim,
                    norm_eps=rms_norm_eps,
                )
                for _ in range(head_layers)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            output_size=latent_size,
            cond_size=self.cond_dim,
            norm_eps=rms_norm_eps,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        x = self.noisy_images_proj(noisy_images)
        t = self.t_embedder(timesteps)
        condition = self.cond_proj(condition)
        c = condition + t
        for layer in self.layers:
            x = layer(x, c)
        x = self.final_layer(x, c)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        use_rotary: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"Per-head dimension ({self.head_dim}) must be even for rotary embeddings."
            )
        self.use_rotary = use_rotary
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * ffn_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(hidden_size * ffn_ratio), hidden_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]
        device = q.device
        dtype = q.dtype
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, half_dim, device=device, dtype=torch.float32)
                / float(half_dim)
            )
        )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).to(dtype=dtype)
        sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).to(dtype=dtype)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = _modulate(self.norm1(x), shift_attn, scale_attn)
        batch_size, seq_len, hidden_size = h.shape
        q = self.q_proj(h).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(batch_size, seq_len, self.num_heads, self.head_dim)
        if self.use_rotary:
            q, k = self._apply_rope(q, k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        h = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        h = h.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        h = self.out_proj(h)
        x = x + gate_attn * h
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp * h
        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiffusionTransformerHead(nn.Module):
    def __init__(
        self,
        latent_size: int,
        cond_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        use_positional: bool = True,
    ) -> None:
        super().__init__()
        self.use_positional = use_positional
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.latent_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.cond_proj = nn.Linear(cond_size, hidden_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    use_rotary=use_positional,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = DiTFinalLayer(hidden_size=hidden_size, output_size=latent_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = noisy_latents.shape
        x = self.latent_proj(noisy_latents)
        t_embed = self.t_embedder(timesteps).unsqueeze(1).expand(batch_size, seq_len, -1)
        c = self.cond_proj(conditioning) + t_embed
        for block in self.blocks:
            x = block(x, c)
        return self.final_layer(x, c)


class DialogueFeaturePredictorLightningModule(LightningModule):
    """Pre-trains a LoRA-adapted SSL model to mimic a frozen teacher."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.student_ssl_model = Wav2VecBertWithConditioning.from_pretrained(
            cfg.ssl_model_name, num_hidden_layers=16, layerdrop=0.0
        ).train()
        self.student_ssl_model.setup_conditioning_layer(256,2)
        self.vae = SSLVAE.load_from_checkpoint(
            cfg.vae_checkpoint_path
        ).eval()
        self.output_linear = torch.nn.Linear(self.student_ssl_model.config.hidden_size,
                                             self.vae.bottleneck.cfg.latent_dim)
        for param in self.student_ssl_model.parameters():
            param.requires_grad = True

        self.ssl_model_criterion = torch.nn.MSELoss()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.pit = PermutationInvariantTraining(self.ssl_model_criterion,'speaker-wise','min')

    def on_fit_start(self) -> None:
        torch.set_float32_matmul_precision("medium")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move only the tensors needed for the loss to the accelerator."""
        keys = (
            "noisy_16k_mixture",
            "input_wav",
        )

        def _move(value):
            if hasattr(value, "to"):
                return value.to(device=device, non_blocking=True)
            if isinstance(value, dict):
                return {k: _move(v) for k, v in value.items()}
            return value

        for key in keys:
            if key in batch:
                batch[key] = _move(batch[key])
        return batch

    def step(self, batch, batch_idx: int, stage: str = "train",log:bool=False) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        noisy_16k_mixture = batch["noisy_16k_mixture"]
        clean_input_wavs = batch["input_wav"]
        with torch.inference_mode():
            clean_input_wavs = torchaudio.functional.resample(clean_input_wavs,batch['sr'], 16_000)
            clean_ssl_inputs_0 = extract_seamless_m4t_features(
                [torch.nn.functional.pad(0.9 * clean_input_wav[0].view(-1) / clean_input_wav[0].abs().max(),(160,160)) for clean_input_wav in clean_input_wavs],
                device=str(self.device)
            )
            clean_ssl_inputs_1 = extract_seamless_m4t_features(
                [torch.nn.functional.pad(0.9 * clean_input_wav[1].view(-1) / clean_input_wav[1].abs().max(),(160,160)) for clean_input_wav in clean_input_wavs],
                device=str(self.device)
            )
            noisy_ssl_inputs = extract_seamless_m4t_features(
                [torch.nn.functional.pad(0.9 * noisy.view(-1) / noisy.abs().max(),(160,160)) for noisy in noisy_16k_mixture],
                device=str(self.device)
            )
        clean_inputs = [clean_ssl_inputs_0, clean_ssl_inputs_1]
        n_speakers = len(clean_inputs)
        batch_size = clean_ssl_inputs_0["input_features"].shape[0]

        def _concat_batchenc(enc_list: list[dict[str, torch.Tensor]]):
            out: dict[str, torch.Tensor] = {}
            for key, value in enc_list[0].items():
                if isinstance(value, torch.Tensor):
                    out[key] = torch.cat([enc[key] for enc in enc_list], dim=0)
                elif value is None:
                    out[key] = None
                else:
                    out[key] = value
            return out

        def _repeat_batchenc(enc: dict[str, torch.Tensor], repeats: int):
            out: dict[str, torch.Tensor] = {}
            for key, value in enc.items():
                if isinstance(value, torch.Tensor):
                    out[key] = torch.cat([value] * repeats, dim=0)
                elif value is None:
                    out[key] = None
                else:
                    out[key] = value
            return out

        teacher_batch = _concat_batchenc(clean_inputs)
        with torch.inference_mode():
            teacher_features,_,_,_ = self.vae.encode(teacher_batch)

        conditioning = (
            torch.arange(n_speakers, device=self.device)
            .unsqueeze(1)
            .expand(-1, batch_size)
            .reshape(-1)
        )
        noisy_batch = _repeat_batchenc(noisy_ssl_inputs, n_speakers)
        student_features = self.student_ssl_model(
            **noisy_batch, conditioning=conditioning
        ).last_hidden_state
        student_features = self.output_linear(student_features)

        p = student_features.view(n_speakers, batch_size, *student_features.shape[1:]).float()
        t = teacher_features.view(n_speakers, batch_size, *teacher_features.shape[1:]).float()
        ssl_loss = self.pit(p.transpose(0,1),t.transpose(0,1).clone())
        
        if log:
            self.log(
                f"{stage}/ssl_loss",
                ssl_loss,
                on_step=stage == "train",
                on_epoch=stage == "val",
                sync_dist=stage == "val",
            )

        return ssl_loss, p, t

    def training_step(self, batch, batch_idx):
        ssl_loss, predicted_features, target_features = self.step(batch, batch_idx, stage="train",log=True)
        return ssl_loss

    def validation_step(self, batch, batch_idx):
        ssl_loss, predicted_features, target_features = self.step(batch, batch_idx, stage="val",log=True)
        if self.global_rank == 0 and batch_idx < 10:
            n_channels = predicted_features.shape[0]
            predicted_wav = self.vae.decoder.forward(predicted_features[:,0].transpose(1,2)).view(n_channels,-1)
            target_wav = self.vae.decoder.forward(target_features[:,0].transpose(1,2)).view(n_channels,-1)
            self.log_audio(
                predicted_wav,
                f'Predicted dialogue {batch_idx}',
                24000,
            )
            self.log_audio(
                target_wav,
                f'Resynthesized dialogue {batch_idx}',
                24000
            )
            self.log_audio(
                batch['input_wav'][0],
                f'Ground truth dialogue {batch_idx}',
                24000
            )
            self.log_audio(
                batch['noisy_mixture'][0],
                f'Input noisy mixture {batch_idx}',
                24000
            )
        return ssl_loss

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2_000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    def log_audio(self, audio: torch.Tensor, name: str, sampling_rate: int) -> None:
        audio = audio.float().cpu().numpy().T
        for logger in self.loggers:
            if isinstance(logger, loggers.WandbLogger):
                import wandb

                wandb.log(
                    {name: wandb.Audio(audio, sample_rate=sampling_rate)},
                    step=self.global_step,
                )
            elif isinstance(logger, loggers.TensorBoardLogger):
                logger.experiment.add_audio(
                    name,
                    audio,
                    self.global_step,
                    sampling_rate,
                )



class DialogueSidonLightningModule(LightningModule):
    """Sidon decoder/discriminator training using a frozen SSL encoder."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.student_ssl_model = Wav2VecBertWithConditioning.from_pretrained(
            cfg.ssl_model_name, num_hidden_layers=16, layerdrop=0.0
        ).train()
        self.student_ssl_model.setup_conditioning_layer(256,2)
        self.vae = SSLVAE.load_from_checkpoint(
            cfg.vae_checkpoint_path
        ).eval()
        # remove deocoder in vae
        self.vae.decoder = None
        self.output_linear = torch.nn.Linear(self.student_ssl_model.config.hidden_size,
                                             self.vae.bottleneck.cfg.latent_dim)
        for param in self.student_ssl_model.parameters():
            param.requires_grad = True

        self.ssl_model_criterion = torch.nn.MSELoss()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.pit = PermutationInvariantTraining(self.ssl_model_criterion,'speaker-wise','min')
        self.decoder = dac.model.dac.Decoder(
            input_channel=(self.vae.bottleneck.cfg.latent_dim*2),  # type: ignore # 2 channels 
            channels=1536,
            rates=[8, 5, 4, 3],
            d_out=2,
        )
        self.discriminator = GANLoss(
            dac.model.discriminator.Discriminator(sample_rate=cfg.sample_rate)
        )
        self.regression_loss = DACLoss(cfg.dac_loss)

        self.automatic_optimization = False
    def on_fit_start(self) -> None:
        torch.set_float32_matmul_precision("medium")


    def _align_predicted_speakers(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """Ensure predicted speaker order matches clean reference."""
        n_speakers = predicted.shape[0]
        if n_speakers != 2:
            return predicted

        def _speaker_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            diff = (a - b).pow(2)
            return diff.view(diff.shape[0], diff.shape[1], -1).mean(dim=-1)

        same_cost = _speaker_distance(predicted, target).sum(dim=0)
        swapped = predicted.flip(0)
        swapped_cost = _speaker_distance(swapped, target).sum(dim=0)
        swap_mask = swapped_cost < same_cost
        if swap_mask.any():
            aligned = predicted.clone()
            aligned[:, swap_mask] = swapped[:, swap_mask]
        else:
            aligned = predicted
        swap_rate = swap_mask.float().mean()
        self.log(
            f"{stage}/speaker_swap_rate",
            swap_rate,
            on_step=stage == "train",
            on_epoch=True,
        )
        return aligned
    def feature_predictor_step(self, batch, batch_idx: int, stage: str = "train",log:bool=False) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        noisy_16k_mixture = batch["noisy_16k_mixture"]
        clean_input_wavs = batch["input_wav"]
        with torch.inference_mode():
            clean_input_wavs = torchaudio.functional.resample(clean_input_wavs,batch['sr'], 16_000)
            clean_ssl_inputs_0 = extract_seamless_m4t_features(
                [torch.nn.functional.pad(0.9 * clean_input_wav[0].view(-1) / clean_input_wav[0].abs().max(),(160,160)) for clean_input_wav in clean_input_wavs],
                device=str(self.device)
            )
            clean_ssl_inputs_1 = extract_seamless_m4t_features(
                [torch.nn.functional.pad(0.9 * clean_input_wav[1].view(-1) / clean_input_wav[1].abs().max(),(160,160)) for clean_input_wav in clean_input_wavs],
                device=str(self.device)
            )
            noisy_ssl_inputs = extract_seamless_m4t_features(
                [torch.nn.functional.pad(0.9 * noisy.view(-1) / noisy.abs().max(),(160,160)) for noisy in noisy_16k_mixture],
                device=str(self.device)
            )
        clean_inputs = [clean_ssl_inputs_0, clean_ssl_inputs_1]
        n_speakers = len(clean_inputs)
        batch_size = clean_ssl_inputs_0["input_features"].shape[0]

        def _concat_batchenc(enc_list: list[dict[str, torch.Tensor]]):
            out: dict[str, torch.Tensor] = {}
            for key, value in enc_list[0].items():
                if isinstance(value, torch.Tensor):
                    out[key] = torch.cat([enc[key] for enc in enc_list], dim=0)
                elif value is None:
                    out[key] = None
                else:
                    out[key] = value
            return out

        def _repeat_batchenc(enc: dict[str, torch.Tensor], repeats: int):
            out: dict[str, torch.Tensor] = {}
            for key, value in enc.items():
                if isinstance(value, torch.Tensor):
                    out[key] = torch.cat([value] * repeats, dim=0)
                elif value is None:
                    out[key] = None
                else:
                    out[key] = value
            return out

        teacher_batch = _concat_batchenc(clean_inputs)
        with torch.inference_mode():
            teacher_features,_,_,_ = self.vae.encode(teacher_batch)

        conditioning = (
            torch.arange(n_speakers, device=self.device)
            .unsqueeze(1)
            .expand(-1, batch_size)
            .reshape(-1)
        )
        noisy_batch = _repeat_batchenc(noisy_ssl_inputs, n_speakers)
        student_features = self.student_ssl_model(
            **noisy_batch, conditioning=conditioning
        ).last_hidden_state
        student_features = self.output_linear(student_features)

        p = student_features.view(n_speakers, batch_size, *student_features.shape[1:]).float()
        t = teacher_features.view(n_speakers, batch_size, *teacher_features.shape[1:]).float()
        ssl_loss = self.pit(p.transpose(0,1),t.transpose(0,1).clone())
        
        if log:
            self.log(
                f"{stage}/ssl_loss",
                ssl_loss,
                on_step=stage == "train",
                on_epoch=stage == "val",
                sync_dist=stage == "val",
            )

        return ssl_loss, p, t

    def step(
        self, batch, batch_idx: int, stage: str = "train"
    ) -> Tuple[torch.Tensor, Optional[audiotools.AudioSignal]]:
        wavs = batch["input_wav"][:,:,:] # only use first channel
        batch_size = wavs.shape[0]
        opt_g, opt_d = self.optimizers()  # type: ignore
        sch_g, sch_d = self.lr_schedulers()  # type: ignore
        ssl_loss, predicted, target = self.feature_predictor_step(batch, batch_idx, stage=stage,log=True)

        predicted = self._align_predicted_speakers(predicted, target, stage)

        input_features = torch.cat(
            [
                predicted[0].detach().transpose(1, 2),
                predicted[1].detach().transpose(1, 2),
            ],
            dim=1,
        )
        predicted_clean_wavs = self.decoder.forward(input_features)
        if abs(predicted_clean_wavs.shape[-1] - wavs.shape[-1]) > 480:
            raise ValueError("Predicted waveform length deviates too much from target")
        min_length = min(predicted_clean_wavs.shape[-1], wavs.shape[-1])
        predicted_clean_wavs = predicted_clean_wavs[:, :, :min_length]
        wavs = wavs[:, :, :min_length]

        predicted_clean_wavs = audiotools.AudioSignal(
            predicted_clean_wavs.view(batch_size, 2, -1),
            sample_rate=self.cfg.sample_rate,
        )
        wavs = audiotools.AudioSignal(
            wavs.view(batch_size, 2, -1), sample_rate=self.cfg.sample_rate
        )
        masked_predicted = audiotools.AudioSignal(
            predicted_clean_wavs.audio_data,
            sample_rate=self.cfg.sample_rate,
        )
        masked_target = audiotools.AudioSignal(
            wavs.audio_data,
            sample_rate=self.cfg.sample_rate,
        )
        regression_loss = self.regression_loss(masked_target, masked_predicted)["mel_loss"]

        discriminator_loss = self.discriminator.discriminator_loss(
            masked_predicted.audio_data.detach()[:,None,0],
            masked_target.audio_data[:,None,0],
        ) + self.discriminator.discriminator_loss(
            masked_predicted.audio_data.detach()[:,None,1],
            masked_target.audio_data[:,None,1],
        )
        if stage == "train":
            opt_d.zero_grad()
            self.manual_backward(discriminator_loss)  # type: ignore
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt_d.step()
            sch_d.step()  # type: ignore
        adv_gen_0, adv_feature_0 = self.discriminator.generator_loss(
            masked_predicted.audio_data[:,None,0],
            masked_target.audio_data[:,None,0],
        )
        adv_gen_1, adv_feature_1 = self.discriminator.generator_loss(
            masked_predicted.audio_data[:,None,1],
            masked_target.audio_data[:,None,1],
        )
        adv_gen = adv_gen_0 + adv_gen_1
        adv_feature = adv_feature_0 + adv_feature_1

        self.log(
            f"{stage}/regression_loss", regression_loss, on_step=True, on_epoch=True
        )
        self.log(
            f"{stage}/discriminator_loss",
            discriminator_loss,
            on_step=True,
            on_epoch=True,
        )
        self.log(f"{stage}/adv_gen", adv_gen, on_step=stage == "train", on_epoch=True)
        self.log(
            f"{stage}/adv_feature", adv_feature, on_step=stage == "train", on_epoch=True
        )
        total_loss = (
            self.cfg.loss.loss_weight["regression_loss"] * regression_loss
            + self.cfg.loss.loss_weight["adv_gen"] * adv_gen
            + self.cfg.loss.loss_weight["adv_feature"] * adv_feature
            + self.cfg.loss.loss_weight["ssl_loss"] * ssl_loss
        )
        self.log(f"{stage}/total_loss", total_loss)
        if stage == "train":
            opt_g.zero_grad()
            self.manual_backward(total_loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt_g.step()
            sch_g.step()  # type: ignore

        return total_loss, predicted_clean_wavs

    def on_exception(self, exception: Exception) -> None:
        raise exception

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predicted_clean_wavs = self.step(batch, batch_idx, stage="val")
        if self.global_rank == 0 and batch_idx == 0:
            self.log_audio(
                predicted_clean_wavs.audio_data[0].detach(),
                "val/synthesized",
                self.cfg.sample_rate,
            )
            self.log_audio(
                batch["noisy_input_wav"][0].detach(),
                "val/noisy_input",
                self.cfg.sample_rate,
            )
            self.log_audio(
                batch["input_wav"][0].detach(),
                "val/original",
                self.cfg.sample_rate,
            )
        return loss

    def configure_optimizers(self):  # type: ignore
        opt_g = torch.optim.AdamW(
            itertools.chain(self.decoder.parameters(), self.student_ssl_model.parameters(), self.output_linear.parameters()),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
            betas=(0.8, 0.98),
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
            betas=(0.8, 0.98),
        )
        sch_g = hydra.utils.instantiate(
            self.cfg.scheduler.generator,  # type: ignore
            optimizer=opt_g,
        )
        sch_d = hydra.utils.instantiate(
            self.cfg.scheduler.discriminator,  # type: ignore
            optimizer=opt_d,
        )
        return (
            {
                "optimizer": opt_g,
                "lr_scheduler": sch_g,
            },
            {
                "optimizer": opt_d,
                "lr_scheduler": sch_d,
            },
        )

    def log_audio(self, audio: torch.Tensor, name: str, sampling_rate: int) -> None:
        audio = audio.float().cpu().numpy().T
        for logger in self.loggers:
            if isinstance(logger, loggers.WandbLogger):
                import wandb

                wandb.log(
                    {name: wandb.Audio(audio, sample_rate=sampling_rate)},
                    step=self.global_step,
                )
            elif isinstance(logger, loggers.TensorBoardLogger):
                logger.experiment.add_audio(
                    name,
                    audio,
                    self.global_step,
                    sampling_rate,
                )

    @torch.inference_mode()
    def predict_separated(
        self,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> Tuple[torch.Tensor, int]:
        """Predict channel-separated speech from a mono mixture using diffusion."""
        was_training = self.training
        self.eval()
        try:
            device = self.device
            wav = torch.as_tensor(wav, dtype=torch.float32, device=device)
            if wav.ndim == 2:
                if wav.shape[0] <= 2 and wav.shape[1] > 2:
                    wav = wav.mean(dim=0, keepdim=True)
                else:
                    # Assume shape is (batch, time) already.
                    pass
            elif wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                raise ValueError("wav must be 1D or 2D (time or batch/time)")

            if sample_rate != 16000:
                wav_16k = torchaudio.functional.resample(wav, sample_rate, 16_000)
            else:
                wav_16k = wav

            def _normalize_and_pad(signal: torch.Tensor) -> torch.Tensor:
                max_val = signal.abs().max().clamp_min(1e-6)
                return torch.nn.functional.pad(0.9 * signal / max_val, (160, 160))

            wav_list = [_normalize_and_pad(sample.view(-1)) for sample in wav_16k]
            noisy_ssl_inputs = extract_seamless_m4t_features(
                wav_list,
                device=str(device),
            )

            batch_size = noisy_ssl_inputs["input_features"].shape[0]
            n_speakers = 2
            conditioning = (
                torch.arange(n_speakers, device=device)
                .unsqueeze(1)
                .expand(-1, batch_size)
                .reshape(-1)
            )

            def _repeat_batchenc(enc: dict[str, torch.Tensor], repeats: int):
                out: dict[str, torch.Tensor] = {}
                for key, value in enc.items():
                    if isinstance(value, torch.Tensor):
                        out[key] = torch.cat([value] * repeats, dim=0)
                    elif value is None:
                        out[key] = None
                    else:
                        out[key] = value
                return out

            noisy_batch = _repeat_batchenc(noisy_ssl_inputs, n_speakers)
            student_features = self.student_ssl_model(
                **noisy_batch, conditioning=conditioning
            ).last_hidden_state
            student_features = self.output_linear(student_features)
            predicted = student_features.view(
                n_speakers, batch_size, *student_features.shape[1:]
            ).float()

            conditioning_latents = torch.cat([predicted[0], predicted[1]], dim=-1).detach()
            sampled_latents = self.sample_latents(
                conditioning_latents,
                conditioning_latents.shape[1],
            )
            predicted_clean_wavs = self._decode_latents(sampled_latents).audio_data

            if predicted_clean_wavs.shape[0] == 1:
                predicted_clean_wavs = predicted_clean_wavs[0]
            return predicted_clean_wavs, self.cfg.sample_rate
        finally:
            if was_training:
                self.train()

class DialogueSidonDiffusionLightningModule(LightningModule):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.student_ssl_model = transformers.Wav2Vec2BertModel.from_pretrained(
            cfg.ssl_model_name, num_hidden_layers=13, layerdrop=0.0
        ).train()

        if cfg.get('lora', True):
            self.student_ssl_model = get_peft_model(
                self.student_ssl_model,
                LoraConfig(
                    r=64,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules=["output_dense","intermediate_dense","linear_q", "linear_k","linear_v"],
                )
            )
        self.vae = SSLVAE.load_from_checkpoint(
            cfg.vae_checkpoint_path
        ).eval()
        # remove deocoder in vae
        self.output_linear1 = torch.nn.Linear(self.student_ssl_model.config.hidden_size,
                                             self.vae.bottleneck.cfg.latent_dim)
        self.output_linear2 = torch.nn.Linear(self.student_ssl_model.config.hidden_size,
                                             self.vae.bottleneck.cfg.latent_dim)

        self.ssl_model_criterion = torch.nn.MSELoss()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.pit = PermutationInvariantTraining(self.ssl_model_criterion,'speaker-wise','min')
        latent_size = self.vae.bottleneck.cfg.latent_dim * 2
        latent_norm_cfg = cfg.get("latent_normalization", {})
        self.enable_latent_normalization = latent_norm_cfg.get("enabled", True)
        self.latent_norm_eps = latent_norm_cfg.get("eps", 1e-6)
        self.register_buffer(
            "latent_norm_mean",
            torch.zeros((1, 1, latent_size), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "latent_norm_std",
            torch.ones((1, 1, latent_size), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "latent_norm_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        input_size = self.student_ssl_model.config.hidden_size + latent_size
        diffusion_head_cfg = cfg.get("diffusion_head", {})
        self.diffusion_head = DiffusionTransformerHead(
            latent_size=latent_size,
            cond_size=input_size,
            hidden_size=diffusion_head_cfg.get("hidden_size", 512),
            num_layers=diffusion_head_cfg.get("head_layers", 6),
            num_heads=diffusion_head_cfg.get("num_heads", 8),
            ffn_ratio=diffusion_head_cfg.get("head_ffn_ratio", 4.0),
            dropout=diffusion_head_cfg.get("dropout", 0.0),
            use_positional=diffusion_head_cfg.get("use_positional", False),
        )
        diffusion_cfg = cfg.get("diffusion", {})
        num_train_steps = diffusion_cfg.get("num_train_timesteps", 1000)
        prediction_type = diffusion_cfg.get("prediction_type", "epsilon")
        if prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(
                f"Unsupported diffusion prediction_type: {prediction_type}. "
                "Expected 'epsilon' or 'v_prediction'."
            )
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_steps,
            prediction_type=prediction_type,
        )

    def on_fit_start(self) -> None:
        torch.set_float32_matmul_precision("medium")

    def _is_latent_norm_initialized(self) -> bool:
        return bool(self.latent_norm_initialized.item())

    @torch.no_grad()
    def _maybe_init_latent_normalization(self, target_latents: torch.Tensor) -> None:
        if not self.enable_latent_normalization or self._is_latent_norm_initialized():
            return

        flat = target_latents.detach().float().reshape(-1, target_latents.shape[-1])
        local_count = torch.tensor(float(flat.shape[0]), device=flat.device, dtype=torch.float32)
        local_sum = flat.sum(dim=0)
        local_sq_sum = flat.square().sum(dim=0)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)

        count = local_count.clamp_min(1.0)
        mean = local_sum / count
        second_moment = local_sq_sum / count
        var = (second_moment - mean.square()).clamp_min(self.latent_norm_eps)
        std = torch.sqrt(var)

        self.latent_norm_mean.copy_(
            mean.view(1, 1, -1).to(
                device=self.latent_norm_mean.device,
                dtype=self.latent_norm_mean.dtype,
            )
        )
        self.latent_norm_std.copy_(
            std.view(1, 1, -1).to(
                device=self.latent_norm_std.device,
                dtype=self.latent_norm_std.dtype,
            )
        )
        self.latent_norm_initialized.fill_(True)

        self.log(
            "train/latent_norm_std_mean",
            self.latent_norm_std.mean(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.enable_latent_normalization or not self._is_latent_norm_initialized():
            return latents
        mean = self.latent_norm_mean.to(device=latents.device)
        std = self.latent_norm_std.to(device=latents.device)
        return ((latents.float() - mean) / std).to(dtype=latents.dtype)

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.enable_latent_normalization or not self._is_latent_norm_initialized():
            return latents
        mean = self.latent_norm_mean.to(device=latents.device)
        std = self.latent_norm_std.to(device=latents.device)
        return (latents.float() * std + mean).to(dtype=latents.dtype)


    def _align_predicted_speakers(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """Ensure predicted speaker order matches clean reference."""
        n_speakers = predicted.shape[0]
        if n_speakers != 2:
            return predicted

        def _speaker_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            diff = (a - b).pow(2)
            return diff.view(diff.shape[0], diff.shape[1], -1).mean(dim=-1)

        same_cost = _speaker_distance(predicted, target).sum(dim=0)
        swapped = predicted.flip(0)
        swapped_cost = _speaker_distance(swapped, target).sum(dim=0)
        swap_mask = swapped_cost < same_cost
        if swap_mask.any():
            aligned = predicted.clone()
            aligned[:, swap_mask] = swapped[:, swap_mask]
        else:
            aligned = predicted
        swap_rate = swap_mask.float().mean()
        self.log(
            f"{stage}/speaker_swap_rate",
            swap_rate,
            on_step=stage == "train",
            on_epoch=True,
        )
        return aligned,swap_mask
    def feature_predictor_step(self, batch, batch_idx: int, stage: str = "train",log:bool=False) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        noisy_16k_mixture = batch["noisy_16k_mixture"]
        clean_input_wavs = batch["input_wav"]
        with torch.inference_mode():
            clean_input_wavs = torchaudio.functional.resample(clean_input_wavs,batch['sr'], 16_000)
            clean_ssl_inputs_0 = extract_seamless_m4t_features(
                [torch.nn.functional.pad(clean_input_wav[0].view(-1),(160,160)) for clean_input_wav in clean_input_wavs],
                device=str(self.device)
            )
            clean_ssl_inputs_1 = extract_seamless_m4t_features(
                [torch.nn.functional.pad(clean_input_wav[1].view(-1),(160,160)) for clean_input_wav in clean_input_wavs],
                device=str(self.device)
            )
            noisy_ssl_inputs = extract_seamless_m4t_features(
                [torch.nn.functional.pad(0.9 * noisy.view(-1) / noisy.abs().max(),(160,160)) for noisy in noisy_16k_mixture],
                device=str(self.device)
            )
        clean_inputs = [clean_ssl_inputs_0, clean_ssl_inputs_1]
        n_speakers = len(clean_inputs)
        batch_size = clean_ssl_inputs_0["input_features"].shape[0]

        def _concat_batchenc(enc_list: list[dict[str, torch.Tensor]]):
            out: dict[str, torch.Tensor] = {}
            for key, value in enc_list[0].items():
                if isinstance(value, torch.Tensor):
                    out[key] = torch.cat([enc[key] for enc in enc_list], dim=0)
                elif value is None:
                    out[key] = None
                else:
                    out[key] = value
            return out

        def _repeat_batchenc(enc: dict[str, torch.Tensor], repeats: int):
            out: dict[str, torch.Tensor] = {}
            for key, value in enc.items():
                if isinstance(value, torch.Tensor):
                    out[key] = torch.cat([value] * repeats, dim=0)
                elif value is None:
                    out[key] = None
                else:
                    out[key] = value
            return out

        teacher_batch = _concat_batchenc(clean_inputs)
        with torch.inference_mode():
            teacher_features,_,_,_ = self.vae.encode(teacher_batch)

        student_features = self.student_ssl_model(
            **{k:v.clone() for k,v in noisy_ssl_inputs.items()}
        ).last_hidden_state
        student_features_small1 = self.output_linear1(student_features)
        student_features_small2 = self.output_linear2(student_features)
        student_features_small = torch.stack([student_features_small1,student_features_small2],dim=0)

        p = student_features_small
        t = teacher_features.view(n_speakers, batch_size, *teacher_features.shape[1:]).float()
        ssl_loss = self.pit(p.transpose(0,1),t.transpose(0,1).clone())
        
        if log:
            self.log(
                f"{stage}/ssl_loss",
                ssl_loss,
                on_step=stage == "train",
                on_epoch=stage == "val",
                sync_dist=stage == "val",
            )

        return ssl_loss, p, t, student_features

    def step(
        self, batch, batch_idx: int, stage: str = "train",sample=False
    ) -> Tuple[torch.Tensor, audiotools.AudioSignal]:
        ssl_loss, predicted, target,features = self.feature_predictor_step(batch, batch_idx, stage=stage,log=True)

        predicted,swap_mask = self._align_predicted_speakers(predicted, target, stage)
        target_latents = torch.cat([target[0], target[1]], dim=-1)
        predicted_latents = torch.cat([predicted[0], predicted[1]], dim=-1)
        if stage == "train":
            self._maybe_init_latent_normalization(target_latents)

        normalized_target_latents = self._normalize_latents(target_latents)
        normalized_predicted_latents = self._normalize_latents(predicted_latents)
        conditioning_latents = torch.cat([normalized_predicted_latents,features], dim=-1)

        diffusion_loss, _ = self._diffusion_loss(
            normalized_target_latents,
            conditioning_latents,
        )
        total_loss = ssl_loss + diffusion_loss
        self.log(
            f"{stage}/diffusion_loss",
            diffusion_loss,
            on_step=stage == "train",
            on_epoch=stage == "val",
            sync_dist=stage == "val",
        )

        predicted_clean_wavs = None
        resynthesis_wavs = None
        if  sample:
            sampled_latents = self.sample_latents(
                conditioning_latents,
                normalized_target_latents.shape[1],
            )
            predicted_clean_wavs = self._decode_latents(sampled_latents, normalized=True)
            resynthesis_wavs = self._decode_latents(target_latents)

        return total_loss, predicted_clean_wavs,resynthesis_wavs


    def training_step(self, batch, batch_idx):
        loss, _,_ = self.step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predicted_clean_wavs,resynthesis_wavs = self.step(batch, batch_idx, stage="val", sample=(batch_idx < 5))
        if self.global_rank == 0 and batch_idx < 5 and predicted_clean_wavs is not None:
            self.log_audio(
                predicted_clean_wavs.audio_data[0].detach(),
                f"val/synthesized_{batch_idx}",
                self.cfg.sample_rate,
            )
            self.log_audio(
                resynthesis_wavs.audio_data[0].detach(),
                f"val/resynthesized_{batch_idx}",
                self.cfg.sample_rate,
            )
            self.log_audio(
                batch["noisy_16k_mixture"][0].detach(),
                f"val/noisy_input_{batch_idx}",
                16_000,
            )
            self.log_audio(
                batch["input_wav"][0].detach(),
                f"val/original_{batch_idx}",
                self.cfg.sample_rate,
            )
        return loss

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2_000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _diffusion_loss(
        self,
        target_latents: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, latent_dim = target_latents.shape
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0,
            self.ddpm_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=target_latents.device,
            dtype=torch.long,
        )
        noisy_latents = self.ddpm_scheduler.add_noise(target_latents, noise, timesteps)
        predicted_noise = self.diffusion_head(noisy_latents, timesteps, conditioning)
        if self.ddpm_scheduler.config.prediction_type == "v_prediction":
            target = self.ddpm_scheduler.get_velocity(target_latents, noise, timesteps)
        else:
            target = noise
        diffusion_loss = F.mse_loss(predicted_noise, target)
        return diffusion_loss, predicted_noise

    @torch.inference_mode()
    def sample_latents(
        self,
        conditioning: torch.Tensor,
        seq_len: int,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, _, cond_dim = conditioning.shape
        latent_dim = self.vae.bottleneck.cfg.latent_dim * 2
        diffusion_cfg = self.cfg.get("diffusion", {})
        total_steps = num_steps or diffusion_cfg.get("num_inference_steps", 1000)
        self.ddpm_scheduler.set_timesteps(total_steps, device=conditioning.device)
        latents = torch.randn(
            (batch_size, seq_len, latent_dim),
            device=conditioning.device,
            dtype=conditioning.dtype,
        )
        for t in self.ddpm_scheduler.timesteps:
            t_batch = torch.full(
                (batch_size,),
                t,
                device=conditioning.device,
                dtype=torch.long,
            )
            model_output = self.diffusion_head(latents, t_batch, conditioning)
            latents = self.ddpm_scheduler.step(model_output, t, latents).prev_sample
        return latents

    def _decode_latents(self, latents: torch.Tensor, normalized: bool = False) -> audiotools.AudioSignal:
        if normalized:
            latents = self._denormalize_latents(latents)
        latent_dim = self.vae.bottleneck.cfg.latent_dim
        first_latent = latents[:,:,:latent_dim] 
        first_latent = first_latent.transpose(1, 2)
        predicted_clean_wavs_first = self.vae.decoder.forward(first_latent)
        second_latent = latents[:,:,latent_dim:] 
        second_latent = second_latent.transpose(1, 2)
        predicted_clean_wavs_second = self.vae.decoder.forward(second_latent)
        predicted_clean_wavs = torch.cat([predicted_clean_wavs_first,predicted_clean_wavs_second],dim=1)
        return audiotools.AudioSignal(
            predicted_clean_wavs,
            sample_rate=self.cfg.sample_rate,
        )

    def log_audio(self, audio: torch.Tensor, name: str, sampling_rate: int) -> None:
        audio = audio.float().cpu().numpy().T
        for logger in self.loggers:
            if isinstance(logger, loggers.WandbLogger):
                import wandb

                wandb.log(
                    {name: wandb.Audio(audio, sample_rate=sampling_rate)},
                    step=self.global_step,
                )
            elif isinstance(logger, loggers.TensorBoardLogger):
                logger.experiment.add_audio(
                    name,
                    audio,
                    self.global_step,
                    sampling_rate,
                )

    @torch.inference_mode()
    def predict_separated(
        self,
        wav: torch.Tensor,
        sample_rate: int,
        num_steps: int|None = None
    ) -> Tuple[torch.Tensor, int]:
        """Predict channel-separated speech from a mono mixture."""
        was_training = self.training
        self.eval()
        try:
            device = self.device
            wav = torch.as_tensor(wav, dtype=torch.float32, device=device)
            if wav.ndim == 2:
                if wav.shape[0] <= 2 and wav.shape[1] > 2:
                    wav = wav.mean(dim=0, keepdim=True)
                else:
                    # Assume shape is (batch, time) already.
                    pass
            elif wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                raise ValueError("wav must be 1D or 2D (time or batch/time)")

            if sample_rate != 16000:
                wav_16k = torchaudio.functional.resample(wav, sample_rate, 16_000)
            else:
                wav_16k = wav

            def _normalize_and_pad(signal: torch.Tensor) -> torch.Tensor:
                max_val = signal.abs().max().clamp_min(1e-6)
                return torch.nn.functional.pad(0.9 * signal / max_val, (160, 160))

            wav_list = [_normalize_and_pad(sample.view(-1)) for sample in wav_16k]
            noisy_ssl_inputs = extract_seamless_m4t_features(
                wav_list,
                device=str(device),
            )

            student_features = self.student_ssl_model(
                **{k: v.clone() for k, v in noisy_ssl_inputs.items()}
            ).last_hidden_state
            predicted_0 = self.output_linear1(student_features)
            predicted_1 = self.output_linear2(student_features)
            predicted = torch.stack([predicted_0, predicted_1], dim=0).float()
            predicted_latents = torch.cat([predicted[0], predicted[1]], dim=-1)

            conditioning = torch.cat(
                [self._normalize_latents(predicted_latents), student_features],
                dim=-1,
            )
            sampled_latents = self.sample_latents(
                conditioning,
                conditioning.shape[1],
                num_steps=num_steps
            )
            predicted_clean_wavs = self._decode_latents(sampled_latents, normalized=True).audio_data

            if predicted_clean_wavs.shape[0] == 1:
                predicted_clean_wavs = predicted_clean_wavs[0]
            return predicted_clean_wavs, self.cfg.sample_rate
        finally:
            if was_training:
                self.train()
