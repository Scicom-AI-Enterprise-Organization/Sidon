"""Lightning modules for Dialogue sidon models."""

from __future__ import annotations

from typing import Tuple

import audiotools
import random
import dac
import hydra
import torch
import torch.nn.functional as F
import transformers
import itertools
from lightning import LightningModule
import torchaudio
import torch.nn as nn
from lightning.pytorch import loggers
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from peft import LoraConfig, inject_adapter_in_model
from torchmetrics.audio import PermutationInvariantTraining
from .model import add_cond_adapters_all_layers, FiLM 
from .audio import extract_seamless_m4t_features
from typing import Optional, Union
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertBaseModelOutput

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
        self.teacher_ssl_model = transformers.Wav2Vec2BertModel.from_pretrained(
            cfg.ssl_model_name, num_hidden_layers=8
        ).eval()
        self.use_lora = cfg.get('use_lora', False)
        if self.use_lora:
            adapter_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="lora_only",
                target_modules='all-linear',
            )
            self.student_ssl_model = inject_adapter_in_model(
                adapter_config,
                self.student_ssl_model,
            )
        else:
            for param in self.student_ssl_model.parameters():
                param.requires_grad = True

        self.ssl_model_criterion = torch.nn.MSELoss()
        for param in self.teacher_ssl_model.parameters():
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
            teacher_features = self.teacher_ssl_model(**teacher_batch).last_hidden_state

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
        ssl_loss, predicted_features, target_features = self.step(batch, batch_idx, stage="train")
        return ssl_loss

    def validation_step(self, batch, batch_idx):
        ssl_loss, predicted_features, target_features = self.step(batch, batch_idx, stage="val")
        if self.global_rank == 0 and batch_idx < 10:
            vocoder_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="decoder_cuda.pt")
            vocoder = torch.jit.load(vocoder_path,map_location='cuda').to('cuda')
            n_channels = predicted_features.shape[0]
            predicted_wav = vocoder(predicted_features[:,0].transpose(1,2)).view(n_channels,-1)[:,:-960]
            target_wav = vocoder(target_features[:,0].transpose(1,2)).view(n_channels,-1)[:,:-960]
            self.log_audio(
                predicted_wav,
                f'Predicted dialogue {batch_idx}',
                48000
            )
            self.log_audio(
                target_wav,
                f'Resynthesized dialogue {batch_idx}',
                48000
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


class DialogueFeaturePredictorWithTransformerDiscriminatorLightningModule(
    DialogueFeaturePredictorLightningModule
):
    """Adversarial feature predictor that adds a transformer discriminator."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        disc_cfg = cfg.get("discriminator", {})
        hidden_size = int(disc_cfg.get("hidden_size", self.student_ssl_model.config.hidden_size))
        self.feature_discriminator = TransformerFeatureDiscriminator(
            hidden_size=hidden_size,
            num_layers=int(disc_cfg.get("num_layers", 2)),
            num_heads=int(disc_cfg.get("num_heads", 8)),
            ff_mult=float(disc_cfg.get("ff_mult", 4.0)),
            dropout=float(disc_cfg.get("dropout", 0.1)),
        )
        self.adv_weight = float(disc_cfg.get("adv_weight", 0.1))
        self.adv_criterion = torch.nn.BCEWithLogitsLoss()
        self.automatic_optimization = False

    def _flatten_for_discriminator(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        predicted_flat = predicted.permute(1, 0, 2, 3).reshape(
            -1, predicted.shape[2], predicted.shape[3]
        )
        target_flat = target.permute(1, 0, 2, 3).reshape(
            -1, target.shape[2], target.shape[3]
        )
        return predicted_flat, target_flat, None

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        ssl_loss, predicted_features, target_features = self.step(
            batch, batch_idx, stage="train"
        )
        predicted_flat, target_flat, attn_mask = self._flatten_for_discriminator(
            predicted_features, target_features
        )

        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        real_logits = self.feature_discriminator(
            target_flat.detach(), attention_mask=attn_mask
        )
        fake_logits = self.feature_discriminator(
            predicted_flat.detach(), attention_mask=attn_mask
        )
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        discriminator_loss = 0.5 * (
            self.adv_criterion(real_logits, real_labels)
            + self.adv_criterion(fake_logits, fake_labels)
        )
        self.manual_backward(discriminator_loss)
        torch.nn.utils.clip_grad_norm_(self.feature_discriminator.parameters(), 1.0)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        adv_logits = self.feature_discriminator(
            predicted_flat, attention_mask=attn_mask
        )
        adv_labels = torch.ones_like(adv_logits)
        adv_loss = self.adv_criterion(adv_logits, adv_labels)
        total_loss = ssl_loss + self.adv_weight * adv_loss
        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.student_ssl_model.parameters(), 1.0)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.log("train/ssl_loss", ssl_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/adv_loss",
            adv_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/discriminator_loss",
            discriminator_loss,
            on_step=True,
            on_epoch=True,
        )
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        ssl_loss, predicted_features, target_features = self.step(
            batch, batch_idx, stage="val"
        )
        predicted_flat, target_flat, attn_mask = self._flatten_for_discriminator(
            predicted_features, target_features
        )
        with torch.inference_mode():
            adv_logits = self.feature_discriminator(
                predicted_flat, attention_mask=attn_mask
            )
            adv_loss = self.adv_criterion(
                adv_logits, torch.ones_like(adv_logits)
            ).detach()
        self.log("val/ssl_loss", ssl_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/adv_loss", adv_loss, on_step=False, on_epoch=True, sync_dist=True)

        if self.global_rank == 0 and batch_idx < 10:
            vocoder_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="decoder_cuda.pt")
            vocoder = torch.jit.load(vocoder_path,map_location='cuda').to('cuda')
            n_channels = predicted_features.shape[0]
            predicted_wav = vocoder(predicted_features[:,0].transpose(1,2)).view(n_channels,-1)[:,:-960]
            target_wav = vocoder(target_features[:,0].transpose(1,2)).view(n_channels,-1)[:,:-960]
            self.log_audio(
                predicted_wav,
                f'Predicted dialogue {batch_idx}',
                48000
            )
            self.log_audio(
                target_wav,
                f'Resynthesized dialogue {batch_idx}',
                48000
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
        generator_opt = torch.optim.AdamW(
            self.student_ssl_model.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )
        discriminator_opt = torch.optim.AdamW(
            self.feature_discriminator.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )
        return [generator_opt, discriminator_opt]


class DialogueSidonLightningModule(LightningModule):
    """Sidon decoder/discriminator training using a frozen SSL encoder."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.feature_predictor = DialogueFeaturePredictorLightningModule.load_from_checkpoint(
            cfg.feature_predictor_checkpoint_path,
        ).eval()
        self.decoder = dac.model.dac.Decoder(
            input_channel=(self.feature_predictor.student_ssl_model.config.hidden_size*2),  # type: ignore # 2 channels 
            channels=1536,
            rates=[8, 5, 4, 3],
            d_out=2,
        )
        self.discriminator = GANLoss(
            dac.model.discriminator.Discriminator(sample_rate=cfg.sample_rate)
        )
        self.regression_loss = DACLoss(cfg.dac_loss)

        self.automatic_optimization = False
        voice_cfg = cfg.get("voice_activity", {})

        def _voice_cfg_get(key: str, default):
            if isinstance(voice_cfg, DictConfig):
                return voice_cfg.get(key, default)
            if isinstance(voice_cfg, dict):
                return voice_cfg.get(key, default)
            return getattr(voice_cfg, key, default) if hasattr(voice_cfg, key) else default

        self.voice_energy_threshold: float = float(_voice_cfg_get("energy_threshold", 3e-3))
        self.voice_window_ms: float = float(_voice_cfg_get("window_ms", 30.0))
        self.voice_context_ms: float = float(_voice_cfg_get("context_ms", 120.0))
        self.silence_weight: float = float(_voice_cfg_get("silence_weight", 0.1))

    def on_fit_start(self) -> None:
        torch.set_float32_matmul_precision("medium")

    def _compute_voice_mask(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return loss weights emphasizing voiced regions."""
        sr = self.cfg.sample_rate
        def _normalize_kernel(size_ms: float) -> int:
            size = max(int(size_ms * sr / 1000.0), 1)
            size = min(size, wav.shape[-1])
            if size % 2 == 0 and size > 1:
                size -= 1
            return max(size, 1)

        window_samples = _normalize_kernel(self.voice_window_ms)
        padding = window_samples // 2
        avg_power = F.avg_pool1d(
            wav.pow(2),
            kernel_size=window_samples,
            stride=1,
            padding=padding,
        )
        rms = torch.sqrt(avg_power + 1e-8)
        base_mask = (rms >= self.voice_energy_threshold).float()
        if self.voice_context_ms > 0:
            context_samples = _normalize_kernel(self.voice_context_ms)
            context_pad = context_samples // 2
            base_mask = F.max_pool1d(
                base_mask,
                kernel_size=context_samples,
                stride=1,
                padding=context_pad,
            )
            base_mask = (base_mask > 0).float()
        weight_mask = base_mask
        if self.silence_weight > 0:
            weight_mask = weight_mask + (1.0 - weight_mask) * self.silence_weight
        return weight_mask.to(wav.dtype), base_mask.mean()

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

    def step(
        self, batch, batch_idx: int, stage: str = "train"
    ) -> Tuple[torch.Tensor, audiotools.AudioSignal]:
        wavs = batch["input_wav"][:,:,:] # only use first channel
        batch_size = wavs.shape[0]
        opt_g, opt_d = self.optimizers()  # type: ignore
        sch_g, sch_d = self.lr_schedulers()  # type: ignore
        with torch.inference_mode():
            ssl_loss, predicted, target = self.feature_predictor.step(batch, batch_idx, stage="val",log=False)

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
        speech_mask, voiced_ratio = self._compute_voice_mask(wavs.audio_data)
        speech_mask = speech_mask.detach()
        self.log(
            f"{stage}/voiced_ratio",
            voiced_ratio.detach(),
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage == "train",
        )
        masked_predicted = audiotools.AudioSignal(
            predicted_clean_wavs.audio_data * speech_mask,
            sample_rate=self.cfg.sample_rate,
        )
        masked_target = audiotools.AudioSignal(
            wavs.audio_data * speech_mask,
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
            self.decoder.parameters(),
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
class CropSSLfeature(DialogueSidonLightningModule):

    def step(
        self, batch, batch_idx: int, stage: str = "train"
    ) -> Tuple[torch.Tensor, audiotools.AudioSignal]:
        wavs = batch["input_wav"][:,:,:] # only use first channel
        batch_size = wavs.shape[0]
        opt_g, opt_d = self.optimizers()  # type: ignore
        sch_g, sch_d = self.lr_schedulers()  # type: ignore
        with torch.inference_mode():
            ssl_loss, predicted, target = self.feature_predictor.step(batch, batch_idx, stage="val",log=False)

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
        speech_mask, voiced_ratio = self._compute_voice_mask(wavs.audio_data)
        speech_mask = speech_mask.detach()
        self.log(
            f"{stage}/voiced_ratio",
            voiced_ratio.detach(),
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage == "train",
        )
        masked_predicted = audiotools.AudioSignal(
            predicted_clean_wavs.audio_data * speech_mask,
            sample_rate=self.cfg.sample_rate,
        )
        masked_target = audiotools.AudioSignal(
            wavs.audio_data * speech_mask,
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
        )
        self.log(f"{stage}/total_loss", total_loss)
        if stage == "train":
            opt_g.zero_grad()
            self.manual_backward(total_loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt_g.step()
            sch_g.step()  # type: ignore
