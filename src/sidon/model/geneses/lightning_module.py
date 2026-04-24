"""GENESES Lightning module adapted to dialogue_sidon dataset batches."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
import torchaudio
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from huggingface_hub import hf_hub_download
from lightning.pytorch import LightningModule, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
from omegaconf import DictConfig

from sidon.model.dialogue_sidion.audio import extract_seamless_m4t_features

from .components import MMDiT
from .dacvae import DACVAE
from .ssl_feature_extractor import SSLFeatureExtractor


class WrappedModel(ModelWrapper):
    """Adapter that matches flow_matching's velocity-model call signature."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        ssl_merged = extras.get("ssl_merged")
        if ssl_merged is None:
            raise ValueError("ssl_merged must be provided to WrappedModel.")
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 1 and t.numel() == 1:
            t = t.expand(x.shape[0])
        elif t.ndim > 1:
            t = t.reshape(-1)
        vae_1 = x[:, 0, :, :]
        vae_2 = x[:, 1, :, :]
        res_1, res_2 = self.model.forward(ssl_merged, t, vae_1, vae_2)
        return torch.stack([res_1, res_2], dim=1)


class GenesesLightningModule(LightningModule):
    """Flow-matching separator that can train on dialogue_sidon batch schema."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.model if "model" in cfg else cfg

        self.mmdit = MMDiT(**self.model_cfg.mmdit)
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        vae_cfg = self.model_cfg.vae
        vae_ckpt_path = self._resolve_vae_checkpoint(vae_cfg)
        self.dacvae = DACVAE(vae_ckpt_path)

        ssl_cfg = self.model_cfg.ssl_model
        self.ssl_feature_extractor = SSLFeatureExtractor(
            ssl_cfg.name,
            int(ssl_cfg.layer),
        )

        self.sample_rate = int(self.model_cfg.get("sample_rate", 24_000))
        self.vae_sample_rate = int(vae_cfg.get("sample_rate", self.sample_rate))
        self.ssl_sample_rate = int(ssl_cfg.get("sample_rate", 16_000))
        self.inference_num_frames = int(self.model_cfg.get("inference_num_frames", 500))
        self.inference_step_size = float(self.model_cfg.get("inference_step_size", 0.01))
        self.val_log_batches = int(self.model_cfg.get("val_log_batches", 4))
        self.save_hyperparameters()

    @staticmethod
    def _resolve_vae_checkpoint(vae_cfg: DictConfig) -> str:
        if "ckpt_path" in vae_cfg and vae_cfg.ckpt_path:
            return str(vae_cfg.ckpt_path)
        if "hf_hub" in vae_cfg and vae_cfg.hf_hub is not None:
            return hf_hub_download(
                repo_id=vae_cfg.hf_hub.repo_id,
                filename=vae_cfg.hf_hub.filename,
            )
        raise ValueError("VAE checkpoint is required via vae.ckpt_path or vae.hf_hub.")

    @staticmethod
    def _parse_sample_rate(sample_rate: object, fallback: int) -> int:
        if isinstance(sample_rate, torch.Tensor):
            if sample_rate.numel() == 0:
                return fallback
            return int(sample_rate.reshape(-1)[0].item())
        if isinstance(sample_rate, (list, tuple)):
            if len(sample_rate) == 0:
                return fallback
            return int(sample_rate[0])
        if sample_rate is None:
            return fallback
        return int(sample_rate)

    @staticmethod
    def _ensure_batch_time(wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 1:
            return wav.unsqueeze(0)
        if wav.ndim == 2:
            return wav
        if wav.ndim == 3:
            if wav.shape[1] == 1:
                return wav[:, 0, :]
            return wav.mean(dim=1)
        raise ValueError(f"Expected [B, T] or [B, C, T], got {tuple(wav.shape)}.")

    @staticmethod
    def _ensure_batch_channel_time(wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 1:
            return wav.unsqueeze(0).unsqueeze(0)
        if wav.ndim == 2:
            return wav.unsqueeze(1)
        if wav.ndim == 3:
            return wav
        raise ValueError(f"Expected [B, T] or [B, C, T], got {tuple(wav.shape)}.")

    @staticmethod
    def _ensure_decoded_batch_time(wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 1:
            return wav.unsqueeze(0)
        if wav.ndim == 2:
            return wav
        if wav.ndim == 3 and wav.shape[1] == 1:
            return wav[:, 0, :]
        raise ValueError(f"Unexpected decoded waveform shape: {tuple(wav.shape)}.")

    def _resample_bt(self, wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
        if orig_sr == new_sr:
            return wav
        return torchaudio.functional.resample(wav, orig_sr, new_sr)

    def _resample_bct(self, wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
        if orig_sr == new_sr:
            return wav
        batch_size, channels, time = wav.shape
        return torchaudio.functional.resample(
            wav.view(batch_size * channels, time),
            orig_sr,
            new_sr,
        ).view(batch_size, channels, -1)

    def _create_mask(self, lengths: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # reference: [B, C, T], lengths: [B] in T units
        seq_len = reference.shape[-1]
        positions = torch.arange(seq_len, device=reference.device).unsqueeze(0)
        lengths = lengths.to(reference.device).clamp(min=1, max=seq_len)
        mask_t = positions < lengths.unsqueeze(1)
        return mask_t.unsqueeze(1).to(dtype=reference.dtype)

    def _extract_target_and_mixture(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if "input_wav" not in batch:
            raise KeyError(
                "Batch must include `input_wav` from dialogue schema."
            )
        if "noisy_16k_mixture" not in batch:
            raise KeyError(
                "Batch must include `noisy_16k_mixture` as model input."
            )

        targets = self._ensure_batch_channel_time(batch["input_wav"].float())
        if targets.shape[1] < 2:
            raise ValueError(
                f"GENESES expects 2 speakers in `input_wav`, got shape {tuple(targets.shape)}."
            )
        targets = targets[:, :2, :]

        input_sr = self._parse_sample_rate(batch.get("sr"), fallback=self.sample_rate)
        lengths_obj = batch.get("input_wav_lens")
        if lengths_obj is None:
            lengths = torch.full(
                (targets.shape[0],),
                targets.shape[-1],
                dtype=torch.long,
                device=targets.device,
            )
        else:
            lengths = torch.as_tensor(
                lengths_obj,
                device=targets.device,
                dtype=torch.long,
            ).clamp(min=1, max=targets.shape[-1])
        noisy_16k = self._ensure_batch_time(batch["noisy_16k_mixture"].float())
        noisy_16k = self._resample_bt(noisy_16k, 16_000, self.ssl_sample_rate)
        if noisy_16k.shape[0] != targets.shape[0]:
            raise ValueError(
                "Batch size mismatch between `noisy_16k_mixture` and `input_wav`: "
                f"{noisy_16k.shape[0]} vs {targets.shape[0]}."
            )
        return targets, noisy_16k, lengths, input_sr

    def _extract_ssl_conditioning(
        self,
        noisy_16k: torch.Tensor,
    ) -> torch.Tensor:
        wav_list = []
        for signal in noisy_16k:
            max_val = signal.abs().max().clamp_min(1e-6)
            normalized = 0.9 * signal / max_val
            wav_list.append(F.pad(normalized.view(-1), (160, 160)))
        ssl_inputs = extract_seamless_m4t_features(
            wav_list,
            sampling_rate=self.ssl_sample_rate,
            device=str(self.device),
        )
        return self.ssl_feature_extractor(ssl_inputs)

    def _prepare_targets_for_vae(
        self,
        targets: torch.Tensor,
        lengths: torch.Tensor,
        input_sr: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets_for_vae = self._resample_bct(targets, input_sr, self.vae_sample_rate)
        if input_sr == self.vae_sample_rate:
            return targets_for_vae, lengths
        vae_lengths = torch.floor(
            lengths.to(torch.float32) * float(self.vae_sample_rate) / float(input_sr)
        ).to(dtype=torch.long)
        vae_lengths = vae_lengths.clamp(min=1, max=targets_for_vae.shape[-1])
        return targets_for_vae, vae_lengths

    def on_fit_start(self) -> None:
        self.dacvae.to(self.device)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        if "optimizer" in self.model_cfg and self.model_cfg.optimizer is not None:
            optimizer = hydra.utils.instantiate(
                self.model_cfg.optimizer,
                params=self.parameters(),
            )
        else:
            optim_cfg = self.model_cfg.get("optim", {})
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(optim_cfg.get("lr", 1e-4)),
                weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
            )

        if "lr_scheduler" in self.model_cfg and self.model_cfg.lr_scheduler is not None:
            lr_scheduler = hydra.utils.instantiate(
                self.model_cfg.lr_scheduler,
                optimizer=optimizer,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "val/loss",
            }
        return {"optimizer": optimizer}

    def sampling_t(self, batch_size: int, m: float = 0.0, s: float = 1.0) -> torch.Tensor:
        schema = self.model_cfg.get("t_sampling_schema", "uniform")
        if schema == "uniform":
            return torch.rand((batch_size,), device=self.device)
        if schema == "logit_normal":
            u = torch.randn((batch_size,), device=self.device) * s + m
            return torch.sigmoid(u)
        raise ValueError(f"Unknown t sampling schema: {schema}")

    def loss_fn(
        self,
        est_dxt1: torch.Tensor,
        est_dxt2: torch.Tensor,
        dxt_1: torch.Tensor,
        dxt_2: torch.Tensor,
    ) -> torch.Tensor:
        est = torch.stack([est_dxt1, est_dxt2], dim=1)
        src = torch.stack([dxt_1, dxt_2], dim=1)
        return F.mse_loss(est, src)

    def calc_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        targets, noisy_16k, lengths, input_sr = self._extract_target_and_mixture(batch)
        with torch.no_grad():
            targets_for_vae, lengths_for_vae = self._prepare_targets_for_vae(
                targets,
                lengths,
                input_sr,
            )
            vae_1 = self.dacvae.encode(targets_for_vae[:, 0, :])
            vae_2 = self.dacvae.encode(targets_for_vae[:, 1, :])
            ssl_merged = self._extract_ssl_conditioning(noisy_16k)

        batch_size = ssl_merged.size(0)
        latent_seq_len = vae_1.shape[-1]
        lengths_for_vae = lengths_for_vae.clamp(min=1, max=targets_for_vae.shape[-1])
        vae_lengths = torch.div(
            lengths_for_vae * latent_seq_len,
            targets_for_vae.shape[-1],
            rounding_mode="floor",
        ).clamp(min=1, max=latent_seq_len)
        mask = self._create_mask(vae_lengths, vae_1)

        t = self.sampling_t(batch_size)
        vae = torch.stack([vae_1, vae_2], dim=1)
        noise = torch.randn_like(vae, device=self.device)
        path_sample = self.path.sample(x_0=noise, x_1=vae, t=t)

        x_t_1 = path_sample.x_t[:, 0, :, :] * mask
        x_t_2 = path_sample.x_t[:, 1, :, :] * mask
        est_dxt_1, est_dxt_2 = self.mmdit.forward(
            ssl_merged,
            t,
            x_t_1.permute(0, 2, 1),
            x_t_2.permute(0, 2, 1),
        )

        loss = self.loss_fn(
            est_dxt_1.permute(0, 2, 1) * mask,
            est_dxt_2.permute(0, 2, 1) * mask,
            path_sample.dx_t[:, 0, :, :] * mask,
            path_sample.dx_t[:, 1, :, :] * mask,
        )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        del batch_idx
        loss = self.calc_loss(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        loss = self.calc_loss(batch)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if self.global_rank == 0 and batch_idx < self.val_log_batches:
            with torch.inference_mode():
                vae_1, vae_2 = self.forward(batch, step_size=self.inference_step_size)
                predicted_1 = self._ensure_decoded_batch_time(self.dacvae.decode(vae_1))
                predicted_2 = self._ensure_decoded_batch_time(self.dacvae.decode(vae_2))

                input_sr = self._parse_sample_rate(batch.get("sr"), fallback=self.sample_rate)
                target = self._ensure_batch_channel_time(batch["input_wav"].float())[:, :2, :]
                target = self._resample_bct(target, input_sr, self.vae_sample_rate)

                target_vae_1 = self.dacvae.encode(target[:, 0, :])
                target_vae_2 = self.dacvae.encode(target[:, 1, :])
                resynth_1 = self._ensure_decoded_batch_time(self.dacvae.decode(target_vae_1))
                resynth_2 = self._ensure_decoded_batch_time(self.dacvae.decode(target_vae_2))

                noisy_16k = self._ensure_batch_time(batch["noisy_16k_mixture"].float())
                target_stereo = torch.stack([target[0, 0], target[0, 1]], dim=0)
                predicted_stereo = torch.stack([predicted_1[0], predicted_2[0]], dim=0)
                resynth_stereo = torch.stack([resynth_1[0], resynth_2[0]], dim=0)

                min_stereo_len = min(
                    target_stereo.shape[-1],
                    predicted_stereo.shape[-1],
                    resynth_stereo.shape[-1],
                )
                target_stereo = target_stereo[:, :min_stereo_len]
                predicted_stereo = predicted_stereo[:, :min_stereo_len]
                resynth_stereo = resynth_stereo[:, :min_stereo_len]

            self.log_audio(
                noisy_16k[0].detach(),
                f"val/noisy_input_{batch_idx}",
                self.ssl_sample_rate,
            )
            self.log_audio(
                target_stereo.detach(),
                f"val/target_stereo_{batch_idx}",
                self.vae_sample_rate,
            )
            self.log_audio(
                predicted_stereo.detach(),
                f"val/predicted_stereo_{batch_idx}",
                self.vae_sample_rate,
            )
            self.log_audio(
                resynth_stereo.detach(),
                f"val/resynthesized_stereo_{batch_idx}",
                self.vae_sample_rate,
            )
        return loss

    @torch.inference_mode()
    def forward(
        self,
        batch: dict[str, Any],
        step_size: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "noisy_16k_mixture" not in batch:
            raise KeyError("Batch must include `noisy_16k_mixture` as model input.")
        noisy_16k = self._ensure_batch_time(batch["noisy_16k_mixture"].float())
        ssl_merged = self._extract_ssl_conditioning(noisy_16k)

        num_frames = self.inference_num_frames
        if "input_wav" in batch:
            targets = self._ensure_batch_channel_time(batch["input_wav"].float())
            if targets.shape[1] >= 1:
                input_sr = self._parse_sample_rate(batch.get("sr"), self.sample_rate)
                targets = self._resample_bct(
                    targets[:, :1, :],
                    input_sr,
                    self.vae_sample_rate,
                )
                ref_feature = self.dacvae.encode(targets[:, 0, :])
                num_frames = ref_feature.shape[-1]

        vae_size = (
            ssl_merged.size(0),
            num_frames,
            int(self.model_cfg.vae.hidden_size),
        )
        noise_1 = torch.randn(vae_size, device=self.device)
        noise_2 = torch.randn(vae_size, device=self.device)
        noise = torch.stack([noise_1, noise_2], dim=1)

        solver = ODESolver(velocity_model=WrappedModel(self.mmdit))
        time_grid = torch.tensor([0.0, 1.0], device=self.device)
        result = solver.sample(
            x_init=noise,
            step_size=step_size or self.inference_step_size,
            time_grid=time_grid,
            ssl_merged=ssl_merged,
        )
        if not isinstance(result, torch.Tensor):
            raise TypeError("Expected tensor output from ODESolver.sample.")

        vae_1 = result[:, 0, :, :].permute(0, 2, 1)
        vae_2 = result[:, 1, :, :].permute(0, 2, 1)
        return vae_1, vae_2

    @torch.inference_mode()
    def predict_separated(
        self,
        wav: torch.Tensor,
        sample_rate: int,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        wav = torch.as_tensor(wav, dtype=torch.float32, device=self.device)
        wav = self._ensure_batch_time(wav)
        if sample_rate != self.ssl_sample_rate:
            noisy_16k = self._resample_bt(wav, sample_rate, self.ssl_sample_rate)
        else:
            noisy_16k = wav

        batch = {"noisy_16k_mixture": noisy_16k, "sr": sample_rate}
        step_size = self.inference_step_size
        if num_steps is not None and num_steps > 0:
            step_size = 1.0 / float(num_steps)
        vae_1, vae_2 = self.forward(batch, step_size=step_size)

        wav_1 = self.dacvae.decode(vae_1).squeeze(1)
        wav_2 = self.dacvae.decode(vae_2).squeeze(1)
        separated = torch.stack([wav_1, wav_2], dim=1)
        if separated.shape[0] == 1:
            separated = separated[0]
        return separated, self.vae_sample_rate

    @torch.inference_mode()
    def separate_and_enhance(
        self,
        noisy_mixed_wav: torch.Tensor,
        sr: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        separated, out_sr = self.predict_separated(
            noisy_mixed_wav,
            sample_rate=sr,
        )
        if separated.ndim == 3:
            separated = separated[0]
        if separated.ndim != 2 or separated.shape[0] < 2:
            raise ValueError(
                f"Expected separated output shape [2, T], got {tuple(separated.shape)}."
            )
        return separated[0], separated[1], out_sr

    def log_audio(self, audio: torch.Tensor, name: str, sampling_rate: int) -> None:
        audio_np = audio.float().cpu().numpy().T
        for logger in self.loggers:
            if isinstance(logger, loggers.WandbLogger):
                import wandb

                wandb.log(
                    {name: wandb.Audio(audio_np, sample_rate=sampling_rate)},
                    step=self.global_step,
                )
            elif isinstance(logger, loggers.TensorBoardLogger):
                logger.experiment.add_audio(
                    name,
                    audio_np,
                    self.global_step,
                    sampling_rate,
                )
