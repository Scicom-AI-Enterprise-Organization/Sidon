"""Low-quality audio degradations for the NeuCodec speech-enhancement finetune.

Ported / adapted from Sidon's online degradation pipeline
(`/home/husein/ssd3/Sidon/src/sidon/data/preprocess/{degrations,functional_degrations}.py`).

The clean 44.1 kHz target is downsampled to 16 kHz (the frozen encoder's input
rate) and then run through a random chain of these degradations. The model must
learn to map the degraded 16 kHz input back to the clean 44.1 kHz target, i.e.
joint denoising + de-reverb + de-clipping + bandwidth extension.

Everything here operates on a mono float32 numpy array at a given sample rate and
runs inside the dataloader workers (one CPU thread each — see data_module), so it
must be cheap and dependency-light:

  - band_limit / clip / packet_loss : always available (numpy + torchaudio.resample)
  - codec (mp3)                      : needs torchaudio built with ffmpeg (wrapped)
  - reverb                          : needs `pyroomacoustics` (optional, off by default)
  - additive noise                  : needs a noise filelist (optional, off by default)

A `.done`-style failure is never fatal: any transform that raises returns the
input unchanged so a bad sample can't kill training.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
import soundfile as sf
import soxr
import torch
import torchaudio

try:  # mp3 codec degradation needs torchaudio's ffmpeg bindings
    from torchaudio.io import AudioEffector, CodecConfig
    _HAS_EFFECTOR = True
except Exception:  # noqa: BLE001
    _HAS_EFFECTOR = False

try:  # synthetic room reverb
    import pyroomacoustics as pra
    _HAS_PRA = True
except Exception:  # noqa: BLE001
    _HAS_PRA = False

try:  # telephone-band IIR filters (call-centre mode)
    from scipy.signal import butter, sosfilt
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _HAS_SCIPY = False


def _clipping(x: np.ndarray, min_q: float, max_q: float) -> np.ndarray:
    """Quantile-based hard clipping distortion (Sidon `clipping`)."""
    lo, hi = np.quantile(x, [min_q, max_q])
    return np.clip(x, lo, hi)


class Degrader:
    """Apply a random chain of degradations to a mono waveform.

    Construct once per dataloader worker (it builds the mp3 effector list and,
    optionally, loads the noise filelist). Call `degrade(wav, sr)`.
    """

    def __init__(self, cfg):
        # `cfg` is the OmegaConf `degrade:` block (dict-like). Use plain .get so
        # this also works with a regular dict.
        g = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: cfg.get(k, d))
        self.enable = bool(g("enable", True))
        self.p_reverb = float(g("prob_reverb", 0.0))
        self.p_noise = float(g("prob_noise", 0.0))
        self.p_band = float(g("prob_band_limit", 0.5))
        self.p_clip = float(g("prob_clip", 0.5))
        self.p_codec = float(g("prob_codec", 0.5))
        self.p_packet = float(g("prob_packet_loss", 0.3))
        self.band_srs = list(g("band_limit_srs", [4000, 6000, 8000, 11025, 12000]))
        qmin, qmax = list(g("codec_qscale", [1, 10]))
        self.codec_qscale = list(range(int(qmin), int(qmax) + 1))
        snr = list(g("noise_snr", [-5, 20]))
        self.noise_snr = (float(snr[0]), float(snr[1]))
        self.rt60_range = list(g("reverb_rt60", [0.1, 0.7]))

        # Noise corpus: a filelist (one audio path per line) sampled for additive
        # background noise. Off unless provided AND p_noise>0.
        self.noise_files: list[str] = []
        nf = g("noise_filelist", None)
        if nf:
            try:
                with open(nf) as f:
                    self.noise_files = [ln.strip().split("\t")[0] for ln in f if ln.strip()]
            except Exception as e:  # noqa: BLE001
                print(f"[degrade] could not read noise_filelist {nf}: {e}")

        self._effectors = None  # lazily built on first codec() call

        # Reverb backend: 'synthetic' (fast, dependency-free exponential-decay RIR)
        # or 'pra' (pyroomacoustics, more realistic). Falls back to synthetic if
        # pra is requested but unavailable. Noise uses the filelist if given, else
        # synthetic gaussian. => noise + reverb are ALWAYS available.
        self.use_pra = (str(g("reverb_backend", "synthetic")).lower() == "pra")
        if self.use_pra and not _HAS_PRA:
            print("[degrade] reverb_backend=pra but pyroomacoustics missing — using synthetic reverb")
            self.use_pra = False
        if self.p_codec > 0 and not _HAS_EFFECTOR:
            print("[degrade] prob_codec>0 but torchaudio AudioEffector unavailable — codec disabled")
            self.p_codec = 0.0

        # ---- call-centre / telephony channel (mode 'telephony' or 'mix') -------
        # Reproduces the real emgs samples: 8 kHz narrowband, telephone high-pass,
        # low-bitrate narrowband codec (GSM / G.711 mu-law) + 48-56 kbps MP3, light
        # line noise. GSM gives the muffled (-20dB edge ~1 kHz) days; mu-law/none
        # the brighter (~3.4 kHz) days — the mix brackets the measured spread.
        self.mode = str(g("mode", "generic")).lower()
        self.tel_share = float(g("telephony_share", 0.8))   # P(telephony) in mode='mix'
        tel = g("telephony", {}) or {}
        tg = tel.get if hasattr(tel, "get") else (lambda k, d=None: tel.get(k, d))
        self.tel_hp = list(tg("hp_hz", [200, 350]))
        self.tel_band_hz = list(tg("band_hz", [2800, 4200]))   # ~4 kHz narrowband (emgs-like)
        self.tel_codecs = list(tg("codecs", ["gsm", "mulaw", "none"]))
        self.tel_codec_w = list(tg("codec_weights", [0.6, 0.25, 0.15]))
        self.tel_mp3_kbps = list(tg("mp3_kbps", [16, 24, 32, 40]))
        self.tel_snr = list(tg("snr_db", [8, 28]))
        self.tel_noise_p = float(tg("noise_prob", 0.85))
        # Valid MP3 sample rates to bottleneck through (MPEG-2/2.5). The smallest
        # whose Nyquist covers the random band ceiling is picked per sample.
        self.tel_sr_choices = list(tg("sr_choices", [8000, 11025, 12000, 16000]))
        if self.mode in ("telephony", "mix") and not _HAS_EFFECTOR:
            print("[degrade] telephony mode needs torchaudio AudioEffector — falling back to generic")
            self.mode = "generic"

    # -- individual transforms (numpy in -> numpy out, mono) ------------------ #
    def _band_limit(self, x: np.ndarray, sr: int) -> np.ndarray:
        target = random.choice([s for s in self.band_srs if s < sr] or [sr])
        down = soxr.resample(x, sr, target)
        return soxr.resample(down, target, sr).astype(np.float32)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return _clipping(x, random.uniform(0.0, 0.1), random.uniform(0.9, 1.0)).astype(np.float32)

    def _packet_loss(self, x: np.ndarray, sr: int) -> np.ndarray:
        total = len(x) / sr
        n_chunks = int(total * 3 / 10)
        x = x.copy()
        for _ in range(n_chunks):
            dur = random.uniform(0.02, 0.2)
            if total - dur <= 0:
                break
            start = random.uniform(0, total - dur)
            i0, i1 = int(start * sr), int((start + dur) * sr)
            x[i0:i1] = 0.0
        return x

    def _codec(self, x: np.ndarray, sr: int) -> np.ndarray:
        if self._effectors is None:
            self._effectors = [
                AudioEffector(format="mp3", codec_config=CodecConfig(qscale=q))
                for q in self.codec_qscale
            ]
        effector = random.choice(self._effectors)
        wav = torch.from_numpy(x).float().view(-1, 1)  # AudioEffector wants [T, C]
        try:
            out = effector.apply(wav, sr)
        except Exception:  # noqa: BLE001
            return x
        out = out[:, 0].numpy().astype(np.float32)
        # mp3 adds encoder/decoder delay -> length drifts; align to input length.
        if len(out) >= len(x):
            return out[: len(x)]
        return np.pad(out, (0, len(x) - len(out)))

    def _mix_noise(self, x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        sig_p = float(np.mean(x ** 2)) + 1e-12
        noi_p = float(np.mean(noise ** 2)) + 1e-12
        scale = (sig_p / (noi_p * (10 ** (snr_db / 10.0)))) ** 0.5
        return (x + scale * noise).astype(np.float32)

    def _add_at_snr(self, x: np.ndarray, noise: np.ndarray) -> np.ndarray:
        return self._mix_noise(x, noise, random.uniform(*self.noise_snr))

    def _noise(self, x: np.ndarray, sr: int) -> np.ndarray:
        # real recorded noise if a corpus was given, else synthetic gaussian
        return self._noise_real(x, sr) if self.noise_files else self._noise_synth(x, sr)

    def _noise_synth(self, x: np.ndarray, sr: int) -> np.ndarray:
        n = np.random.randn(len(x)).astype(np.float32)
        if random.random() < 0.4:  # sometimes low-frequency / coloured (~brown)
            n = np.cumsum(n).astype(np.float32)
            n -= n.mean()
        return self._add_at_snr(x, n)

    def _noise_real(self, x: np.ndarray, sr: int) -> np.ndarray:
        path = random.choice(self.noise_files)
        try:
            noise, nsr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:  # noqa: BLE001
            return self._noise_synth(x, sr)
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        if nsr != sr:
            noise = soxr.resample(noise, nsr, sr).astype(np.float32)
        if len(noise) == 0:
            return self._noise_synth(x, sr)
        reps = len(x) // len(noise) + 1
        noise = np.tile(noise, reps)[: len(x)]
        return self._add_at_snr(x, noise)

    def _reverb(self, x: np.ndarray, sr: int) -> np.ndarray:
        return self._reverb_pra(x, sr) if self.use_pra else self._reverb_synth(x, sr)

    def _reverb_synth(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Fast dependency-free reverb: convolve with an exponential-decay noise
        RIR whose tail length is set by a random RT60."""
        rt60 = random.uniform(*self.rt60_range)
        length = int(sr * min(max(rt60, 0.05), 1.0))
        if length < 8:
            return x
        t = np.arange(length, dtype=np.float32)
        decay = np.exp(-6.9077 * t / (rt60 * sr)).astype(np.float32)  # -60 dB at rt60
        rir = np.random.randn(length).astype(np.float32) * decay
        rir[0] = 1.0  # direct path
        rir = rir / (np.abs(rir).max() + 1e-8)
        out = np.convolve(x, rir)[: len(x)].astype(np.float32)
        peak = float(np.abs(out).max())
        if peak > 1e-6:
            out = out * (float(np.abs(x).max()) / peak)  # roughly preserve level
        return out.astype(np.float32)

    def _reverb_pra(self, x: np.ndarray, sr: int) -> np.ndarray:
        rt60 = random.uniform(*self.rt60_range)
        for _ in range(5):
            try:
                room_dim = [random.uniform(2, 12) for _ in range(3)]
                e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
                room = pra.ShoeBox(room_dim, fs=sr, max_order=min(max_order, 12),
                                   materials=pra.Material(e_abs))
                m = 0.2
                room.add_source([random.uniform(m, d - m) for d in room_dim])
                room.add_microphone([[random.uniform(m, d - m)] for d in room_dim])
                room.compute_rir()
                rir = np.asarray(room.rir[0][0], dtype=np.float32)
                rir = rir / (np.abs(rir).max() + 1e-8)
                out = np.convolve(x, rir)[: len(x)].astype(np.float32)
                return out
            except Exception:  # noqa: BLE001
                continue
        return x

    # -- call-centre / telephony channel -------------------------------------- #
    def _effect(self, x: np.ndarray, sr: int, kw: dict) -> np.ndarray:
        """Apply one torchaudio AudioEffector (codec) at `sr`, length-aligned."""
        if not _HAS_EFFECTOR:
            return x
        wav = torch.from_numpy(np.ascontiguousarray(x)).float().view(-1, 1)
        try:
            out = AudioEffector(**kw).apply(wav, sr)[:, 0].numpy().astype(np.float32)
        except Exception:  # noqa: BLE001
            return x
        if len(out) >= len(x):
            return out[: len(x)]
        return np.pad(out, (0, len(x) - len(out)))

    def _highpass(self, x: np.ndarray, sr: int, fc: float) -> np.ndarray:
        if not _HAS_SCIPY or fc <= 0:
            return x
        sos = butter(2, min(fc, sr / 2 - 1) / (sr / 2), "highpass", output="sos")
        return sosfilt(sos, x).astype(np.float32)

    def _lowpass(self, x: np.ndarray, sr: int, fc: float) -> np.ndarray:
        if not _HAS_SCIPY or fc <= 0 or fc >= sr / 2:
            return x
        sos = butter(4, fc / (sr / 2), "lowpass", output="sos")
        return sosfilt(sos, x).astype(np.float32)

    def _pick_bottleneck(self, band: float) -> int:
        """Smallest valid MP3 sample rate whose Nyquist covers `band`."""
        for s in self.tel_sr_choices:
            if s / 2.0 >= band:
                return int(s)
        return int(self.tel_sr_choices[-1])

    def _codecs_for(self, ts: int):
        """GSM is 8 kHz-only -> drop it (renormalize) for higher bottleneck SRs."""
        cs, ws = [], []
        for c, w in zip(self.tel_codecs, self.tel_codec_w):
            if c == "gsm" and ts != 8000:
                continue
            cs.append(c); ws.append(w)
        return (cs, ws) if cs else (["none"], [1.0])

    def _telephony(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Call-centre channel: telephone high-pass -> RANDOM narrowband ceiling
        (< ~6 kHz, drawn per sample) via an 8/11/12/16 kHz bottleneck + lowpass ->
        low-rate codec (GSM=muffled, G.711 mu-law / none = brighter) -> 48-56 kbps
        MP3 -> light line noise -> back to `sr`. Mimics the emgs call recordings."""
        y = self._highpass(x, sr, random.uniform(*self.tel_hp))
        band = random.uniform(*self.tel_band_hz)          # random narrowband ceiling (< ~6 kHz)
        ts = self._pick_bottleneck(band)
        y = soxr.resample(y, sr, ts).astype(np.float32) if sr != ts else y.copy()
        y = self._lowpass(y, ts, band)                    # realize the random ceiling
        peak = float(np.abs(y).max())
        if peak > 1e-6:
            y = y / peak
        cs, ws = self._codecs_for(ts)
        codec = random.choices(cs, weights=ws, k=1)[0]
        if codec == "gsm":
            y = self._effect(y, ts, dict(format="gsm"))
        elif codec == "mulaw":
            y = self._effect(y, ts, dict(format="wav", encoder="pcm_mulaw"))
        # final low-bitrate MP3 container (matches the 48-56 kbps emgs files)
        br = int(random.choice(self.tel_mp3_kbps)) * 1000
        y = self._effect(y, ts, dict(format="mp3", codec_config=CodecConfig(bit_rate=br)))
        if random.random() < self.tel_noise_p:
            y = self._mix_noise(y, np.random.randn(len(y)).astype(np.float32),
                                random.uniform(*self.tel_snr))
        return soxr.resample(y, ts, sr).astype(np.float32) if sr != ts else y

    # -- chain ---------------------------------------------------------------- #
    def degrade(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Apply the degradation chain. mode='generic' -> random generic chain;
        'telephony' -> call-centre cascade; 'mix' -> per-sample pick (telephony with
        prob telephony_share, else generic). Returns mono float32, same length."""
        if not self.enable:
            return x
        x = np.asarray(x, dtype=np.float32)
        try:
            use_tel = self.mode == "telephony" or (
                self.mode == "mix" and random.random() < self.tel_share)
            if use_tel:
                x = self._telephony(x, sr)
                if self.p_packet and random.random() < self.p_packet:
                    x = self._packet_loss(x, sr)   # VoIP dropouts
            else:
                if self.p_reverb and random.random() < self.p_reverb:
                    x = self._reverb(x, sr)
                if self.p_noise and random.random() < self.p_noise:
                    x = self._noise(x, sr)
                if self.p_band and random.random() < self.p_band:
                    x = self._band_limit(x, sr)
                if self.p_clip and random.random() < self.p_clip:
                    x = self._clip(x)
                if self.p_codec and random.random() < self.p_codec:
                    x = self._codec(x, sr)
                if self.p_packet and random.random() < self.p_packet:
                    x = self._packet_loss(x, sr)
        except Exception as e:  # noqa: BLE001 — never let augmentation kill a sample
            print(f"[degrade] chain failed, using clean input: {e}")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(x, -1.0, 1.0).astype(np.float32)
