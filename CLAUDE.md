# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sidon is a speech restoration research system. It pairs a LoRA-adapted w2v-BERT 2.0 feature predictor with a DAC-based vocoder to restore degraded speech. A separate branch (`dialogue`) extends the system with **GENESES** — a flow-matching two-speaker dialogue separator built on MMDiT + DACVAE.

> **Active task in this checkout:** a **call-centre / telephony restoration finetune** on
> **1× H100 (RunPod, US, SECURE, everything under `/`, never `/workspace`)**. Two stages, both
> trained from scratch (the released `sidon-v0.1` is frozen TorchScript — not loadable):
> **stage 1** LoRA-adapts the *full 24-layer* w2v-BERT 2.0 to map telephony-degraded → clean
> features (MSE distillation); **stage 2** trains a 188M DAC decoder + GAN to reconstruct clean
> 48 kHz from those features. Degradation is the realistic `runpod/degradations.py` `Degrader`
> (call-centre mimic). Teachers are a clean ≥44 kHz pool (EARS + Expresso + DNSMOS-filtered HF
> datasets + DNSMOS-filtered Malaysian/Singaporean podcast). Lives in `runpod/`; see
> "Call-centre finetune" below. (The earlier sidon-v0.1 dry-run is superseded — see "RunPod
> dry-run" for that history.)

## Call-centre finetune (current)

All under `runpod/`. **1× H100, US, SECURE, `/` only.** Drive long jobs detached
(`setsid … </dev/null >log 2>&1 &`) and poll the log — the RunPod SSH proxy is flaky.

**Pipeline (two stages, both use the realistic degradation + the clean teacher pool):**
```
stage 1 (FE):  clean 48k --16k--> Degrader --> (student 24L w2v-BERT + LoRA) --> H_student
               clean 48k --16k-->          --> (teacher 24L w2v-BERT, frozen) --> H_clean
               loss = MSE(H_student, H_clean)         # only ~16M LoRA trains
stage 2 (decoder):  telephony-degraded --(FROZEN stage-1 FE)--> features[T,1024]
                    --(TRAINABLE DAC decoder 188M)--> 48 kHz   loss = 15·mel + 2·adv + 1·feat (GAN)
```
- **Degradation** = `runpod/degradations.py` `Degrader` (ported from `../neucodec-44k-speech-enhancement`,
  tuned vs real `emgs` call samples). `DEGRADE_CFG` in both trainers: `mode=mix` (80% telephony /
  20% generic). Telephony = telephone HP → random <6 kHz narrowband ceiling (8/11/12/16k bottleneck)
  → GSM / G.711-µ-law codec → 16–40 kbps MP3 → line noise + VoIP dropouts. Needs `soxr`+`scipy`+ffmpeg
  (GSM/µ-law/mp3 codecs).
- **Teacher pool** `/data/clean48k/` (clean ≥44 kHz only — the model learns to *output* the teacher):
  `ears` + `expresso_{read,conv}` (studio), `extra/` (DNSMOS-filtered ≥44 kHz HF datasets via
  `clean_teacher_datasets.json`), `podcast_{sg,my}/` (DNSMOS-filtered Malaysian + Singaporean podcast
  chunks, bak ≥ 3.64). Built by `prepare_clean48k.py` (+`clean_extra` source) and `prepare_podcast_clean.py`.
- **Cleanliness metric** = DNSMOS P.835 `bak` (background-noise MOS) via `dnsmos_metric.py`
  (`speechmos`); baseline EARS+Expresso ≈ 3.64. Candidate ≥44 kHz HF datasets were probed/filtered
  by `../dataset/multilingual-tts/discover/noisefilter/`.
- **Released:** `Scicom-intl/sidon-callcentre` (FE adapter `fe_adapter_full.pt` + `decoder_only.pt`
  + `*_train_state.pt` with optimizer). Local **peft-free** inference: `runpod/infer_callcentre.py`
  (merges LoRA into the base). README: `runpod/README_callcentre.md`.

| run/launch | file |
|---|---|
| stage-1 FE train | `train_fe_callcentre.py` (24L w2v-BERT + LoRA, Degrader, map-style Dataset) |
| stage-2 decoder | `train_decoder_callcentre.py` (DAC decoder + GAN, frozen FE, bf16, grad-accum) |
| FE re-train wrapper | `go_fe_realdeg.sh` |  decoder launch | `go_decoder_b6w6.sh` (batch4/accum6/win6/3072ch) |
| clean data | `prepare_clean48k.py` (ears/expresso/clean_extra) · `prepare_podcast_clean.py` (`go_podcast.sh`) |
| inference | `infer_callcentre.py` · pod→HF push `pod_push_decoder.py` / `pod_prep_infer.py` |
| pod control | `launch_pod.py` (provision/status/terminate) · `bootstrap_fe.sh` (targeted venv) |

## Environment Setup

```bash
uv sync               # install all dependencies (full env incl. GENESES deps)
python -m compileall src   # quick syntax sweep before submitting cluster jobs
```

The project uses `uv` for dependency management (Python 3.10+, PyTorch 2.6+, CUDA required).

> The RunPod dry-run does **not** `uv sync`. It builds a *targeted* venv (`runpod/bootstrap.sh`)
> that skips `flash-attn` / `mmdit` / `flow-matching` / `diffusers` / `ray` — those are only needed
> for the GENESES *dialogue* path, and flash-attn's source build is slow/fragile. The single-speaker
> restoration pipeline (feature predictor + DAC vocoder) only needs the subset that bootstrap installs.

## Common Commands

### Training (canonical, cluster)
```bash
uv run python -m sidon.train model=sidon_feature_predictor data=preprocessed      # stage 1
uv run python -m sidon.train model=sidon_vocoder_pretrain  data=preprocessed      # stage 2
uv run python -m sidon.train model=sidon_vocoder_finetune  data=preprocessed_48k \ # stage 3
  model.cfg.ssl_model_name=/path/to/feature_predictor.ckpt \
  model.cfg.pretrain_path=/path/to/vocoder_pretrain.ckpt
uv run python -m sidon.train model=geneses_dialogue data=dialogue_preprocessed    # GENESES
uv run python -m sidon.train ... train.ckpt_path=/path/to/last.ckpt               # resume
```

### Preprocessing (generate WebDataset shards)
```bash
uv run python -m sidon.preprocess data=webdataset_preprocess_24k preprocess.writer_name=my_run
```

## RunPod dry-run (continue-finetune from sidon-v0.1)

**Goal:** prove the restoration training loop runs end-to-end on 1× H100, continued from the released
model, on a small slice of sg-podcast (same dataset source as `../neucodec-44k`). *"At least we have
something."* **Status: verified working** — 20 steps run, mel + adversarial losses backprop, a 52.4M-param
DAC decoder is trained and saved (`~0.75 s/step`).

**The released-weights reality (important).** `sarulab-speech/sidon-v0.1` ships `feature_extractor_{cpu,cuda}.pt`
(~795 MB) and `decoder_{cpu,cuda}.pt` (~210 MB), and **both are frozen TorchScript** — `state_dict`,
`named_parameters`, and `named_buffers` are all **empty**; the weights are inlined into the forward graph
as constants. So they **cannot be loaded back into the trainable Sidon `nn.Module`s** for a real
weight-resume. `runpod/build_ckpt_from_hf.py` demonstrates this and **refuses** to write a random-init
checkpoint (`0 keys matched`); `runpod/inspect_v01.py` / `inspect_v01b.py` probe the modules
(FE forward: `forward(input_features[B,T,160]) -> Dict[str,Tensor]`; decoder: `forward(x[B,1024,T]) -> wav`).

**What "continue from v0.1" therefore means here.** Keep v0.1's **frozen feature extractor** (it's used
frozen in stage-3 finetuning anyway — it's the 8-layer w2v-BERT+LoRA restoration "brain") and train a
**fresh DAC decoder + GAN** on top of its features. That is `runpod/dryrun_v01_direct.py`:

```
random mp3 window --(degrade)--> noisy 16k --(w2v-bert FE)--> input_features[160]
      --(frozen v0.1 FE, TorchScript)--> SSL features [T,1024]
      --(TRAINABLE DAC decoder)--------> 48 kHz waveform
loss = DAC multi-res mel + adversarial/feature-matching vs the clean window.
```

48 kHz because the released decoder is a 48 kHz model (hop `8·5·4·3·2 = 960` × 50 fps = 48 000). GENESES
(two-speaker dialogue) is excluded — sg-podcast is single-speaker.

**Direct dataloader, NOT WebDataset.** Sidon's normal pipeline trains from WebDataset `.tar` shards
(`sidon.data.preprocess.WebDatasetDataModule`, online degradation). For this dry run we skip all packing:
`dryrun_v01_direct.py` has a small `IterableDataset` that reads random windows **straight from the
extracted mp3s** via `ffmpeg -ss/-t` (long podcasts are never fully decoded), degrades in-process, and
runs the w2v-BERT feature extractor in `collate`. A manual training loop (no Lightning) keeps the dry run
free of the cluster-specific Trainer config. *(Packing mp3s into shards is fragile: ffmpeg's segment muxer
produces FLAC/WAV that libsndfile can't seek from a BytesIO → `psf_fseek failed`; if you ever do pack,
decode to one file then re-encode chunks via `soundfile`.)*

```bash
# 0. secrets in .env: RUNPOD_API_KEY, HF_TOKEN, WANDB_API_KEY (rsync'd to the pod)
# 1. provision 1x H100 SECURE, 500GB CONTAINER disk (mounted at /, volumeInGb=0 => no /workspace)
python3 runpod/launch_pod.py launch         # waits for RUNNING + SSH, caches runpod/pod.json
python3 runpod/launch_pod.py status         # status + ssh endpoint
python3 runpod/launch_pod.py terminate      # tear it down (STOP THE BILL)

# 2. sync code, install deps, run the dry run (reads runpod/pod.json)
./runpod/sync_and_launch.sh sync            # rsync the code  (top-level excludes are "/"-anchored!)
./runpod/sync_and_launch.sh bootstrap       # build the targeted venv + prewarm w2v-bert
./runpod/sync_and_launch.sh launch          # download sg mp3s -> dry-run finetune (background)
./runpod/sync_and_launch.sh tail            # follow /Sidon/train.log
```

**Drive long steps detached + poll the log.** RunPod's SSH TCP proxy is intermittently congested; start
long work with `setsid bash … </dev/null >log 2>&1 &` and poll the log file. Do **not** hammer SSH with
rapid retry loops — that trips sshd `MaxStartups` and makes it worse; space attempts ≥15–20 s apart.

### Pod layout (everything under `/`)
| path | what |
|---|---|
| `/Sidon` | the repo (rsync'd from local) |
| `/Sidon/.venv` | targeted venv (uv, py3.10, torch 2.8 cu128) |
| `/Sidon/ckpt_v0.1/dryrun_decoder.pt` | the trained decoder saved by the dry run |
| `/Sidon/{bootstrap,train,dryrun}.log` | stdout/stderr of bootstrap / the run |
| `/data/sg` | extracted sg-podcast mp3s (the direct loader reads these) |
| `/hf_cache` | `HF_HOME` (w2v-bert + sidon-v0.1 downloads) |

### Dry-run tunables (env for `runpod/run_sidon_dryrun.sh`)
`MAX_FILES` [60] podcasts sampled · `STEPS` [20] · `BATCH` [2] · `WIN` [10] s window · `NUM_WORKERS` [2].
The sg download/extract step skips if `/data/sg/.done` exists. **Cost:** SECURE H100 ≈ $3.29/hr —
`terminate` when done.

### Bootstrap note
`runpod/bootstrap.sh` builds a **targeted** venv (not `uv sync`): it skips `flash-attn` / `mmdit` /
`flow-matching` / `diffusers` / `ray` (GENESES-only; flash-attn's source build is slow/fragile) and
installs just what the restoration pipeline imports — including `descript-audio-codec` (`dac` + `audiotools`,
which `sidon.model` imports unconditionally) and `pyroomacoustics`.

## Architecture

### Training entry point
`src/sidon/train.py` — Hydra entrypoint. Resolves `cfg.data.datamodule` and `cfg.model.lightning_module`
and calls `trainer.fit()`. Top-level config `config/config.yaml`; override via `model=<name>`/`data=<name>`.

### Model variants and their Lightning modules
| Config key | Lightning module | Description |
|---|---|---|
| `sidon_feature_predictor` | `sidon.model.sidon.lightning_module.FeaturePredictorLightningModule` | LoRA-adapts w2v-BERT student to predict clean SSL features from noisy input (MSE vs frozen teacher) |
| `sidon_vocoder_pretrain` / `sidon_vocoder_finetune` | `sidon.model.sidon.lightning_module.SidonLightningModule` | DAC decoder + GAN discriminator; pretrain on clean SSL, finetune on denoised SSL |
| `geneses_dialogue` | `sidon.model.geneses.lightning_module.GenesesLightningModule` | Flow-matching separator: MMDiT + DACVAE + w2v-BERT conditioning |
| `diffusion_dialogue_sidon*` | `sidon.model.dialogue_sidion.lightning_module.*` | Diffusion-based dialogue separation variants |

### Key components
- `src/sidon/data/preprocess/webdataset_datamodule.py` — `WebDatasetDataModule` (online degradation +
  SSL feature extraction; the dry run trains directly from this).
- `src/sidon/data/datamodule.py` — `PreprocessedDataModule` (reads pre-materialized shards).
- `src/sidon/data/preprocess/functional_degrations.py` — RIR / noise / band-limit / clip / codec /
  packet-loss transforms applied online.
- `src/sidon/model/sidon/lightning_module.py` — `FeaturePredictorLightningModule`, `SidonLightningModule`.
- `src/sidon/model/losses.py` — `DACLoss` (multi-resolution mel/STFT) and `GANLoss` wrappers.
- `export.py` — how the released sidon-v0.1 `.pt`s map back onto the training modules.

### Config structure (`config/`)
```
config/
  config.yaml                       # defaults: data=preprocessed, model=sidon_vocoder_pretrain
  model/                            # per-model configs (cfg.* -> Lightning module constructor)
  data/                             # datamodule configs (incl. sg_podcast_online_48k.yaml for the dry run)
  train/default.yaml                # trainer, WandB logger, callbacks, scheduler
```
All Hydra overrides are CLI args (e.g. `model.cfg.ssl_model_name=...`, `~train.trainer.plugins`).

## RunPod dry-run key files
| file | role |
|---|---|
| `runpod/launch_pod.py` | provision / status / ssh / terminate the pod (REST API, stdlib only) |
| `runpod/bootstrap.sh` | system deps + targeted uv venv + w2v-bert prewarm (on pod) |
| `runpod/run_sidon_dryrun.sh` | **the dry run**: download sg mp3s -> `dryrun_v01_direct.py` (on pod) |
| `runpod/dryrun_v01_direct.py` | **verified path**: direct mp3 loader + frozen v0.1 FE + fresh DAC decoder/GAN |
| `runpod/prepare_sg_data.py` | download/extract sg-podcast (`--download-only`); also packs WebDataset shards |
| `runpod/sync_and_launch.sh` | local: rsync + bootstrap + launch + tail |
| `runpod/inspect_v01.py`, `inspect_v01b.py` | probes proving the released `.pt`s are frozen TorchScript |
| `runpod/build_ckpt_from_hf.py` | attempts `.pt`->Lightning `.ckpt`; refuses (frozen TS) — kept as evidence |
| `runpod/dryrun_v01_decoder.py` | earlier WebDataset-based variant (superseded by the direct loader) |
| `config/data/sg_podcast_online_48k.yaml` | online 48 kHz WebDataset data module (for the full pipeline) |
```
