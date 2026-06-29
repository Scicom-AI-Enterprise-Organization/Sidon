# Sidon call-centre finetune (RunPod)

Adapt Sidon to **call-centre / telephony** audio. The released `sarulab-speech/sidon-v0.1`
feature extractor is frozen and underperforms on telephony, so we **finetune the feature
extractor** (and train a matching decoder) for the telephony domain. All on **1× H100,
US region, SECURE, container disk `/` only (never `/workspace`)**.

## Pipeline (two stages)

```
Stage 1 — FE finetune (distillation):
  clean 48k (EARS/Expresso) --16k--> --telephony-degrade--> --(student 24L w2v-BERT + LoRA)--> H_student
  clean 48k                 --16k-->                          --(teacher 24L w2v-BERT, frozen)--> H_clean
  loss = MSE(H_student, H_clean)        # only the ~16M LoRA adapter trains; base + teacher frozen

Stage 2/3 — decoder finetune (GAN):
  telephony-degraded clean --(FROZEN stage-1 FE)--> features[T,1024]
                           --(TRAINABLE DAC decoder, 188M)--> 48 kHz waveform
  loss = 15·mel(multi-res) + 2·adv_gen + 1·adv_feat   vs clean 48k
```

- **FE = full 24-layer `facebook/w2v-bert-2.0` + fresh LoRA** (r=64, α=16, dropout=0.1,
  bias=`lora_only`, target=`output_dense`). NOT the 8-layer Sidon default, and NOT warm-started
  from v0.1 (its weights are frozen TorchScript — not loadable; and 24L ≠ v0.1's 8L anyway).
- **Telephony degradation** (simulated from clean): 8 kHz band-limit + G.711 µ-law + mp3 codec
  + packet loss + additive noise.
- **Decoder** = `dac.model.dac.Decoder(input_channel=1024, channels=3072, rates=[8,5,4,3,2])`
  (188M, 50 fps × 960 = 48 kHz), trained with the DAC multi-resolution mel + GAN discriminators.

## Run (on the pod)

```bash
# provision: 1x H100 US SECURE, 500 GB at /  (launch_pod.py pins US datacenters, volumeInGb=0)
python3 runpod/launch_pod.py launch --name sidon-fe-callcentre --disk-gb 500
./runpod/sync_and_launch.sh sync
ssh <pod> 'bash /Sidon/runpod/bootstrap_fe.sh'                # targeted venv (see deps note)

# stage 1 — FE finetune (downloads EARS+Expresso, then trains). ~0.32 s/step, 20k steps.
ssh <pod> 'cd /Sidon && setsid env STEPS=20000 bash runpod/run_fe_callcentre.sh >fe_train.log 2>&1 &'
#   -> /Sidon/fe_callcentre/last.pt   (mse 0.0092 -> ~0.0030)

# stage 2/3 — decoder finetune (frozen FE + DAC decoder + GAN). bf16, eff batch 24, win 6 s.
ssh <pod> 'setsid bash /Sidon/runpod/go_decoder_b6w6.sh >/Sidon/decoder_train.log 2>&1 &'
#   batch=4 accum=6 (eff 24), win=6 s, channels=3072 -> ~63 GB on the H100 (batch6/win6 OOMs ~79 GB)
#   -> /Sidon/decoder_callcentre/last.pt
```

Tip: drive long steps **detached** (`setsid … </dev/null >log 2>&1 &`) and poll the log — the
RunPod SSH proxy is intermittently congested. Keep launch commands tiny (see `go_decoder_b6w6.sh`).

## wandb
- FE (stage 1): https://wandb.ai/aies-scicom-scicom-ai/sidon/runs/fe-callcentre-d24-run1
- Decoder: https://wandb.ai/aies-scicom-scicom-ai/sidon/runs/decoder-callcentre-3072v2

## Checkpoints

Two slim artifacts (drop the optimizer/discriminator/base state):

| file | what | size |
|---|---|---|
| `checkpoints/fe_adapter_full.pt` | FE adapter: **144 tensors** — 96 LoRA (`lora_A`,`lora_B`) + 48 trained `output_dense` biases (`bias=lora_only`). 580M base is stock `facebook/w2v-bert-2.0`. | ~63 MB |
| `checkpoints/decoder_only.pt` | the 188M DAC decoder state (`{'step','dec_channels','decoder'}`) | ~750 MB |

> **Why "full" matters:** `bias=lora_only` also trains the adapted layers' biases (measured
> max\|Δ\|≈0.08 vs base — not negligible), so a LoRA-only (96-tensor) export is incomplete.
> Always use the 144-tensor `fe_adapter_full.pt`.

Pull them off the pod (slim them first to skip optimizer state — see `pod_prep_infer.py`):
```bash
python runpod/pod_prep_infer.py        # on the pod: writes fe_adapter_full.pt + decoder_only.pt
scp <pod>:/Sidon/fe_callcentre/fe_adapter_full.pt      checkpoints/
scp <pod>:/Sidon/decoder_callcentre/decoder_only.pt    checkpoints/
```

### Load the feature extractor — option A: merge the adapter (no `peft` needed)
The effective adapted layer is `W_eff = W_base + (α/r)·BᴬBᴮ`, bias = trained bias. Merging once
means inference needs only `transformers` + `descript-audio-codec` (this is what
`infer_callcentre.py` does):
```python
import torch
from transformers import Wav2Vec2BertModel
ck = torch.load("checkpoints/fe_adapter_full.pt", map_location="cpu"); ad = ck["adapter"]
scaling = ck["lora_alpha"] / ck["r"]                                   # 16/64 = 0.25
fe = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", num_hidden_layers=24, layerdrop=0.0)
sd = fe.state_dict()
for p in sorted({k[:-len(".lora_A.default.weight")] for k in ad if k.endswith(".lora_A.default.weight")}):
    sd[p + ".weight"] = sd[p + ".weight"].float() + scaling * (ad[p+".lora_B.default.weight"].float()
                                                               @ ad[p+".lora_A.default.weight"].float())
    if p + ".base_layer.bias" in ad:                                  # trained lora_only bias
        sd[p + ".bias"] = ad[p + ".base_layer.bias"]
fe.load_state_dict(sd); fe.eval()
```

### Load the feature extractor — option B: `peft` (matches training exactly)
```python
from peft import LoraConfig, inject_adapter_in_model
fe = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", num_hidden_layers=24, layerdrop=0.0)
fe = inject_adapter_in_model(
    LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="lora_only", target_modules=["output_dense"]), fe)
res = fe.load_state_dict(torch.load("checkpoints/fe_adapter_full.pt")["adapter"], strict=False)
assert not res.unexpected_keys      # missing = the frozen base (loaded by from_pretrained)
fe.eval()
```

### Load the decoder
```python
import dac, torch
ck = torch.load("checkpoints/decoder_only.pt", map_location="cpu")
decoder = dac.model.dac.Decoder(input_channel=1024, channels=ck["dec_channels"], rates=[8,5,4,3,2])
decoder.load_state_dict(ck["decoder"]); decoder.eval()
# wav48k = decoder(feats.transpose(1,2))   # [B,T,1024] -> [B,1,T*960]  (50 fps x 960 = 48 kHz)
```

## Inference (restoration)

`runpod/infer_callcentre.py` restores telephony/call-centre audio to clean 48 kHz end-to-end:

```
input audio --16k--> (FE: 24L w2v-BERT + merged LoRA) --features[T,1024]--> (DAC decoder 188M) --> 48 kHz
```

```bash
python runpod/infer_callcentre.py \
    --input audio --out-dir audio/out \
    --fe-adapter checkpoints/fe_adapter_full.pt \
    --decoder    checkpoints/decoder_only.pt
# per file -> audio/out/<name>_restored48k.wav  +  <name>_orig48k.wav (naive-upsampled, for A/B)
```

- **Input**: any sample rate / format `ffmpeg`/libsndfile reads (the demo `audio/` are 8 kHz stereo
  call recordings). Input is peak-normalized and resampled to 16 kHz for the FE.
- **Stereo** (e.g. agent/customer on separate channels) is restored **per channel** and recombined to
  stereo; pass `--mono` to downmix first.
- Long clips are windowed (`--chunk`, default 35 s) with a click-free output-domain crossfade.
- **bf16** autocast on by default; RTF ≈ 0.03 on an RTX 3090 Ti. CPU works (no GPU) but is slow.
- Needs `transformers`, `descript-audio-codec` (`dac`), `torchaudio`, `soundfile` — **no `peft`**
  (the adapter is merged at load). Note: `peft` against bleeding-edge `transformers` (≥5.5) can break
  imports; the merge path side-steps that.

## Deps note
`bootstrap_fe.sh` builds a targeted venv (torch cu128, transformers, peft, datasets, soundfile, wandb).
The decoder stage adds `descript-audio-codec` + `lightning` + `hydra-core`. **Version pins that work
together** (audiotools soft-pins old protobuf; wandb/numpy clash otherwise):
`numpy==1.26.4`, `protobuf==4.25.9`, `wandb==0.16.6`.

## Files
| file | role |
|---|---|
| `launch_pod.py` | provision H100 (US, SECURE, `/`-only) |
| `bootstrap_fe.sh` | targeted venv for the call-centre finetune |
| `prepare_clean48k.py` | download EARS + Expresso clean 48k → `/data/clean48k` |
| `train_fe_callcentre.py` | stage-1 FE distillation (24L w2v-BERT + LoRA, telephony degrade) |
| `run_fe_callcentre.sh` | on-pod: prepare data → FE finetune |
| `train_decoder_callcentre.py` | stage-2/3 DAC decoder + GAN (frozen FE), bf16, grad-accum |
| `run_decoder_callcentre.sh` / `go_decoder_b6w6.sh` | on-pod: decoder finetune launchers |
| `pod_prep_infer.py` | on-pod: slim `last.pt`s → `fe_adapter_full.pt` (144 tensors) + `decoder_only.pt` |
| `infer_callcentre.py` | **local inference**: restore telephony audio → 48 kHz (merges LoRA, no `peft`) |
| `sync_and_launch.sh` | local: rsync repo to the pod |
