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
- Decoder: https://wandb.ai/aies-scicom-scicom-ai/sidon/runs/decoder-callcentre-3072

## Checkpoints

The trained artifact for the FE is the **LoRA adapter only** (~16M, ~62 MB); the 580M base is stock
`facebook/w2v-bert-2.0`. The decoder checkpoint is the full 188M decoder state.

### Load the finetuned feature extractor
```python
import torch
from transformers import Wav2Vec2BertModel, AutoFeatureExtractor
from peft import LoraConfig, inject_adapter_in_model

fe = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", num_hidden_layers=24, layerdrop=0.0)
fe = inject_adapter_in_model(
    LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="lora_only", target_modules=["output_dense"]), fe)

ck = torch.load("checkpoints/fe_callcentre_lora.pt", map_location="cpu")   # {'step', 'lora': {...96 tensors...}}
missing, unexpected = fe.load_state_dict(ck["lora"], strict=False)         # only adapter keys are loaded
assert not unexpected, unexpected
fe.eval()

proc = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
# feats = fe(**proc(wav_16k, sampling_rate=16000, return_tensors="pt")).last_hidden_state  # [B,T,1024]
```
Verified loading locally: step 20000, 96 LoRA tensors (48 `lora_A` (64,4096) + 48 `lora_B` (1024,64)).

### Load the decoder
```python
import dac, torch
decoder = dac.model.dac.Decoder(input_channel=1024, channels=3072, rates=[8,5,4,3,2])
decoder.load_state_dict(torch.load("checkpoints/decoder_callcentre_last.pt", map_location="cpu")["decoder"])
decoder.eval()
# wav48k = decoder(feats.transpose(1,2))   # [B,T,1024] -> [B,1,T*960]
```

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
| `sync_and_launch.sh` | local: rsync repo to the pod |
