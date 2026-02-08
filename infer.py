import torchaudio

wav,sr = torchaudio.load('comedy2.mp4')

wav = wav.mean(dim=0,keepdim=True)

from sidon.model.dialogue_sidion.lightning_module import DialogueSidonDiffusionLightningModule

# %%
path = "/home/qch10240fz/nakata/github.com/Sidon/sidon/6oqvn3is/checkpoints/epoch=1-step=312171.ckpt"
model = DialogueSidonDiffusionLightningModule.load_from_checkpoint(path)
model = model.to('cpu')

output = model.predict_separated(wav[:,:sr*30].to(model.device),sample_rate=sr)

# %%
output

# %%
from IPython.display import Audio

# %%
# %%
torchaudio.save("comedy2_r.wav",output[0].cpu(),sample_rate=24_000)


