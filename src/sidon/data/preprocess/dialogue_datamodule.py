from sidon.data.preprocess.webdataset_datamodule import (
    WebDatasetDataModule,
    rename_audio,
    lowcut,
    normalize,
    skip_nan,
)
import torch
import webdataset as wds
from functools import partial
from ..datamodule import random_crop, torch_audio, PreprocessedDataModule, skip_nan_preprocessed
from .functional_degrations import (
    add_non_parametric_noise,
    band_limit,
    clip,
    codec,
    convolve_rir,
    convolve_rir_pra,
    packet_loss,
    random_apply,
)
from typing import Any
import random
import silero_vad
import torchaudio

def select_by_vad(samples, input_key=None,sr=24000):
    model = silero_vad.load_silero_vad(onnx=True)

    for sample in samples:
        wav = sample[input_key]
        wav = wav[0]
        vad_flag = True
        if sr != 16000:
            wav = torchaudio.functional.resample(wav,sr,16000)
        for i in range(wav.shape[0]):
            speech_timestamps = silero_vad.get_speech_timestamps(
                wav[i],
                model,
                return_seconds=True
            )
            if len(speech_timestamps) == 0:
                vad_flag=False
        if vad_flag:
            yield sample
                



def chunk_crop(samples, input_key=None, chunk_seconds: int = 30):
    """Yield fixed-size chunks from each sample's audio.

    - Always yields dict samples (never raw tensors).
    - Uses an integer hop of chunk_seconds * sr.
    - The final chunk may be shorter than chunk_seconds.
    """
    for sample in samples:
        if input_key is None:
            audio_key = [k for k in sample.keys() if "audio" in k][0]
        else:
            audio_key = input_key
        wav, sr = sample[audio_key]
        n_samples = wav.shape[1]
        step = int(chunk_seconds * sr)
        if step <= 0:
            # Fallback: forward the original sample if step is invalid
            yield sample
            continue
        if n_samples <= step:
            new_sample = sample.copy()
            new_sample[audio_key] = (wav, sr)
            yield new_sample
            continue
        for i in range(0, n_samples, step):
            cropped = wav[:, i : i + step]
            new_sample = sample.copy()
            new_sample[audio_key] = (cropped, sr)
            yield new_sample
def split_by_channel(sample,channel_idx:int,split_sample_key:str):
    wav, sr = sample[split_sample_key]
    new_sample = sample.copy()
    new_sample[split_sample_key] = (wav[channel_idx].unsqueeze(0),sr)
    return new_sample
def drop_key(sample,key_to_drop:str):
    sample.pop(key_to_drop)
    return sample
def pad_to_duration(sample,key,pad_to_duration:int):
    wav, sr = sample[key]
    wav_len = wav.shape[-1]
    if wav_len > pad_to_duration *sr:
        wav = wav[:,:pad_to_duration*sr]
    else:
        wav = torch.nn.functional.pad(wav,(0,int(pad_to_duration*sr - wav_len)))
    sample[key] = (wav,sr)
    return sample

def merge_dataset(
    sample: dict[str, Any],
    merge_keys: list[str],
    other_ds: Any,
    randomize_order: bool = True,
):
    if randomize_order:
        other_last = random.random() > 0.5
    other_sample = next(other_ds)
    if sample['__key__'] !=  other_sample['__key__']:
        raise ValueError
    output_sample = sample.copy()
    for merge_key in merge_keys:
        wav1, sr = sample[merge_key]
        wav2, _ = other_sample[merge_key]
        other_last = True
        if other_last:
            output_sample[merge_key] = (torch.cat([wav1, wav2], dim=0), sr)
        else:
            output_sample[merge_key] = (torch.cat([wav2, wav1], dim=0), sr)
    return output_sample


class DialogueDatasetDataModule(WebDatasetDataModule):
    def setup(self, stage = None):
        self.train_dataset = (
            wds.WebDataset(
                self.train_wds_patterns,
                shardshuffle=True,
                nodesplitter=lambda x: x,
                workersplitter=self.workersplitter,
                repeat=True,
                empty_check=True,
            )
            .repeat(self.n_repeats)
            .decode(wds.autodecode.basichandlers, wds.autodecode.torch_audio)
            .map(partial(rename_audio,input_key='flac', output_key="audio"))
            .map(partial(lowcut, input_key="audio", cutoff=50))
            .map(partial(normalize, input_key="audio", output_key="audio"))
            .compose(partial(chunk_crop,chunk_seconds=self.max_duration))
            .map(partial(pad_to_duration,key="audio", pad_to_duration=self.max_duration))
            .map(partial(rename_audio, input_key="audio", output_key="clean"))
            .map(partial(rename_audio, input_key="audio", output_key="noisy"))
            .map(partial(drop_key,key_to_drop='flac'))
            .map(partial(drop_key,key_to_drop='audio'))
        )
        self.val_dataset = (
            wds.WebDataset(
                self.val_wds_patterns,
                shardshuffle=True,
                nodesplitter=lambda x: x,
                workersplitter=self.workersplitter,
                repeat=True,
                empty_check=True,
            )
            .repeat(self.n_repeats)
            .decode(wds.autodecode.basichandlers, wds.autodecode.torch_audio)
            .map(partial(rename_audio,input_key='flac', output_key="audio"))
            .map(partial(lowcut, input_key="audio", cutoff=50))
            .map(partial(normalize, input_key="audio", output_key="audio"))
            .compose(partial(chunk_crop,chunk_seconds=self.max_duration))
            .map(partial(pad_to_duration,key="audio", pad_to_duration=self.max_duration))
            .map(partial(rename_audio, input_key="audio", output_key="clean"))
            .map(partial(rename_audio, input_key="audio", output_key="noisy"))
            .map(partial(drop_key,key_to_drop='flac'))
            .map(partial(drop_key,key_to_drop='audio'))
        )
        if self.use_noise:
            train_dataset_0 = self.train_dataset.map(
                partial(split_by_channel,channel_idx=0,split_sample_key="noisy")
            ).map(
                partial(split_by_channel,channel_idx=0,split_sample_key="clean")
            )
            train_dataset_1 = self.train_dataset.map(
                partial(split_by_channel,channel_idx=1,split_sample_key='noisy')
            ).map(
                partial(split_by_channel,channel_idx=1,split_sample_key='clean')
            )
            train_dataset_0:wds.compat.WebDataset = self.add_noise_pipeline(train_dataset_0).map(partial(pad_to_duration,key="clean", pad_to_duration=self.max_duration)).map(partial(pad_to_duration,key="noisy", pad_to_duration=self.max_duration))
            train_dataset_1:wds.compat.WebDataset = self.add_noise_pipeline(train_dataset_1).map(partial(pad_to_duration,key="clean", pad_to_duration=self.max_duration)).map(partial(pad_to_duration,key="noisy", pad_to_duration=self.max_duration))
            self.train_dataset = train_dataset_0.map(partial(merge_dataset, merge_keys=['clean', 'noisy'],other_ds=iter(train_dataset_1)),handler=wds.handlers.warn_and_continue)

            val_dataset_0 = self.val_dataset.map(
                partial(split_by_channel,channel_idx=0,split_sample_key="noisy")
            ).map(
                partial(split_by_channel,channel_idx=0,split_sample_key="clean")
            )
            val_dataset_1 = self.val_dataset.map(
                partial(split_by_channel,channel_idx=1,split_sample_key='noisy')
            ).map(
                partial(split_by_channel,channel_idx=1,split_sample_key='clean')
            )
            val_dataset_0:wds.compat.WebDataset = self.add_noise_pipeline(val_dataset_0).map(partial(pad_to_duration,key="clean", pad_to_duration=self.max_duration)).map(partial(pad_to_duration,key="noisy", pad_to_duration=self.max_duration))
            val_dataset_1:wds.compat.WebDataset = self.add_noise_pipeline(val_dataset_1).map(partial(pad_to_duration,key="clean", pad_to_duration=self.max_duration)).map(partial(pad_to_duration,key="noisy", pad_to_duration=self.max_duration))
            self.val_dataset = val_dataset_0.map(partial(merge_dataset, merge_keys=['clean', 'noisy'],other_ds=iter(val_dataset_1)),handler=wds.handlers.warn_and_continue)
        self.train_dataset = self.add_resample_pipeline(self.train_dataset)
        self.train_dataset = (
            self.train_dataset
            .map(partial(normalize, input_key="clean", output_key="clean"))
            .map(partial(normalize, input_key="noisy", output_key="noisy"))
            .map(partial(normalize, input_key="clean_16k", output_key="clean_16k"))
            .map(partial(normalize, input_key="noisy_16k", output_key="noisy_16k"))
        )
        self.train_dataset = self.train_dataset.batched(
            self.batch_size, collation_fn=self.collate_fn
        )
        self.val_dataset = self.add_resample_pipeline(self.val_dataset)
        self.val_dataset = (
            self.val_dataset.map(partial(normalize, input_key="clean", output_key="clean"))
            .map(partial(normalize, input_key="noisy", output_key="noisy"))
            .map(partial(normalize, input_key="clean_16k", output_key="clean_16k"))
            .map(partial(normalize, input_key="noisy_16k", output_key="noisy_16k"))
        )
        self.val_dataset = self.val_dataset.batched(
            1, collation_fn=self.collate_fn
        )
    @torch.inference_mode()
    def collate_fn(
        self,
        samples: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor | int | list[torch.Tensor]]:
        clean_wavs = torch.nn.utils.rnn.pad_sequence(
            [x["clean"][0] for x in samples],
            batch_first=True,
        ).float()
        noisy_wavs = torch.nn.utils.rnn.pad_sequence(
            [x["noisy"][0] for x in samples],
            batch_first=True,
        ).float()
        clean_wav_lens = torch.tensor(
            [x["clean"][0].size(-1) for x in samples],
            dtype=torch.long,
        )
        clean_16k_wavs = torch.nn.utils.rnn.pad_sequence(
            [x["clean_16k"][0] for x in samples],
            batch_first=True,
        ).float()
        noisy_16k_wavs = torch.nn.utils.rnn.pad_sequence(
            [x["noisy_16k"][0] for x in samples],
            batch_first=True,
        ).float()
        ratio = random.uniform(0.3,0.7)

        clean_mixture = clean_wavs[:,0,:] * ratio + (1-ratio) * clean_wavs[:,1,:]
        noisy_mixture = noisy_wavs[:,0,:] * ratio + (1-ratio) * noisy_wavs[:,1,:]

        clean_16k_mixture = clean_16k_wavs[:,0,:] * ratio + (1-ratio) * clean_16k_wavs[:,1,:]
        noisy_16k_mixture = noisy_16k_wavs[:,0,:] * ratio + (1-ratio) * noisy_16k_wavs[:,1,:]

        ssl_inputs_0 = self.processor(
            [
                torch.nn.functional.pad(x["clean_16k"][0][0], (160, 160)).numpy()
                for x in samples
            ],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        ssl_inputs_1 = self.processor(
            [
                torch.nn.functional.pad(x["clean_16k"][0][1], (160, 160)).numpy()
                for x in samples
            ],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        noisy_ssl_inputs_0 = self.processor(
            [
                torch.nn.functional.pad(x["noisy_16k"][0][0], (160, 160)).numpy()
                for x in samples
            ],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        noisy_ssl_inputs_1 = self.processor(
            [
                torch.nn.functional.pad(x["noisy_16k"][0][1], (160, 160)).numpy()
                for x in samples
            ],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        clean_mixture_ssl_inputs = self.processor(
            [
                torch.nn.functional.pad(x.view(-1), (160, 160)).numpy()
                for x in clean_16k_mixture
            ],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        noisy_mixture_ssl_inputs = self.processor(
            [
                torch.nn.functional.pad(x.view(-1), (160, 160)).numpy()
                for x in noisy_16k_mixture
            ],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        # pad to max length
        if self.force_max_length:
            max_length = int(self.max_duration * self.sampling_rate)
            clean_wavs = torch.nn.functional.pad(
                clean_wavs,
                (0, max_length - clean_wavs.size(-1)),
            )


        sample = {
            "input_wav": clean_wavs,
            "input_wav_lens": clean_wav_lens,
            "sr": self.sampling_rate,
            "clean_ssl_inputs_0": ssl_inputs_0,
            "clean_ssl_inputs_1": ssl_inputs_1,
            "clean_mixture_ssl_inputs": clean_mixture_ssl_inputs,
            "noisy_ssl_inputs_0": noisy_ssl_inputs_0,
            "noisy_ssl_inputs_1": noisy_ssl_inputs_1,
            "noisy_mixture_ssl_inputs": noisy_mixture_ssl_inputs,
            "names": [x["__key__"] for x in samples],
            "sample_rate": [x["clean"][1] for x in samples],
            "clean_mixture": clean_mixture,
            "clean_16k_mixture": clean_16k_mixture,
            "noisy_mixture": noisy_mixture,
            "noisy_16k_mixture": noisy_16k_mixture,
        }

        if self.use_noise:
            sample["noisy_input_wav16k"] = noisy_16k_wavs
            sample["noisy_input_wav"] = noisy_wavs
            if self.force_max_length:
                max_length = int(self.max_duration * self.sampling_rate)
                noisy_wavs = torch.nn.functional.pad(
                    noisy_wavs,
                    (0, max_length - noisy_wavs.size(-1)),
                )

        return sample

class PreprocessedDialogueDataModule(PreprocessedDataModule):
    """Loads preprocessed dialogue tensors packaged as WebDataset shards.

    Expects records materialised from DialogueDatasetDataModule via
    src/sidon/preprocess.py. This collate stacks channel-preserving waveforms
    and reassembles multiple SSL feature groups (per-speaker and mixtures).
    """
    @property
    def get_shuffle_buffer_size(self):
        return 100

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = (
            wds.WebDataset(
                self.train_urls,
                shardshuffle=True,
                nodesplitter=lambda x: x,
                workersplitter=wds.split_by_worker,
                repeat=True,
                empty_check=True,
                handler=wds.warn_and_continue,
            )
            .decode(
                wds.autodecode.basichandlers,
                torch_audio,
                handler=wds.warn_and_continue,
            )
            .compose(skip_nan_preprocessed)
            .compose(partial(select_by_vad,input_key='input_wav.pth'))
            .shuffle(self.get_shuffle_buffer_size)
            .batched(self.batch_size, collation_fn=self.collate_fn)
        )
        self.val_dataset = (
            wds.WebDataset(
                self.val_urls,
                shardshuffle=True,
                nodesplitter=lambda x: x,
                workersplitter=wds.split_by_worker,
                repeat=True,
                empty_check=True,
                handler=wds.warn_and_continue,
            )
            .decode(
                wds.autodecode.basichandlers,
                torch_audio,
                handler=wds.warn_and_continue,
            )
            .compose(skip_nan_preprocessed)
            .compose(partial(select_by_vad,input_key='input_wav.pth'))
            .batched(self.val_batch_size, collation_fn=self.collate_fn)
        )
    def collate_fn(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble pickled dialogue features and wave tensors into a batch.

        Reconstructs a batch equivalent to DialogueDatasetDataModule's output from
        preprocessed WebDataset shards: preserves stereo-like speaker channels,
        concatenates SSL BatchEncodings across items, and pads variable lengths.
        """
        import torch.nn.functional as F

        def _as_batched(t: torch.Tensor) -> torch.Tensor:
            # Expect [B0, C, T] or [B0, T] or [C, T]; normalize to [B0, C, T]
            if t.dim() == 1:  # [T]
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.dim() == 2:  # [C, T] or [B0, T]
                # Heuristic: treat leading dim as channels
                t = t.unsqueeze(0)
            # else keep as-is [B0, C, T]
            return t

        def _pad_cat(tensors: list[torch.Tensor]) -> torch.Tensor:
            if len(tensors) == 0:
                return torch.empty(0)
            tensors = [_as_batched(t) for t in tensors]
            max_len = max(t.size(-1) for t in tensors)
            padded = [F.pad(t, (0, max_len - t.size(-1))) for t in tensors]
            return torch.cat(padded, dim=0)

        def _cat_batchenc(encs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            if len(encs) == 0:
                return {}
            keys = encs[0].keys()
            out: dict[str, torch.Tensor] = {}
            for k in keys:
                vals = [e[k] for e in encs if k in e]
                if len(vals) == 0:
                    continue
                # Expect tensors with batch in dim=0
                out[k] = torch.cat(vals, dim=0)
            return out

        # Audio tensors
        input_wavs = [s["input_wav.pth"] for s in samples]
        noisy_input_wavs = [s.get("noisy_input_wav.pth") for s in samples if "noisy_input_wav.pth" in s]
        noisy_input_wavs_16k = [s.get("noisy_input_wav16k.pth") for s in samples if "noisy_input_wav16k.pth" in s]
        clean_mix = [s.get("clean_mixture.pth") for s in samples if "clean_mixture.pth" in s]
        clean_mix_16k = [s.get("clean_16k_mixture.pth") for s in samples if "clean_16k_mixture.pth" in s]
        noisy_mix = [s.get("noisy_mixture.pth") for s in samples if "noisy_mixture.pth" in s]
        noisy_mix_16k = [s.get("noisy_16k_mixture.pth") for s in samples if "noisy_16k_mixture.pth" in s]

        input_wav = _pad_cat(input_wavs)  # [B, C, T]
        if noisy_input_wavs:
            noisy_input_wav = _pad_cat(noisy_input_wavs)
        if noisy_input_wavs_16k:
            noisy_input_wav16k = _pad_cat(noisy_input_wavs_16k)
        if clean_mix:
            clean_mixture = _pad_cat(clean_mix).squeeze(1)  # [B, T]
        if clean_mix_16k:
            clean_16k_mixture = _pad_cat(clean_mix_16k).squeeze(1)
        if noisy_mix:
            noisy_mixture = _pad_cat(noisy_mix).squeeze(1)
        if noisy_mix_16k:
            noisy_16k_mixture = _pad_cat(noisy_mix_16k).squeeze(1)

        # Lengths (per primary sampling rate)
        input_wav_lens = torch.tensor(
            [int(t.size(-1)) for t in [_as_batched(s["input_wav.pth"]) for s in samples]],
            dtype=torch.long,
        )


        # Names and sampling rate
        names: list[str] = []
        for s in samples:
            if "names.pickle" in s:
                n = s["names.pickle"]
                if isinstance(n, list):
                    names.extend([str(x) for x in n])
                else:
                    names.append(str(n))
            elif "__key__" in s:
                names.append(str(s["__key__"]))

        batch: dict[str, Any] = {
            "input_wav": input_wav.float(),
            "input_wav_lens": input_wav_lens,
            "sr": samples[0].get("sr.index", 48000),
            "names": names,
        }

        if noisy_input_wavs:
            batch["noisy_input_wav"] = noisy_input_wav.float()
        if noisy_input_wavs_16k:
            batch["noisy_input_wav16k"] = noisy_input_wav16k.float()
        if clean_mix:
            batch["clean_mixture"] = clean_mixture.float()
        if clean_mix_16k:
            batch["clean_16k_mixture"] = clean_16k_mixture.float()
        if noisy_mix:
            batch["noisy_mixture"] = noisy_mixture.float()
        if noisy_mix_16k:
            batch["noisy_16k_mixture"] = noisy_16k_mixture.float()

        return batch
