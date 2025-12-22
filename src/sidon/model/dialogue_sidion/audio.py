"""Feature extraction helpers reused by the dataset cleansing scripts."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio

PaddingStrategy = Union[bool, str]
ReturnType = Union[torch.Tensor, np.ndarray]


def _pad_batch(
    features: List[torch.Tensor],
    padding_strategy: PaddingStrategy = "longest",
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of feature tensors and return an attention mask."""
    if padding_strategy == "longest":
        target_length = max(f.shape[0] for f in features)
    elif max_length is not None:
        target_length = max_length
    else:
        raise ValueError(
            "max_length must be provided when padding_strategy is not 'longest'"
        )

    if pad_to_multiple_of is not None:
        target_length = (
            (target_length + pad_to_multiple_of - 1)
            // pad_to_multiple_of
            * pad_to_multiple_of
        )

    batch_size = len(features)
    feature_dim = features[0].shape[1]
    device = features[0].device

    padded_features = torch.full(
        (batch_size, target_length, feature_dim),
        padding_value,
        dtype=torch.float32,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, target_length),
        dtype=torch.int64,
        device=device,
    )

    for index, feature_tensor in enumerate(features):
        seq_len = feature_tensor.shape[0]
        padded_features[index, :seq_len] = feature_tensor
        attention_mask[index, :seq_len] = 1

    return padded_features, attention_mask


def extract_seamless_m4t_features(
    raw_speech: Union[torch.Tensor, List[float], List[torch.Tensor], List[List[float]]],
    sampling_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length: int = 25,
    frame_shift: int = 10,
    preemphasis_coefficient: float = 0.97,
    dither: float = 0.0,
    window_type: str = "povey",
    do_normalize_per_mel_bins: bool = True,
    stride: int = 2,
    padding: PaddingStrategy = "longest",
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = 2,
    return_tensors: Optional[str] = "pt",
    return_attention_mask: bool = True,
    padding_value: float = 0.0,
    device: str = "cuda",
) -> Dict[str, ReturnType]:
    """Extract SeamlessM4T features using Torch-only operators."""
    if device == "cuda" and not torch.cuda.is_available():
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device)

    if not isinstance(raw_speech, list):
        raw_speech = [raw_speech]

    processed_speech = [
        torch.as_tensor(sample, dtype=torch.float32, device=torch_device)
        for sample in raw_speech
    ]

    features: List[torch.Tensor] = []
    for waveform in processed_speech:
        if waveform.ndim > 1:
            waveform = waveform[0]
        waveform_tensor = waveform.unsqueeze(0)
        feature = torchaudio.compliance.kaldi.fbank(
            waveform=waveform_tensor,
            sample_frequency=sampling_rate,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            preemphasis_coefficient=preemphasis_coefficient,
            remove_dc_offset=True,
            window_type=window_type,
            use_energy=False,
            energy_floor=1.192092955078125e-07,
        )
        features.append(feature.squeeze(0))

    if do_normalize_per_mel_bins:
        normalised: List[torch.Tensor] = []
        for feature in features:
            mean = feature.mean(0, keepdim=True)
            var = feature.var(0, keepdim=True)
            normalised.append((feature - mean) / torch.sqrt(var + 1e-5))
        features = normalised

    input_features, attention_mask = _pad_batch(
        features,
        padding_strategy=padding,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        padding_value=padding_value,
    )

    batch_size, num_frames, num_channels = input_features.shape
    new_num_frames = (num_frames // stride) * stride
    input_features = input_features[:, :new_num_frames, :]
    if return_attention_mask:
        attention_mask = attention_mask[:, :new_num_frames]

    input_features = input_features.reshape(
        batch_size, new_num_frames // stride, num_channels * stride
    )

    output: Dict[str, ReturnType] = {"input_features": input_features}
    if return_attention_mask:
        output["attention_mask"] = attention_mask[:, 1::stride]

    if return_tensors == "np":
        for key, value in output.items():
            output[key] = value.cpu().numpy()  # type: ignore[assignment]

    return output