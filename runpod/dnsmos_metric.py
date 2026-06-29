"""DNSMOS P.835 (Microsoft) cleanliness metric via the `speechmos` package.

Returns MOS-style scores on the SPEECH itself (1-5), robust to silence padding:
  bak  = background-noise quality (HIGHER = less background noise / cleaner)
  sig  = speech-signal quality (distortion)
  ovrl = overall
  p808 = P.808 overall MOS
We filter on `bak` (and report ovrl). Input is resampled to 16 kHz and peak-
normalized to [-1, 1] (speechmos requires that range)."""
import numpy as np

try:
    import soxr
    _HAVE_SOXR = True
except Exception:  # noqa: BLE001
    _HAVE_SOXR = False

# Force onnxruntime sessions to a SINGLE thread BEFORE speechmos creates them.
# Otherwise each worker's ONNX session sizes its thread pool to the *host* core
# count (the pod sees ~192 cores) -> pthread_setaffinity errors + brutal
# oversubscription across our process pool. We parallelize across datasets, so
# 1 intra/inter-op thread per session is exactly right.
import onnxruntime as _ort

_ORIG_SESS = _ort.InferenceSession


def _single_thread_session(*args, **kwargs):
    so = _ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    kwargs["sess_options"] = so
    kwargs.setdefault("providers", ["CPUExecutionProvider"])
    return _ORIG_SESS(*args, **kwargs)


_ort.InferenceSession = _single_thread_session

from speechmos import dnsmos


def clip_dnsmos(x, sr, max_s: float = 15.0):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        if _HAVE_SOXR:
            x = soxr.resample(x, sr, 16000).astype(np.float32)
        else:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000
    x = x[: int(max_s * sr)]
    if len(x) < sr:                     # need >= 1 s
        return None
    m = float(np.max(np.abs(x)))
    if m < 1e-6:
        return None
    x = (x / m * 0.99).astype(np.float32)
    try:
        r = dnsmos.run(x, sr)
    except Exception:  # noqa: BLE001
        return None
    return {"bak": float(r["bak_mos"]), "sig": float(r["sig_mos"]),
            "ovrl": float(r["ovrl_mos"]), "p808": float(r["p808_mos"])}
