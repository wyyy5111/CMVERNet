"""Auditable low-dimensional acoustic views for the Henan recordings.

Every view is a deterministic transformation of one real WAV file.  The
module never writes modified audio, augments samples, or uses labels.  It is
intended for controlled comparisons of recording conditioning and eGeMAPS
functionals under fold-local model fitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt
import soundfile as sf


@dataclass(frozen=True)
class HenanAcousticViewConfig:
    sample_rate: int = 16000
    highpass_hz: float = 55.0
    trim_top_db: float = 35.0
    denoise_prop_decrease: float = 0.75
    denoise_time_constant_s: float = 0.5
    denoise_n_fft: int = 512
    denoise_hop_length: int = 128
    feature_set: str = "eGeMAPSv02"
    feature_level: str = "Functionals"
    implementation_version: str = "henan-real-acoustic-views-v1"


DEFAULT_CONFIG = HenanAcousticViewConfig()
VIEW_NAMES = ("egemaps_raw", "egemaps_conditioned", "egemaps_denoised")
FEATURE_DIMENSION = 88
_OPENSMILE_RETRY_LOCK = threading.Lock()
_OPENSMILE_MAX_ATTEMPTS = 4


def configuration_metadata(
    config: HenanAcousticViewConfig = DEFAULT_CONFIG,
) -> Mapping[str, object]:
    return asdict(config)


def _load_real_wav(path: Path, sample_rate: int) -> np.ndarray:
    waveform, source_rate = sf.read(
        path, dtype="float32", always_2d=False
    )
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise RuntimeError(f"empty or non-finite waveform: {path}")
    if int(source_rate) != int(sample_rate):
        waveform = librosa.resample(
            waveform,
            orig_sr=int(source_rate),
            target_sr=int(sample_rate),
        ).astype(np.float32)
    if float(np.max(np.abs(waveform))) <= 1e-8:
        raise RuntimeError(f"silent waveform: {path}")
    return waveform


def _condition(
    waveform: np.ndarray,
    config: HenanAcousticViewConfig,
) -> np.ndarray:
    output = np.asarray(waveform, dtype=np.float32)
    output = output - np.median(output)
    sos = butter(
        4,
        config.highpass_hz,
        btype="highpass",
        fs=config.sample_rate,
        output="sos",
    )
    if len(output) > 64:
        output = sosfiltfilt(sos, output).astype(np.float32)
    trimmed, _ = librosa.effects.trim(
        output,
        top_db=config.trim_top_db,
        frame_length=512,
        hop_length=128,
    )
    if len(trimmed) >= int(0.25 * config.sample_rate):
        output = np.asarray(trimmed, dtype=np.float32)
    return output


def _denoise(
    waveform: np.ndarray,
    config: HenanAcousticViewConfig,
) -> np.ndarray:
    # Denoising is needed only when an acoustic cache row must be rebuilt.
    # Keep cache-only V15 evaluation importable under the project's canonical
    # system Python, and fail explicitly at the real extraction boundary when
    # the optional runtime dependency is unavailable.
    try:
        import noisereduce as nr
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "real-WAV denoising requires noisereduce>=3.0.3; install the "
            "declared requirements before rebuilding acoustic caches"
        ) from error
    output = nr.reduce_noise(
        y=np.asarray(waveform, dtype=np.float32),
        sr=config.sample_rate,
        stationary=False,
        prop_decrease=config.denoise_prop_decrease,
        time_constant_s=config.denoise_time_constant_s,
        n_fft=config.denoise_n_fft,
        hop_length=config.denoise_hop_length,
        use_tqdm=False,
    )
    output = np.asarray(output, dtype=np.float32)
    if output.shape != waveform.shape or not np.all(np.isfinite(output)):
        raise RuntimeError("non-stationary spectral denoising failed")
    return _condition(output, config)


def _smile() -> Any:
    try:
        import opensmile
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "real-WAV eGeMAPS extraction requires opensmile; install the "
            "declared requirements before rebuilding acoustic caches"
        ) from error
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def _functionals(
    smile: Any,
    waveform: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    frame = smile.process_signal(waveform, sample_rate)
    vector = np.asarray(frame.iloc[0].to_numpy(), dtype=np.float32)
    if vector.shape != (FEATURE_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"unexpected eGeMAPS feature vector: {vector.shape}")
    return vector


def _functionals_with_retry(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    max_attempts: int = _OPENSMILE_MAX_ATTEMPTS,
) -> np.ndarray:
    """Extract eGeMAPS while containing transient openSMILE init failures.

    The Python openSMILE wrapper initializes a native component during every
    ``process_signal`` call.  Concurrent native initializations can fail with
    ``OpenSmileException(Code: 1)`` even for a valid waveform.  Every native
    openSMILE call is therefore protected by one process-wide lock.  WAV
    loading, resampling, conditioning, task scheduling, and cache I/O remain
    parallel.  A failed native call is retried with a fresh wrapper instance.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    try:
        from opensmile.core.lib import OpenSmileException
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "real-WAV eGeMAPS extraction requires opensmile; install the "
            "declared requirements before rebuilding acoustic caches"
        ) from error

    last_error: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            with _OPENSMILE_RETRY_LOCK:
                if attempt:
                    time.sleep(0.02 * attempt)
                return _functionals(_smile(), waveform, sample_rate)
        except OpenSmileException as error:
            last_error = error
    raise RuntimeError(
        f"openSMILE failed after {max_attempts} attempts on a valid waveform"
    ) from last_error


def extract_acoustic_views(
    path: Path,
    config: HenanAcousticViewConfig = DEFAULT_CONFIG,
) -> Mapping[str, np.ndarray]:
    """Extract raw, conditioned, and denoised eGeMAPS views from one WAV."""

    waveform_views = load_waveform_views(path, config)
    waveform = waveform_views["raw"]
    conditioned = waveform_views["conditioned"]
    denoised = _denoise(conditioned, config)
    return {
        "egemaps_raw": _functionals_with_retry(waveform, config.sample_rate),
        "egemaps_conditioned": _functionals_with_retry(
            conditioned, config.sample_rate
        ),
        "egemaps_denoised": _functionals_with_retry(
            denoised, config.sample_rate
        ),
    }


def extract_acoustic_view(
    path: Path,
    view_name: str,
    config: HenanAcousticViewConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Extract exactly one requested eGeMAPS view from a real WAV.

    The all-view extractor remains the default research interface.  This
    narrow entry point exists for frozen models such as V15 that consume only
    ``egemaps_raw``; it avoids computing and caching unused transformations.
    """

    if view_name not in VIEW_NAMES:
        raise ValueError(f"unknown acoustic view: {view_name}")
    waveform_views = load_waveform_views(path, config)
    waveform = waveform_views["raw"]
    if view_name == "egemaps_raw":
        selected = waveform
    elif view_name == "egemaps_conditioned":
        selected = waveform_views["conditioned"]
    else:
        selected = _denoise(waveform_views["conditioned"], config)
    return _functionals_with_retry(selected, config.sample_rate)


def load_waveform_views(
    path: Path,
    config: HenanAcousticViewConfig = DEFAULT_CONFIG,
) -> Mapping[str, np.ndarray]:
    """Load one real WAV and return raw plus fixed conditioned waveforms."""

    waveform = _load_real_wav(Path(path), config.sample_rate)
    return {
        "raw": waveform,
        "conditioned": _condition(waveform, config),
    }


__all__ = [
    "DEFAULT_CONFIG",
    "FEATURE_DIMENSION",
    "HenanAcousticViewConfig",
    "VIEW_NAMES",
    "configuration_metadata",
    "extract_acoustic_view",
    "extract_acoustic_views",
    "load_waveform_views",
]
