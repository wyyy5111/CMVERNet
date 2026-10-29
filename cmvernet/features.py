"""Feature assembly for the two original branch inputs.

The arrays are caller-supplied. This module does not create or download datasets,
models or feature caches.
"""
from __future__ import annotations

import numpy as np


def _wavlm_views(raw: np.ndarray, conditioned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float32)
    conditioned = np.asarray(conditioned, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (13, 1536) or raw.shape != conditioned.shape:
        raise ValueError("both WavLM views must have shape (rows, 13, 1536)")
    if not (np.isfinite(raw).all() and np.isfinite(conditioned).all()):
        raise ValueError("WavLM features must be finite")
    return raw.reshape(len(raw), -1), conditioned.reshape(len(raw), -1)


def stable_features(raw: np.ndarray, conditioned: np.ndarray, egemaps: np.ndarray) -> np.ndarray:
    """Assemble raw + conditioned WavLM + eGeMAPS into N x 40,024."""
    raw, conditioned = _wavlm_views(raw, conditioned)
    e = np.asarray(egemaps, dtype=np.float32)
    if e.shape != (len(raw), 88) or not np.isfinite(e).all():
        raise ValueError("eGeMAPS must be a finite (rows, 88) array")
    return np.concatenate((raw, conditioned, e), axis=1)


def compact_features(static: np.ndarray, raw: np.ndarray, conditioned: np.ndarray) -> np.ndarray:
    """Assemble the original static + raw + conditioned input into N x 53,568."""
    raw, conditioned = _wavlm_views(raw, conditioned)
    static = np.asarray(static, dtype=np.float32)
    if static.shape != (len(raw), 13632) or not np.isfinite(static).all():
        raise ValueError("static features must be a finite (rows, 13632) array")
    return np.concatenate((static, raw, conditioned), axis=1)
