"""Fold-wise AURORA aggregation used by the historical evaluation runner.

Unlike the paper's training-metadata description, that runner computed temporal
coverage from each evaluation fold's metadata. This function intentionally
preserves that behavior for auditability; it never accesses evaluation labels.
It expects caller-provided aligned, out-of-fold expert outputs.
"""
from __future__ import annotations

import numpy as np
from .aurora_router import AuroraConfig, fit_context_gate, predict_proba

# Values retained from the original run_all_fast_fused.py, not a claim of
# independent train-only calibration for every outer fold.
PER_DATASET_MARGIN = {
    "EmoDB": 0.45,
    "SAVEE": 0.0,
    "CASIA": 0.15,
    "IEMOCAP": 0.0,
    "BodEMODB": 0.35,
    "CREMA": 0.40,
    "Henan": 0.45,
}


def route_oof(stable_probability, compact_prediction, context_probability,
              folds, temporal_positions, margin):
    """Route aligned OOF predictions with the original per-test-fold gate."""
    stable = np.asarray(stable_probability, dtype=np.float64)
    compact = np.asarray(compact_prediction, dtype=np.int64)
    context = np.asarray(context_probability, dtype=np.float64)
    folds = np.asarray(folds)
    positions = np.asarray(temporal_positions, dtype=np.float64)
    if stable.ndim != 2 or context.shape != stable.shape:
        raise ValueError("stable and context probabilities must have identical 2-D shapes")
    if compact.shape != (len(stable),) or folds.shape != (len(stable),) or positions.shape != (len(stable),):
        raise ValueError("predictions, folds and temporal positions must align")
    if not np.isfinite(margin) or margin < 0 or margin >= 1:
        raise ValueError("margin must be in [0,1)")
    if not np.isin(folds, (1, 2, 3, 4, 5)).all():
        raise ValueError("fold identifiers must be in 1..5")
    output = np.full_like(stable, np.nan)
    for fold in range(1, 6):
        test = folds == fold
        if not np.any(test):
            continue
        if margin <= 0:
            gate = fit_context_gate(positions[test], AuroraConfig(confidence_margin_boundary=0.45))
            output[test] = context[test] if bool(gate["context_active"]) else stable[test]
        else:
            cfg = AuroraConfig(confidence_margin_boundary=float(margin))
            gate = fit_context_gate(positions[test], cfg)
            output[test], _ = predict_proba(stable[test], compact[test], context[test], gate, cfg)
    return output
