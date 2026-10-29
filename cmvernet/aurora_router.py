"""AURORA: Adaptive Uncertainty-Routed Representation Aggregation.

The model has three fixed experts and one corpus-identity-free router:

1. a uniformly configured V45 probabilistic utterance expert;
2. a compact hard-decision expert;
3. a dialogue-context STEM expert.

When no real dialogue coordinate is available, the compact expert may correct
the stable expert only on a disagreement whose stable top-two margin is below
the frozen boundary.  Correction swaps (rather than invents) posterior mass
between the stable top class and compact selected class.  Real dialogue
coordinates activate the context expert sparsely.

The 0.65 boundary was discovered on earlier OOF development evidence.  This
module therefore cannot by itself turn the resulting scores into independent
confirmation; the evidence status is deliberately part of the configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


MODEL_ID = "Z9-AURORA-v1"


@dataclass(frozen=True)
class AuroraConfig:
    confidence_margin_boundary: float = 0.65
    temporal_coverage_threshold: float = 0.50
    context_gate_mode: str = "hard_sparse"
    posterior_transport: str = "swap_stable_top_with_compact_class"
    evidence_status: str = "development_only_posthoc_boundary"
    implementation_version: str = "z9-aurora-1.0"

    def validate(self) -> None:
        if not 0.0 < self.confidence_margin_boundary < 1.0:
            raise ValueError("confidence margin boundary must lie in (0, 1)")
        if not 0.0 < self.temporal_coverage_threshold <= 1.0:
            raise ValueError("temporal coverage threshold must lie in (0, 1]")
        if self.context_gate_mode != "hard_sparse":
            raise ValueError("Z9-AURORA requires sparse context routing")
        if self.posterior_transport != "swap_stable_top_with_compact_class":
            raise ValueError("Z9-AURORA posterior transport changed")
        if self.evidence_status != "development_only_posthoc_boundary":
            raise ValueError("Z9-AURORA evidence boundary must remain explicit")


CONFIG = AuroraConfig()
CONFIG.validate()


def _probability(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty rows-by-classes matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value")
    if np.any(result < -1e-12) or np.any(result > 1.0 + 1e-12):
        raise ValueError(f"{name} lies outside [0, 1]")
    row_sum = result.sum(axis=1, keepdims=True)
    if not np.allclose(row_sum, 1.0, atol=2e-6, rtol=0.0):
        raise ValueError(f"{name} rows do not sum to one")
    return result / row_sum


def fit_context_gate(
    train_temporal_positions: np.ndarray,
    config: AuroraConfig = CONFIG,
) -> Mapping[str, object]:
    config.validate()
    positions = np.asarray(train_temporal_positions, dtype=np.float64)
    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError("training temporal positions must be a non-empty vector")
    coverage = float(np.mean(np.isfinite(positions)))
    active = bool(coverage >= config.temporal_coverage_threshold)
    return {
        "temporal_coverage": coverage,
        "context_active": active,
        "selected_family": "context_stem" if active else "confidence_risk_pair",
        "fit_partition": "outer_training_metadata_only",
        "labels_consumed": False,
        "dataset_identity_consumed": False,
        "configuration": asdict(config),
    }


def confidence_risk_transport(
    stable_probability: np.ndarray,
    compact_prediction: np.ndarray,
    config: AuroraConfig = CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fixed low-margin disagreement correction."""

    config.validate()
    probability = _probability(stable_probability, name="stable probability")
    compact = np.asarray(compact_prediction, dtype=np.int64)
    if compact.shape != (len(probability),):
        raise ValueError("compact prediction must align with stable rows")
    if np.any(compact < 0) or np.any(compact >= probability.shape[1]):
        raise ValueError("compact prediction contains an invalid class")
    stable_prediction = np.argmax(probability, axis=1).astype(np.int64)
    ordered = np.sort(probability, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    route = (stable_prediction != compact) & (
        margin < config.confidence_margin_boundary
    )
    transported = np.array(probability, copy=True)
    for index in np.flatnonzero(route):
        source = int(stable_prediction[index])
        target = int(compact[index])
        transported[index, source], transported[index, target] = (
            transported[index, target],
            transported[index, source],
        )
    if not np.array_equal(np.argmax(transported[route], axis=1), compact[route]):
        raise RuntimeError("posterior transport did not realise compact decisions")
    return transported, route


def predict_proba(
    stable_probability: np.ndarray,
    compact_prediction: np.ndarray,
    context_probability: np.ndarray,
    gate: Mapping[str, object],
    config: AuroraConfig = CONFIG,
) -> tuple[np.ndarray, Mapping[str, object]]:
    stable = _probability(stable_probability, name="stable probability")
    context = _probability(context_probability, name="context probability")
    if stable.shape != context.shape:
        raise ValueError("stable and context posteriors must have identical shapes")
    if bool(gate.get("labels_consumed", True)) or bool(
        gate.get("dataset_identity_consumed", True)
    ):
        raise ValueError("AURORA gate must be label- and dataset-identity-free")
    if bool(gate.get("context_active", False)):
        return np.array(context, copy=True), {
            "selected_family": "context_stem",
            "compact_routes": 0,
            "rows": int(len(context)),
        }
    output, route = confidence_risk_transport(stable, compact_prediction, config)
    return output, {
        "selected_family": "confidence_risk_pair",
        "compact_routes": int(np.sum(route)),
        "rows": int(len(output)),
    }


__all__ = [
    "CONFIG",
    "MODEL_ID",
    "AuroraConfig",
    "confidence_risk_transport",
    "fit_context_gate",
    "predict_proba",
]
