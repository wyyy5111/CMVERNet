"""Corpus-blind support-topology evidence mixture for speech emotion recognition.

The model has the same three heads and the same numerical configuration for
every corpus.  It does not accept a corpus name.  Instead, a *training-only*
support topology softly routes evidence according to quantities that are
available in any labelled SER training set:

``t``
    Fraction of training utterances with a finite temporal coordinate.
``l``
    Fraction supported by a speaker-local classifier (enough samples from
    every class for that speaker).
``m``
    Effective number of speakers, i.e. ``exp(entropy(speaker frequency))``.

With logistic step ``sigma_a(u) = 1 / (1 + exp(-a u))``, the gates are

``g_T = sigma_a(t - tau_T)``
``g_D = (1-g_T) sigma_a(l-tau_L) sigma_a(m-tau_M)``
``g_C = 1-g_T-g_D``.

The temporal head is an OOF-trained ridge-logit evidence stacker.  The direct
sum and content-residual heads are two members of the same fixed expert bank.
All heads are retained and the gates are continuous; different fitted gate
values are ordinary learned parameters, not per-corpus hyperparameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERT_ORDER = (
    "global",
    "speaker_local",
    "speaker_prior_adjusted",
    "content_residual",
    "shared_pooled",
    "semantic_context_temporal_prior",
    "bnq",
    "direct_sum",
)


@dataclass(frozen=True)
class SupportTopologyMixtureConfig:
    """One numerical specification shared by every fitted corpus model."""

    temporal_threshold: float = 0.50
    local_support_threshold: float = 0.95
    effective_speaker_threshold: float = 1.50
    gate_slope: float = 20.0
    minimum_local_samples: int = 48
    minimum_local_per_class: int = 4
    meta_c: float = 0.01
    meta_max_iter: int = 4000
    seed: int = 2026


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -50.0, 50.0))
    return exponent / np.maximum(np.sum(exponent, axis=1, keepdims=True), 1e-12)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0))))


def evidence_features(score_bank: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return a class-count-aware but corpus-identity-free evidence map."""

    missing = [name for name in EXPERT_ORDER if name not in score_bank]
    if missing:
        raise ValueError(f"score_bank omitted experts: {missing}")
    blocks = []
    sample_count = None
    class_count = None
    for expert in EXPERT_ORDER:
        score = np.asarray(score_bank[expert], dtype=np.float64)
        if score.ndim != 2 or score.shape[0] == 0 or score.shape[1] < 2:
            raise ValueError(f"{expert}: scores must be a non-empty 2D matrix")
        if not np.all(np.isfinite(score)):
            raise ValueError(f"{expert}: scores contain non-finite values")
        if sample_count is None:
            sample_count, class_count = score.shape
        elif score.shape != (sample_count, class_count):
            raise ValueError("all expert score matrices must have the same shape")
        probability = _softmax(score)
        ordered = np.sort(probability, axis=1)
        margin = (ordered[:, -1] - ordered[:, -2])[:, None]
        entropy = -np.sum(
            probability * np.log(np.maximum(probability, 1e-12)),
            axis=1,
            keepdims=True,
        ) / np.log(score.shape[1])
        prediction = np.argmax(score, axis=1)
        one_hot = np.eye(score.shape[1], dtype=np.float64)[prediction]
        blocks.extend((score, probability, one_hot, margin, entropy))
    return np.concatenate(blocks, axis=1).astype(np.float32)


class SupportTopologyEvidenceMixture:
    """Softly route a fixed expert bank using training support topology."""

    def __init__(self, config: SupportTopologyMixtureConfig | None = None):
        self.config = config or SupportTopologyMixtureConfig()
        if self.config.gate_slope <= 0:
            raise ValueError("gate_slope must be positive")
        if self.config.minimum_local_samples < 1:
            raise ValueError("minimum_local_samples must be positive")
        if self.config.minimum_local_per_class < 1:
            raise ValueError("minimum_local_per_class must be positive")
        if self.config.meta_c <= 0:
            raise ValueError("meta_c must be positive")

    def fit(
        self,
        score_bank: Mapping[str, np.ndarray],
        y: np.ndarray,
        speakers: Sequence[str],
        temporal_positions: np.ndarray,
    ) -> "SupportTopologyEvidenceMixture":
        x = evidence_features(score_bank)
        y = np.asarray(y, dtype=np.int64)
        positions = np.asarray(temporal_positions, dtype=np.float64)
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("y must be one-dimensional and aligned to scores")
        if len(speakers) != len(y) or len(positions) != len(y):
            raise ValueError("speakers/temporal_positions must align to scores")
        if any(not str(value) for value in speakers):
            raise ValueError("speaker identifiers must be non-empty")
        classes = np.unique(y)
        expected = np.arange(len(classes), dtype=np.int64)
        if not np.array_equal(classes, expected):
            raise ValueError("classes must be contiguous integers starting at zero")
        self.n_classes_ = int(len(classes))

        self.meta_model_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.config.meta_c,
                class_weight="balanced",
                max_iter=self.config.meta_max_iter,
                solver="lbfgs",
                random_state=self.config.seed,
            ),
        ).fit(x, y)

        speaker_values = np.asarray([str(value) for value in speakers])
        unique_speakers, speaker_counts = np.unique(
            speaker_values, return_counts=True
        )
        eligible = {}
        for speaker in unique_speakers:
            indices = np.flatnonzero(speaker_values == speaker)
            counts = np.bincount(y[indices], minlength=self.n_classes_)
            eligible[str(speaker)] = bool(
                len(indices) >= self.config.minimum_local_samples
                and np.all(counts >= self.config.minimum_local_per_class)
            )
        local_support = float(
            np.mean([eligible[str(value)] for value in speaker_values])
        )
        proportions = speaker_counts.astype(np.float64) / len(speaker_values)
        effective_speakers = float(
            np.exp(-np.sum(proportions * np.log(np.maximum(proportions, 1e-12))))
        )
        temporal_coverage = float(np.mean(np.isfinite(positions)))

        slope = self.config.gate_slope
        temporal_gate = _sigmoid(
            slope * (temporal_coverage - self.config.temporal_threshold)
        )
        direct_gate = (
            (1.0 - temporal_gate)
            * _sigmoid(
                slope
                * (local_support - self.config.local_support_threshold)
            )
            * _sigmoid(
                slope
                * (
                    effective_speakers
                    - self.config.effective_speaker_threshold
                )
            )
        )
        content_gate = 1.0 - temporal_gate - direct_gate
        gates = np.asarray(
            [temporal_gate, direct_gate, content_gate], dtype=np.float64
        )
        if np.any(gates < -1e-12) or not np.isclose(np.sum(gates), 1.0):
            raise RuntimeError("invalid support-topology gates")
        self.gates_ = np.maximum(gates, 0.0)
        self.gates_ /= np.sum(self.gates_)
        self.topology_ = {
            "temporal_coordinate_coverage": temporal_coverage,
            "speaker_local_support_fraction": local_support,
            "effective_speaker_count": effective_speakers,
            "observed_speaker_count": int(len(unique_speakers)),
            "eligible_speaker_count": int(sum(eligible.values())),
        }
        self.eligible_speakers_ = eligible
        return self

    def predict_proba(self, score_bank: Mapping[str, np.ndarray]) -> np.ndarray:
        if not hasattr(self, "meta_model_"):
            raise RuntimeError("model must be fitted before prediction")
        x = evidence_features(score_bank)
        meta_probability = np.asarray(
            self.meta_model_.predict_proba(x), dtype=np.float64
        )
        direct_probability = _softmax(
            np.asarray(score_bank["direct_sum"], dtype=np.float64)
        )
        content_probability = _softmax(
            np.asarray(score_bank["content_residual"], dtype=np.float64)
        )
        if meta_probability.shape[1] != self.n_classes_:
            raise RuntimeError("meta model omitted a fitted class")
        probability = (
            self.gates_[0] * meta_probability
            + self.gates_[1] * direct_probability
            + self.gates_[2] * content_probability
        )
        return probability / np.maximum(
            np.sum(probability, axis=1, keepdims=True), 1e-12
        )

    def predict(self, score_bank: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.argmax(self.predict_proba(score_bank), axis=1).astype(np.int64)

    def metadata(self) -> Mapping[str, object]:
        if not hasattr(self, "topology_"):
            raise RuntimeError("model must be fitted before requesting metadata")
        names = ("temporal_meta", "direct_sum", "content_residual")
        return {
            "configuration": asdict(self.config),
            "expert_order": list(EXPERT_ORDER),
            "support_topology": dict(self.topology_),
            "gates": {
                name: float(value) for name, value in zip(names, self.gates_)
            },
            "dominant_head": names[int(np.argmax(self.gates_))],
            "dataset_identity_input": False,
            "equations": {
                "effective_speakers": "exp(-sum_s q_s log(q_s))",
                "temporal_gate": "sigmoid(a*(t-tau_T))",
                "direct_gate": (
                    "(1-g_T)*sigmoid(a*(l-tau_L))*sigmoid(a*(m-tau_M))"
                ),
                "content_gate": "1-g_T-g_D",
            },
        }
