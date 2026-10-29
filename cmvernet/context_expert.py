"""Uniform Z2 STEM context expert, with fold-local cross-fitting."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .context_helpers import (FROZEN_DIRECT_SUM, SupportGatedDirectSumSpeakerKernel, _project_blocks,
    NumericalSpec, BarycentricNuisanceQuotientRBF, FROZEN_CONTEXT_SPEC,
    SemanticContextTemporalPrior, ContentResidualRBF, RobustPreparedRBF, UniversalSpeakerConditionalRBF)
from .support_topology import EXPERT_ORDER, SupportTopologyEvidenceMixture, SupportTopologyMixtureConfig

MODEL_ID = "Z2-IEMOCAP-STEM-W40K-UNIFORM-v1"


FROZEN_NUMERICAL = NumericalSpec("k2048_c12_g0p7", 2048, 12.0, 0.70)


@dataclass(frozen=True)
class UniformStemConfig:
    outer_selected_dimension: int = 3072
    nested_core_dimension: int = 1024
    crossfit_folds: int = 5
    expert_seed: int = 2026
    router: SupportTopologyMixtureConfig = SupportTopologyMixtureConfig()
    implementation_version: str = "z2-stem-w40k-uniform-1.0"

    def validate(self) -> None:
        if self.outer_selected_dimension != 3072:
            raise ValueError("uniform STEM requires the frozen 3,072-D outer selection")
        if self.nested_core_dimension != 1024:
            raise ValueError("uniform STEM requires the frozen 1,024-D nested core")
        if self.crossfit_folds != 5 or self.expert_seed != 2026:
            raise ValueError("uniform STEM cross-fitting constants changed")


CONFIG = UniformStemConfig()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _values(values: Sequence[str], indices: np.ndarray) -> list[str]:
    return [str(values[int(index)]) for index in indices]


def _blocks(base: np.ndarray, side: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "base": np.asarray(base[indices], dtype=np.float32),
        # The historical direct-sum name is retained for API identity.  On the
        # uniform port this is the same 88-D normalised eGeMAPS side block for
        # every corpus, not a corpus-specific WSTMP cache.
        "wstmp": np.asarray(side[indices], dtype=np.float32),
    }


def _score_fitted_experts(
    *,
    base: np.ndarray,
    side: np.ndarray,
    labels: np.ndarray,
    speakers: Sequence[str],
    contents: Sequence[str],
    contexts: Sequence[str],
    positions: np.ndarray,
    fit_indices: np.ndarray,
    target_indices: np.ndarray,
    n_classes: int,
) -> tuple[dict[str, np.ndarray], Mapping[str, object]]:
    fit_x = np.asarray(base[fit_indices], dtype=np.float32)
    target_x = np.asarray(base[target_indices], dtype=np.float32)
    fit_y = np.asarray(labels[fit_indices], dtype=np.int64)
    fit_speakers = _values(speakers, fit_indices)
    target_speakers = _values(speakers, target_indices)
    fit_contents = _values(contents, fit_indices)
    target_contents = _values(contents, target_indices)
    fit_contexts = _values(contexts, fit_indices)
    target_contexts = _values(contexts, target_indices)

    conditional = UniversalSpeakerConditionalRBF(
        FROZEN_NUMERICAL, n_classes
    ).fit(fit_x, fit_y, fit_speakers)
    conditional_scores, conditional_coverage = conditional.decision_functions(
        target_x, target_speakers
    )

    content = ContentResidualRBF(FROZEN_NUMERICAL, n_classes).fit(
        fit_x, fit_y, fit_contents
    )
    content_score, content_coverage = content.decision_function(
        target_x, target_contents
    )

    # The original cross-corpus shared expert cannot be fitted jointly with
    # Henan's non-canonical six labels.  A fixed nested-core pooled RBF is used
    # for every corpus, preserving an independent fifth evidence geometry
    # without corpus-specific rules or hyperparameters.
    pooled_core = RobustPreparedRBF(FROZEN_NUMERICAL, n_classes).fit(
        fit_x[:, : CONFIG.nested_core_dimension], fit_y
    )
    pooled_score = pooled_core.decision_function(
        target_x[:, : CONFIG.nested_core_dimension]
    )

    context = SemanticContextTemporalPrior(FROZEN_CONTEXT_SPEC, n_classes).fit(
        fit_y,
        fit_contexts,
        np.asarray(positions[fit_indices], dtype=np.float64),
    )
    context_score, context_coverage = context.decision_function(
        conditional_scores["global"],
        target_contexts,
        np.asarray(positions[target_indices], dtype=np.float64),
    )

    bnq = BarycentricNuisanceQuotientRBF(
        FROZEN_NUMERICAL, n_classes
    ).fit(fit_x, fit_y, fit_speakers, fit_contents)
    bnq_score, bnq_coverage = bnq.decision_function(
        target_x, target_speakers, target_contents
    )

    fit_blocks, target_blocks, direct_projection = _project_blocks(
        _blocks(base, side, fit_indices),
        _blocks(base, side, target_indices),
        fit_speakers,
        target_speakers,
        fit_contents,
        target_contents,
    )
    direct = SupportGatedDirectSumSpeakerKernel(
        FROZEN_DIRECT_SUM, n_classes
    ).fit(fit_blocks, fit_y, fit_speakers)
    direct_score, direct_coverage = direct.decision_function(
        target_blocks, target_speakers
    )

    scores = {
        "global": np.asarray(conditional_scores["global"], dtype=np.float64),
        "speaker_local": np.asarray(
            conditional_scores["speaker_local"], dtype=np.float64
        ),
        "speaker_prior_adjusted": np.asarray(
            conditional_scores["speaker_prior_adjusted"], dtype=np.float64
        ),
        "content_residual": np.asarray(content_score, dtype=np.float64),
        "shared_pooled": np.asarray(pooled_score, dtype=np.float64),
        "semantic_context_temporal_prior": np.asarray(
            context_score, dtype=np.float64
        ),
        "bnq": np.asarray(bnq_score, dtype=np.float64),
        "direct_sum": np.asarray(direct_score, dtype=np.float64),
    }
    if tuple(scores) != EXPERT_ORDER:
        raise RuntimeError("uniform STEM expert order differs from the IEMOCAP router")
    expected = (len(target_indices), n_classes)
    for name, score in scores.items():
        if score.shape != expected or not np.all(np.isfinite(score)):
            raise RuntimeError(f"{name}: invalid expert score matrix {score.shape}")
    audit = {
        "fit_rows": int(len(fit_indices)),
        "target_rows": int(len(target_indices)),
        "fit_index_sha256": _array_sha256(fit_indices),
        "target_index_sha256": _array_sha256(target_indices),
        "conditional_coverage": conditional_coverage,
        "content_coverage": content_coverage,
        "context_coverage": context_coverage,
        "bnq_coverage": bnq_coverage,
        "direct_sum_coverage": direct_coverage,
        "direct_sum_projection": direct_projection,
        "fitted_models": {
            "conditional": conditional.metadata(),
            "content_residual": content.metadata(),
            "shared_pooled_uniform_port": pooled_core.metadata(),
            "semantic_context_temporal_prior": context.metadata(),
            "bnq": bnq.metadata(),
            "direct_sum": direct.metadata(),
        },
    }
    return scores, audit


def fit_predict_outer_fold(
    *,
    base: np.ndarray,
    side: np.ndarray,
    labels: np.ndarray,
    speakers: Sequence[str],
    contents: Sequence[str],
    contexts: Sequence[str],
    positions: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Mapping[str, object]:
    """Fit all experts/router on outer training and score outer test once."""

    base = np.asarray(base, dtype=np.float32)
    side = np.asarray(side, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    rows = len(labels)
    if base.shape != (rows, CONFIG.outer_selected_dimension):
        raise ValueError(f"base matrix must be ({rows}, 3072), got {base.shape}")
    if side.ndim != 2 or side.shape[0] != rows or side.shape[1] != 88:
        raise ValueError("uniform STEM side matrix must be rows x 88")
    if not all(len(values) == rows for values in (speakers, contents, contexts, positions)):
        raise ValueError("metadata arrays must align with features")
    if set(train_indices.tolist()).intersection(test_indices.tolist()):
        raise ValueError("outer train and test indices overlap")
    classes = np.unique(labels[train_indices])
    if not np.array_equal(classes, np.arange(len(classes), dtype=np.int64)):
        raise ValueError("outer training classes must be contiguous from zero")
    n_classes = int(len(classes))

    outer_y = labels[train_indices]
    oof_bank = {
        name: np.full((len(train_indices), n_classes), np.nan, dtype=np.float64)
        for name in EXPERT_ORDER
    }
    crossfit_audit = []
    splitter = StratifiedKFold(
        n_splits=CONFIG.crossfit_folds,
        shuffle=True,
        random_state=CONFIG.expert_seed,
    )
    for inner_fold, (fit_relative, hold_relative) in enumerate(
        splitter.split(np.zeros(len(outer_y)), outer_y), 1
    ):
        fit_absolute = train_indices[fit_relative]
        hold_absolute = train_indices[hold_relative]
        scores, audit = _score_fitted_experts(
            base=base,
            side=side,
            labels=labels,
            speakers=speakers,
            contents=contents,
            contexts=contexts,
            positions=positions,
            fit_indices=fit_absolute,
            target_indices=hold_absolute,
            n_classes=n_classes,
        )
        for name in EXPERT_ORDER:
            oof_bank[name][hold_relative] = scores[name]
        crossfit_audit.append(
            {
                "inner_fold": inner_fold,
                "fit_rows": int(len(fit_relative)),
                "hold_rows": int(len(hold_relative)),
                "score_sha256": {
                    name: _array_sha256(scores[name]) for name in EXPERT_ORDER
                },
                "fit_audit": audit,
            }
        )
    if any(not np.all(np.isfinite(oof_bank[name])) for name in EXPERT_ORDER):
        raise RuntimeError("uniform STEM train OOF bank is incomplete")

    router = SupportTopologyEvidenceMixture(CONFIG.router).fit(
        oof_bank,
        outer_y,
        _values(speakers, train_indices),
        positions[train_indices],
    )
    test_bank, final_fit_audit = _score_fitted_experts(
        base=base,
        side=side,
        labels=labels,
        speakers=speakers,
        contents=contents,
        contexts=contexts,
        positions=positions,
        fit_indices=train_indices,
        target_indices=test_indices,
        n_classes=n_classes,
    )
    probability = np.asarray(router.predict_proba(test_bank), dtype=np.float64)
    prediction = np.argmax(probability, axis=1).astype(np.int64)
    if probability.shape != (len(test_indices), n_classes):
        raise RuntimeError("uniform STEM output shape differs")
    if not np.all(np.isfinite(probability)) or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-10
    ):
        raise RuntimeError("uniform STEM emitted invalid probabilities")
    return {
        "prediction": prediction,
        "probability": probability,
        "router": router.metadata(),
        "configuration": {
            "uniform_stem": asdict(CONFIG),
            "rbf": asdict(FROZEN_NUMERICAL),
            "context": asdict(FROZEN_CONTEXT_SPEC),
            "direct_sum": asdict(FROZEN_DIRECT_SUM),
            "expert_order": list(EXPERT_ORDER),
        },
        "crossfit_audit": crossfit_audit,
        "final_fit_audit": final_fit_audit,
        "oof_score_sha256": {
            name: _array_sha256(oof_bank[name]) for name in EXPERT_ORDER
        },
        "test_score_sha256": {
            name: _array_sha256(test_bank[name]) for name in EXPERT_ORDER
        },
    }
