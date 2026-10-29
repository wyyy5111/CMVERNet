"""Helper implementations for the frozen Z2 context expert.

This file contains the classes/functions actually invoked by the original
Z2 context model, extracted without the surrounding experiment runners.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import warnings
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Extracted from scripts/dev_shared_multicorpus_ser.py
@dataclass(frozen=True)
class NumericalSpec:
    identifier: str
    k: int
    c: float
    gamma_factor: float


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _rank_percentiles(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            scores, _ = f_classif(X, y)
    # A perfectly separated feature legitimately yields +inf.  Treating it as
    # zero would invert its rank, so place it strictly above the largest finite
    # F statistic while mapping undefined/negative infinities to zero.
    finite = scores[np.isfinite(scores)]
    finite_max = float(np.max(finite)) if finite.size else 0.0
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=max(1.0, finite_max + abs(finite_max) + 1.0),
        neginf=0.0,
    )
    order = np.argsort(scores, kind="mergesort")
    percentiles = np.empty(len(order), dtype=np.float64)
    if len(order) == 1:
        percentiles[order] = 1.0
    else:
        percentiles[order] = np.linspace(0.0, 1.0, len(order))
    return percentiles


# Extracted from scripts/dev_kernel_screen.py
def _sha256_keys(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


# Extracted from scripts/dev_savee_fractional_affective_kernel_probe.py
def fractional_distance(left: np.ndarray, right: np.ndarray, p: float) -> np.ndarray:
    """Return mean coordinate-wise |x-z|**p without materialising a 3-D tensor."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("left/right must be aligned 2-D matrices")
    if not 0.0 < float(p) <= 2.0:
        raise ValueError("p must lie in (0, 2]")
    if float(p) == 1.0:
        distance = cdist(left, right, metric="cityblock")
    elif float(p) == 2.0:
        distance = cdist(left, right, metric="sqeuclidean")
    else:
        # SciPy's Minkowski distance raises the coordinate sum to 1/p; undo
        # that final root to obtain the negative-type fractional semimetric.
        distance = cdist(left, right, metric="minkowski", p=float(p)) ** float(p)
    return distance / float(left.shape[1])


# Extracted from scripts/dev_unified_context_temporal_prior.py
@dataclass(frozen=True)
class ContextPriorSpec:
    alpha: float
    temporal_bandwidth: float
    score_strength: float

    @property
    def identifier(self) -> str:
        return "a{:g}_t{:g}_b{:g}".format(
            self.alpha, self.temporal_bandwidth, self.score_strength
        )


class SemanticContextTemporalPrior:
    """Dirichlet-smoothed context posterior with optional temporal weighting."""

    def __init__(self, spec: ContextPriorSpec, n_classes: int):
        self.spec = spec
        self.n_classes = int(n_classes)

    def fit(
        self,
        y: np.ndarray,
        contexts: Sequence[str],
        positions: np.ndarray,
    ) -> "SemanticContextTemporalPrior":
        y = np.asarray(y, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.float64)
        if len(y) == 0 or len(y) != len(contexts) or len(y) != len(positions):
            raise ValueError("y/contexts/positions must be non-empty and aligned")
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        if np.any(counts <= 0):
            raise ValueError("context prior fit partition omitted a class")
        self.global_prior_ = counts / np.sum(counts)
        self.rows_: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for context in sorted(set(str(value) for value in contexts if str(value))):
            indices = np.asarray(
                [i for i, value in enumerate(contexts) if str(value) == context],
                dtype=np.int64,
            )
            self.rows_[context] = (y[indices].copy(), positions[indices].copy())
        return self

    def decision_function(
        self,
        base_scores: np.ndarray,
        contexts: Sequence[str],
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, Mapping[str, int]]:
        scores = np.asarray(base_scores, dtype=np.float64).copy()
        positions = np.asarray(positions, dtype=np.float64)
        if scores.shape != (len(contexts), self.n_classes):
            raise ValueError("base score shape does not match contexts/classes")
        adjusted = 0
        temporal = 0
        for index, (context, position) in enumerate(zip(contexts, positions)):
            rows = self.rows_.get(str(context))
            if rows is None:
                continue
            labels, train_positions = rows
            finite_temporal = np.isfinite(position) & np.isfinite(train_positions)
            if np.any(finite_temporal):
                weights = np.ones(len(labels), dtype=np.float64)
                weights[finite_temporal] = np.exp(
                    -np.abs(train_positions[finite_temporal] - position)
                    / self.spec.temporal_bandwidth
                )
                temporal += 1
            else:
                weights = np.ones(len(labels), dtype=np.float64)
            local_counts = np.bincount(
                labels, weights=weights, minlength=self.n_classes
            ).astype(np.float64)
            posterior = (
                local_counts + self.spec.alpha * self.global_prior_
            ) / (np.sum(local_counts) + self.spec.alpha)
            log_ratio = np.log(np.maximum(posterior, 1e-12)) - np.log(
                np.maximum(self.global_prior_, 1e-12)
            )
            scores[index] += self.spec.score_strength * log_ratio
            adjusted += 1
        return scores, {
            "context_adjusted_samples": int(adjusted),
            "temporal_kernel_samples": int(temporal),
            "global_fallback_samples": int(len(contexts) - adjusted),
        }

    def metadata(self) -> Mapping[str, object]:
        ordered = sorted(self.rows_)
        return {
            "spec": asdict(self.spec),
            "context_count": int(len(ordered)),
            "context_ids_sha256": _sha256_keys(ordered),
            "global_prior_sha256": _array_sha256(self.global_prior_),
            "fit_partition": "current_training_rows_only",
            "unseen_context_policy": "unchanged_global_acoustic_score",
            "equation": "s_c(x)+beta*log(p(c|context,time)/p_global(c))",
        }


# Extracted from scripts/dev_unified_speaker_conditional_ser.py
MINIMUM_LOCAL_SAMPLES = 48


MINIMUM_LOCAL_PER_CLASS = 4


PRIOR_ALPHA = 32.0


PRIOR_SCORE_COEFFICIENT = 0.35


class RobustPreparedRBF:
    """Train-only rank, scale, and RBF with stable handling of infinite F."""

    def __init__(self, numerical: NumericalSpec, n_classes: int):
        self.numerical = numerical
        self.n_classes = int(n_classes)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RobustPreparedRBF":
        percentile = _rank_percentiles(X, y)
        self.selected_ = np.argsort(-percentile, kind="mergesort")[
            : min(self.numerical.k, X.shape[1])
        ].astype(np.int64)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(X[:, self.selected_]).astype(np.float32)
        self.classifier_ = SVC(
            kernel="rbf",
            C=self.numerical.c,
            gamma=self.numerical.gamma_factor / scaled.shape[1],
            class_weight="balanced",
            cache_size=2048,
            decision_function_shape="ovr",
        )
        self.classifier_.fit(scaled, y)
        if len(self.classifier_.classes_) != self.n_classes:
            raise RuntimeError("RBF training partition omitted a class")
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scaled = self.scaler_.transform(X[:, self.selected_]).astype(np.float32)
        score = np.asarray(
            self.classifier_.decision_function(scaled), dtype=np.float64
        )
        if score.ndim == 1:
            score = np.stack([-score, score], axis=1)
        aligned = np.full((len(X), self.n_classes), -1e6, dtype=np.float64)
        aligned[:, np.asarray(self.classifier_.classes_, dtype=np.int64)] = score
        return aligned

    def metadata(self) -> Mapping[str, object]:
        return {
            "selected_features": int(len(self.selected_)),
            "selected_indices_sha256": _array_sha256(self.selected_),
            "ranking": "train_only_ANOVA_percentile_stable_inf_preserving",
            "scaler": "StandardScaler_train_only",
            "scaler_mean_sha256": _array_sha256(self.scaler_.mean_),
            "scaler_scale_sha256": _array_sha256(self.scaler_.scale_),
            "support_vectors_per_class": self.classifier_.n_support_.astype(int).tolist(),
        }


class UniversalSpeakerConditionalRBF:
    def __init__(self, numerical: NumericalSpec, n_classes: int):
        self.numerical = numerical
        self.n_classes = int(n_classes)

    def fit(
        self, X: np.ndarray, y: np.ndarray, speakers: Sequence[str]
    ) -> "UniversalSpeakerConditionalRBF":
        if len(X) != len(y) or len(X) != len(speakers):
            raise ValueError("X/y/speakers must align")
        self.global_model_ = RobustPreparedRBF(self.numerical, self.n_classes).fit(
            X, y
        )
        global_counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        if np.any(global_counts <= 0):
            raise RuntimeError("global training partition omitted a class")
        self.global_prior_ = global_counts / np.sum(global_counts)
        self.local_models_: Dict[str, RobustPreparedRBF] = {}
        self.local_audit_: Dict[str, Mapping[str, object]] = {}
        self.speaker_log_prior_ratio_: Dict[str, np.ndarray] = {}
        for speaker in sorted(set(str(value) for value in speakers)):
            indices = np.asarray(
                [index for index, value in enumerate(speakers) if str(value) == speaker],
                dtype=np.int64,
            )
            counts = np.bincount(y[indices], minlength=self.n_classes)
            posterior = (
                counts.astype(np.float64) + PRIOR_ALPHA * self.global_prior_
            ) / (len(indices) + PRIOR_ALPHA)
            self.speaker_log_prior_ratio_[speaker] = np.log(posterior) - np.log(
                self.global_prior_
            )
            eligible = bool(
                len(indices) >= MINIMUM_LOCAL_SAMPLES
                and np.all(counts >= MINIMUM_LOCAL_PER_CLASS)
            )
            audit: Dict[str, object] = {
                "samples": int(len(indices)),
                "class_counts": counts.astype(int).tolist(),
                "eligible": eligible,
                "minimum_local_samples": MINIMUM_LOCAL_SAMPLES,
                "minimum_local_per_class": MINIMUM_LOCAL_PER_CLASS,
            }
            if eligible:
                model = RobustPreparedRBF(self.numerical, self.n_classes).fit(
                    X[indices], y[indices]
                )
                self.local_models_[speaker] = model
                audit["policy"] = "speaker_local_model"
                audit["model"] = model.metadata()
            else:
                audit["policy"] = "global_fallback"
            self.local_audit_[speaker] = audit
        return self

    def decision_functions(
        self, X: np.ndarray, speakers: Sequence[str]
    ) -> Tuple[Mapping[str, np.ndarray], Mapping[str, object]]:
        if len(X) != len(speakers):
            raise ValueError("X/speakers must align")
        global_score = self.global_model_.decision_function(X)
        local_score = global_score.copy()
        prior_score = global_score.copy()
        local_samples = 0
        prior_samples = 0
        per_speaker: Dict[str, Mapping[str, object]] = {}
        for speaker in sorted(set(str(value) for value in speakers)):
            indices = np.asarray(
                [index for index, value in enumerate(speakers) if str(value) == speaker],
                dtype=np.int64,
            )
            model = self.local_models_.get(speaker)
            if model is not None:
                local_score[indices] = model.decision_function(X[indices])
                local_samples += len(indices)
                local_policy = "speaker_local_model"
            else:
                local_policy = "global_fallback"
            ratio = self.speaker_log_prior_ratio_.get(speaker)
            if ratio is not None:
                prior_score[indices] += PRIOR_SCORE_COEFFICIENT * ratio
                prior_samples += len(indices)
            per_speaker[speaker] = {
                "samples": int(len(indices)),
                "local_policy": local_policy,
                "prior_policy": (
                    "train_smoothed_speaker_prior"
                    if ratio is not None
                    else "global_prior_fallback"
                ),
            }
        return {
            "global": global_score,
            "speaker_local": local_score,
            "speaker_prior_adjusted": prior_score,
        }, {
            "local_model_samples": int(local_samples),
            "global_fallback_samples": int(len(X) - local_samples),
            "speaker_prior_samples": int(prior_samples),
            "global_prior_fallback_samples": int(len(X) - prior_samples),
            "per_speaker": per_speaker,
        }

    def metadata(self) -> Mapping[str, object]:
        ordered = sorted(self.speaker_log_prior_ratio_)
        ratio_matrix = np.vstack(
            [self.speaker_log_prior_ratio_[speaker] for speaker in ordered]
        )
        return {
            "numerical_spec": asdict(self.numerical),
            "global_model": self.global_model_.metadata(),
            "local_model_count": int(len(self.local_models_)),
            "local_models": self.local_audit_,
            "fixed_local_support": {
                "minimum_samples": MINIMUM_LOCAL_SAMPLES,
                "minimum_per_class": MINIMUM_LOCAL_PER_CLASS,
            },
            "speaker_prior": {
                "alpha": PRIOR_ALPHA,
                "score_coefficient": PRIOR_SCORE_COEFFICIENT,
                "speaker_ids_sha256": _sha256_keys(ordered),
                "log_ratio_sha256": _array_sha256(ratio_matrix),
                "unseen_speaker_policy": "global_prior_fallback",
            },
        }


# Extracted from scripts/dev_unified_oof_geometry_mixture.py
class SupportGatedResidualizer:
    """Subtract repeated content centroids; singleton/unseen groups use global."""

    def __init__(self, minimum_support: int = 2):
        self.minimum_support = int(minimum_support)

    def fit(
        self, X: np.ndarray, groups: Sequence[str]
    ) -> "SupportGatedResidualizer":
        if len(X) != len(groups) or len(X) == 0:
            raise ValueError("X/groups must be non-empty and aligned")
        self.global_centroid_ = np.mean(X, axis=0).astype(np.float32)
        indices_by_group: Dict[str, List[int]] = {}
        for index, group in enumerate(groups):
            group = str(group or "")
            if group:
                indices_by_group.setdefault(group, []).append(index)
        self.centroids_ = {
            group: np.mean(X[indices], axis=0).astype(np.float32)
            for group, indices in sorted(indices_by_group.items())
            if len(indices) >= self.minimum_support
        }
        self.supports_ = {
            group: int(len(indices))
            for group, indices in sorted(indices_by_group.items())
        }
        return self

    def transform(
        self, X: np.ndarray, groups: Sequence[str]
    ) -> Tuple[np.ndarray, Mapping[str, int]]:
        if len(X) != len(groups):
            raise ValueError("X/groups must align")
        output = np.empty_like(X, dtype=np.float32)
        seen = 0
        for index, group in enumerate(groups):
            centroid = self.centroids_.get(str(group or ""))
            if centroid is None:
                centroid = self.global_centroid_
            else:
                seen += 1
            output[index] = X[index] - centroid
        return output, {
            "repeated_content_centroid": int(seen),
            "global_centroid_fallback": int(len(X) - seen),
        }

    def metadata(self) -> Mapping[str, object]:
        ordered = sorted(self.centroids_)
        centroid_matrix = (
            np.vstack([self.centroids_[group] for group in ordered])
            if ordered
            else np.empty((0, len(self.global_centroid_)), dtype=np.float32)
        )
        return {
            "fit_partition": "current_training_rows_only",
            "minimum_group_support": self.minimum_support,
            "eligible_group_count": int(len(ordered)),
            "eligible_group_ids_sha256": _sha256_keys(ordered),
            "centroid_matrix_sha256": _array_sha256(centroid_matrix),
            "global_centroid_sha256": _array_sha256(self.global_centroid_),
            "singleton_unseen_or_empty_policy": "subtract_training_global_centroid",
        }


class ContentResidualRBF:
    def __init__(self, numerical: NumericalSpec, n_classes: int):
        self.numerical = numerical
        self.n_classes = int(n_classes)

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: Sequence[str]
    ) -> "ContentResidualRBF":
        self.residualizer_ = SupportGatedResidualizer(2).fit(X, groups)
        transformed, self.fit_coverage_ = self.residualizer_.transform(X, groups)
        percentile = _rank_percentiles(transformed, y)
        self.selected_ = np.argsort(-percentile, kind="mergesort")[
            : min(self.numerical.k, transformed.shape[1])
        ].astype(np.int64)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(
            transformed[:, self.selected_]
        ).astype(np.float32)
        self.classifier_ = SVC(
            kernel="rbf",
            C=self.numerical.c,
            gamma=self.numerical.gamma_factor / scaled.shape[1],
            class_weight="balanced",
            cache_size=2048,
            decision_function_shape="ovr",
        )
        self.classifier_.fit(scaled, y)
        if len(self.classifier_.classes_) != self.n_classes:
            raise RuntimeError("content RBF training fold omitted a class")
        return self

    def decision_function(
        self, X: np.ndarray, groups: Sequence[str]
    ) -> Tuple[np.ndarray, Mapping[str, int]]:
        transformed, coverage = self.residualizer_.transform(X, groups)
        scaled = self.scaler_.transform(
            transformed[:, self.selected_]
        ).astype(np.float32)
        score = np.asarray(self.classifier_.decision_function(scaled), dtype=np.float64)
        if score.ndim == 1:
            score = np.stack([-score, score], axis=1)
        aligned = np.full((len(X), self.n_classes), -1e6, dtype=np.float64)
        aligned[:, np.asarray(self.classifier_.classes_, dtype=np.int64)] = score
        return aligned, coverage

    def metadata(self) -> Mapping[str, object]:
        return {
            "selected_features": int(len(self.selected_)),
            "selected_indices_sha256": _array_sha256(self.selected_),
            "ranking": "train_only_ANOVA_percentile_stable_inf_preserving",
            "scaler_mean_sha256": _array_sha256(self.scaler_.mean_),
            "scaler_scale_sha256": _array_sha256(self.scaler_.scale_),
            "fit_coverage": self.fit_coverage_,
            "residualizer": self.residualizer_.metadata(),
        }


# Extracted from scripts/dev_unified_barycentric_nuisance_quotient.py
BNQ_ALPHA = 8.0


BNQ_ITERATIONS = 3


BNQ_MINIMUM_SUPPORT = 2


def _group_indices(values: Sequence[str], minimum_support: int):
    rows: Dict[str, List[int]] = {}
    for index, value in enumerate(values):
        identifier = str(value or "")
        if identifier:
            rows.setdefault(identifier, []).append(index)
    return {
        identifier: np.asarray(indices, dtype=np.int64)
        for identifier, indices in sorted(rows.items())
        if len(indices) >= minimum_support
    }, {identifier: int(len(indices)) for identifier, indices in sorted(rows.items())}


def _weighted_center_effects(
    effects: Mapping[str, np.ndarray], supports: Mapping[str, int]
) -> Dict[str, np.ndarray]:
    if not effects:
        return {}
    ordered = sorted(effects)
    weights = np.asarray([supports[name] for name in ordered], dtype=np.float64)
    matrix = np.vstack([effects[name] for name in ordered]).astype(
        np.float64, copy=False
    )
    barycentre = np.average(matrix, axis=0, weights=weights).astype(np.float32)
    return {
        name: (np.asarray(effects[name], dtype=np.float32) - barycentre).astype(
            np.float32
        )
        for name in ordered
    }


class BarycentricNuisanceProjector:
    """Train-only alternating shrinkage projection of two nuisance factors."""

    def __init__(
        self,
        alpha: float = BNQ_ALPHA,
        iterations: int = BNQ_ITERATIONS,
        minimum_support: int = BNQ_MINIMUM_SUPPORT,
    ) -> None:
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.minimum_support = int(minimum_support)
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.minimum_support < 1:
            raise ValueError("minimum support must be positive")

    def fit(
        self,
        X: np.ndarray,
        speakers: Sequence[str],
        contents: Sequence[str],
    ) -> "BarycentricNuisanceProjector":
        value = np.asarray(X, dtype=np.float32)
        if value.ndim != 2 or len(value) == 0:
            raise ValueError("X must be a non-empty matrix")
        if len(value) != len(speakers) or len(value) != len(contents):
            raise ValueError("X/speakers/contents must align")
        if not np.all(np.isfinite(value)):
            raise ValueError("X must be finite")
        self.global_centroid_ = np.mean(value, axis=0).astype(np.float32)
        speaker_rows, self.speaker_supports_ = _group_indices(
            speakers, self.minimum_support
        )
        content_rows, self.content_supports_ = _group_indices(
            contents, self.minimum_support
        )
        zero = np.zeros(value.shape[1], dtype=np.float32)
        speaker_effects = {name: zero.copy() for name in speaker_rows}
        content_effects = {name: zero.copy() for name in content_rows}
        for _ in range(self.iterations):
            updated_speaker = {}
            for name, indices in speaker_rows.items():
                content_mean = np.mean(
                    np.vstack(
                        [
                            content_effects.get(str(contents[int(i)]), zero)
                            for i in indices
                        ]
                    ),
                    axis=0,
                )
                residual_mean = (
                    np.mean(value[indices], axis=0)
                    - self.global_centroid_
                    - content_mean
                )
                weight = len(indices) / (len(indices) + self.alpha)
                updated_speaker[name] = (weight * residual_mean).astype(np.float32)
            speaker_effects = _weighted_center_effects(
                updated_speaker, self.speaker_supports_
            )
            updated_content = {}
            for name, indices in content_rows.items():
                speaker_mean = np.mean(
                    np.vstack(
                        [
                            speaker_effects.get(str(speakers[int(i)]), zero)
                            for i in indices
                        ]
                    ),
                    axis=0,
                )
                residual_mean = (
                    np.mean(value[indices], axis=0)
                    - self.global_centroid_
                    - speaker_mean
                )
                weight = len(indices) / (len(indices) + self.alpha)
                updated_content[name] = (weight * residual_mean).astype(np.float32)
            content_effects = _weighted_center_effects(
                updated_content, self.content_supports_
            )
        self.speaker_effects_ = speaker_effects
        self.content_effects_ = content_effects
        return self

    def transform(
        self,
        X: np.ndarray,
        speakers: Sequence[str],
        contents: Sequence[str],
    ) -> Tuple[np.ndarray, Mapping[str, int]]:
        value = np.asarray(X, dtype=np.float32)
        if value.ndim != 2 or len(value) != len(speakers) or len(value) != len(
            contents
        ):
            raise ValueError("X/speakers/contents must be aligned")
        output = value.copy()
        speaker_seen = 0
        content_seen = 0
        for index, (speaker, content) in enumerate(zip(speakers, contents)):
            speaker_effect = self.speaker_effects_.get(str(speaker or ""))
            content_effect = self.content_effects_.get(str(content or ""))
            if speaker_effect is not None:
                output[index] -= speaker_effect
                speaker_seen += 1
            if content_effect is not None:
                output[index] -= content_effect
                content_seen += 1
        return output, {
            "speaker_effect_samples": int(speaker_seen),
            "speaker_zero_fallback_samples": int(len(value) - speaker_seen),
            "content_effect_samples": int(content_seen),
            "content_zero_fallback_samples": int(len(value) - content_seen),
        }

    def metadata(self) -> Mapping[str, object]:
        speaker_ids = sorted(self.speaker_effects_)
        content_ids = sorted(self.content_effects_)
        dimension = len(self.global_centroid_)
        speaker_matrix = (
            np.vstack([self.speaker_effects_[name] for name in speaker_ids])
            if speaker_ids
            else np.empty((0, dimension), dtype=np.float32)
        )
        content_matrix = (
            np.vstack([self.content_effects_[name] for name in content_ids])
            if content_ids
            else np.empty((0, dimension), dtype=np.float32)
        )
        return {
            "fit_partition": "current_training_rows_only",
            "equation": "x - centred_ridge_speaker_effect - centred_ridge_content_effect",
            "alpha": self.alpha,
            "iterations": self.iterations,
            "minimum_support": self.minimum_support,
            "speaker_effect_count": len(speaker_ids),
            "content_effect_count": len(content_ids),
            "speaker_ids_sha256": _sha256_keys(speaker_ids),
            "content_ids_sha256": _sha256_keys(content_ids),
            "speaker_effects_sha256": _array_sha256(speaker_matrix),
            "content_effects_sha256": _array_sha256(content_matrix),
            "global_centroid_sha256": _array_sha256(self.global_centroid_),
            "unseen_or_low_support_policy": "zero_nuisance_effect",
            "label_free": True,
        }


class BarycentricNuisanceQuotientRBF:
    def __init__(self, numerical, n_classes: int):
        self.numerical = numerical
        self.n_classes = int(n_classes)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        speakers: Sequence[str],
        contents: Sequence[str],
    ) -> "BarycentricNuisanceQuotientRBF":
        self.projector_ = BarycentricNuisanceProjector().fit(
            X, speakers, contents
        )
        transformed, self.fit_coverage_ = self.projector_.transform(
            X, speakers, contents
        )
        self.classifier_ = RobustPreparedRBF(
            self.numerical, self.n_classes
        ).fit(transformed, y)
        return self

    def decision_function(
        self,
        X: np.ndarray,
        speakers: Sequence[str],
        contents: Sequence[str],
    ) -> Tuple[np.ndarray, Mapping[str, int]]:
        transformed, coverage = self.projector_.transform(X, speakers, contents)
        return self.classifier_.decision_function(transformed), coverage

    def metadata(self) -> Mapping[str, object]:
        return {
            "numerical_spec": asdict(self.numerical),
            "projector": self.projector_.metadata(),
            "classifier": self.classifier_.metadata(),
            "fit_coverage": self.fit_coverage_,
        }


# Extracted from src/models/direct_sum_fractional_kernel.py
@dataclass(frozen=True)
class DirectSumFractionalConfig:
    block_names: Tuple[str, ...] = ("base", "wstmp")
    selected_features: Tuple[int, ...] = (512, 512)
    powers: Tuple[float, ...] = (1.5, 1.5)
    weights: Tuple[float, ...] = (0.8, 0.2)
    gamma_factor: float = 0.7
    c: float = 12.0

    def validate(self) -> None:
        length = len(self.block_names)
        if length < 1 or not (
            len(self.selected_features)
            == len(self.powers)
            == len(self.weights)
            == length
        ):
            raise ValueError("all block configuration tuples must align")
        if len(set(self.block_names)) != length:
            raise ValueError("block names must be unique")
        if any(int(k) < 1 for k in self.selected_features):
            raise ValueError("selected feature counts must be positive")
        if any(not 0.0 < float(p) <= 2.0 for p in self.powers):
            raise ValueError("fractional powers must lie in (0,2]")
        if any(float(weight) < 0.0 for weight in self.weights):
            raise ValueError("block weights must be non-negative")
        if not np.isclose(sum(self.weights), 1.0, atol=1e-12):
            raise ValueError("block weights must sum to one")
        if self.gamma_factor <= 0.0 or self.c <= 0.0:
            raise ValueError("gamma_factor and C must be positive")


class DirectSumFractionalSVC:
    """Train-only per-block preprocessing and one direct-sum precomputed SVC."""

    def __init__(self, config: DirectSumFractionalConfig, n_classes: int):
        config.validate()
        self.config = config
        self.n_classes = int(n_classes)

    def _validate_blocks(
        self, blocks: Mapping[str, np.ndarray], expected_rows: int | None = None
    ) -> int:
        if tuple(blocks) != self.config.block_names:
            raise ValueError("block mapping order/names differ from configuration")
        row_counts = {len(np.asarray(blocks[name])) for name in self.config.block_names}
        if len(row_counts) != 1:
            raise ValueError("block row counts differ")
        rows = row_counts.pop()
        if expected_rows is not None and rows != expected_rows:
            raise ValueError("block row count differs from labels")
        for name in self.config.block_names:
            value = np.asarray(blocks[name])
            if value.ndim != 2 or value.shape[1] < 1 or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid finite 2-D block: {name}")
        return rows

    def _distance(
        self,
        left: Mapping[str, np.ndarray],
        right: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        total = None
        for position, name in enumerate(self.config.block_names):
            current = fractional_distance(
                left[name], right[name], self.config.powers[position]
            )
            weighted = self.config.weights[position] * current
            total = weighted if total is None else total + weighted
        return np.asarray(total, dtype=np.float64)

    def fit(
        self, blocks: Mapping[str, np.ndarray], y: np.ndarray
    ) -> "DirectSumFractionalSVC":
        y = np.asarray(y, dtype=np.int64)
        self._validate_blocks(blocks, len(y))
        if len(y) == 0:
            raise ValueError("training partition is empty")
        self.selected_ = {}
        self.scalers_ = {}
        self.train_scaled_ = {}
        for position, name in enumerate(self.config.block_names):
            X = np.asarray(blocks[name])
            percentile = _rank_percentiles(X, y)
            selected = np.argsort(-percentile, kind="mergesort")[
                : min(self.config.selected_features[position], X.shape[1])
            ].astype(np.int64)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(X[:, selected]).astype(np.float64)
            self.selected_[name] = selected
            self.scalers_[name] = scaler
            self.train_scaled_[name] = scaled
        kernel = np.exp(
            -self.config.gamma_factor
            * self._distance(self.train_scaled_, self.train_scaled_)
        )
        self.classifier_ = SVC(
            kernel="precomputed",
            C=self.config.c,
            class_weight="balanced",
            cache_size=2048,
            decision_function_shape="ovr",
        ).fit(kernel, y)
        return self

    def _transform(self, blocks: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        self._validate_blocks(blocks)
        return {
            name: self.scalers_[name].transform(
                np.asarray(blocks[name])[:, self.selected_[name]]
            ).astype(np.float64)
            for name in self.config.block_names
        }

    def decision_function(self, blocks: Mapping[str, np.ndarray]) -> np.ndarray:
        transformed = self._transform(blocks)
        kernel = np.exp(
            -self.config.gamma_factor
            * self._distance(transformed, self.train_scaled_)
        )
        score = np.asarray(self.classifier_.decision_function(kernel), dtype=np.float64)
        if score.ndim == 1:
            score = np.stack([-score, score], axis=1)
        aligned = np.full((len(score), self.n_classes), -1e6, dtype=np.float64)
        aligned[:, np.asarray(self.classifier_.classes_, dtype=np.int64)] = score
        return aligned

    def predict(self, blocks: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.argmax(self.decision_function(blocks), axis=1).astype(np.int64)

    def metadata(self) -> Mapping[str, object]:
        return {
            "configuration": asdict(self.config),
            "equation": (
                "exp(-gamma_factor * sum_b weight_b * "
                "mean(abs(x_b-z_b)**power_b))"
            ),
            "selected_indices_sha256": {
                name: _array_sha256(self.selected_[name])
                for name in self.config.block_names
            },
            "scaler_mean_sha256": {
                name: _array_sha256(self.scalers_[name].mean_)
                for name in self.config.block_names
            },
            "support_vectors_per_class": self.classifier_.n_support_.astype(int).tolist(),
        }


FROZEN_NUMERICAL = NumericalSpec("k2048_c12_g0p7", 2048, 12.0, 0.70)
FROZEN_CONTEXT_SPEC = ContextPriorSpec(4.0, 16.0, 0.75)

FROZEN_DIRECT_SUM = DirectSumFractionalConfig(
    selected_features=(512, 256),
    powers=(1.5, 1.5),
    weights=(0.96, 0.04),
    gamma_factor=0.7,
    c=4.0,
)


def _project_blocks(
    fit_blocks: Mapping[str, np.ndarray],
    target_blocks: Mapping[str, np.ndarray],
    fit_speakers: Sequence[str],
    target_speakers: Sequence[str],
    fit_contents: Sequence[str],
    target_contents: Sequence[str],
) -> Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray], Mapping[str, object]]:
    transformed_fit = {}
    transformed_target = {}
    audit = {}
    for name in FROZEN_DIRECT_SUM.block_names:
        projector = BarycentricNuisanceProjector().fit(
            fit_blocks[name], fit_speakers, fit_contents
        )
        transformed_fit[name], fit_coverage = projector.transform(
            fit_blocks[name], fit_speakers, fit_contents
        )
        transformed_target[name], target_coverage = projector.transform(
            target_blocks[name], target_speakers, target_contents
        )
        audit[name] = {
            "projector": projector.metadata(),
            "fit_coverage": fit_coverage,
            "target_coverage": target_coverage,
        }
    return transformed_fit, transformed_target, audit


class SupportGatedDirectSumSpeakerKernel:
    """Speaker-local direct-sum kernel with the frozen base-RBF fallback."""

    def __init__(self, config: DirectSumFractionalConfig, n_classes: int):
        self.config = config
        self.n_classes = int(n_classes)

    def fit(
        self,
        blocks: Mapping[str, np.ndarray],
        y: np.ndarray,
        speakers: Sequence[str],
    ) -> "SupportGatedDirectSumSpeakerKernel":
        self.fallback_ = UniversalSpeakerConditionalRBF(
            FROZEN_NUMERICAL, self.n_classes
        ).fit(blocks["base"], y, speakers)
        self.models_: Dict[str, DirectSumFractionalSVC] = {}
        self.support_: Dict[str, Mapping[str, object]] = {}
        for speaker in sorted(set(str(value or "") for value in speakers)):
            indices = np.asarray(
                [i for i, value in enumerate(speakers) if str(value or "") == speaker],
                dtype=np.int64,
            )
            counts = np.bincount(y[indices], minlength=self.n_classes)
            eligible = bool(
                speaker
                and len(indices) >= MINIMUM_LOCAL_SAMPLES
                and np.all(counts >= MINIMUM_LOCAL_PER_CLASS)
            )
            if eligible:
                local_blocks = {
                    name: blocks[name][indices] for name in self.config.block_names
                }
                self.models_[speaker] = DirectSumFractionalSVC(
                    self.config, self.n_classes
                ).fit(local_blocks, y[indices])
            self.support_[speaker] = {
                "samples": int(len(indices)),
                "class_counts": counts.astype(int).tolist(),
                "eligible": eligible,
            }
        return self

    def decision_function(
        self, blocks: Mapping[str, np.ndarray], speakers: Sequence[str]
    ) -> Tuple[np.ndarray, Mapping[str, object]]:
        fallback, _ = self.fallback_.decision_functions(blocks["base"], speakers)
        output = np.asarray(fallback["global"], dtype=np.float64).copy()
        local_samples = 0
        for speaker in sorted(set(str(value or "") for value in speakers)):
            indices = np.asarray(
                [i for i, value in enumerate(speakers) if str(value or "") == speaker],
                dtype=np.int64,
            )
            model = self.models_.get(speaker)
            if model is not None:
                local_blocks = {
                    name: blocks[name][indices] for name in self.config.block_names
                }
                output[indices] = model.decision_function(local_blocks)
                local_samples += len(indices)
        return output, {
            "local_direct_sum_samples": int(local_samples),
            "global_base_RBF_fallback_samples": int(len(output) - local_samples),
            "eligible_speaker_models": int(len(self.models_)),
        }

    def metadata(self) -> Mapping[str, object]:
        return {
            "configuration": asdict(self.config),
            "minimum_local_samples": MINIMUM_LOCAL_SAMPLES,
            "minimum_local_per_class": MINIMUM_LOCAL_PER_CLASS,
            "support": self.support_,
            "local_model_count": len(self.models_),
            "local_models": {
                speaker: model.metadata() for speaker, model in self.models_.items()
            },
            "fallback": "frozen_base_BNQ_space_global_RBF",
            "fallback_model": self.fallback_.metadata(),
        }
