"""CMUR-Fast stable expert: ANOVA-3072 coordinate selection + V45 model.

The V45 model and training use the same
contract (100 epochs, batch 256, lr 5e-4, wd 1e-3, label smoothing 0.04,
clip 2.0, seed 420608) with TF32 and AMP disabled.  Per outer fold:
ANOVA F-selection and z-scoring are fit on the training split only; the model
is trained on that split and evaluated on the untouched 20% test split.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import warnings
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score

from . import config as cfg
from .model_stable import V45Config, V45SalienceStratifiedPerceiver


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _enable_fast_path() -> None:
    # 纯 float32 训练，与 Mac 审计路径逐位一致（不用 TF32，避免训练轨迹
    # 漂移掉点）。速度仍由 WavLM batch-16 提取和 GPU 天然算力保证。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _reference_keys(records: Sequence[Mapping[str, str]]) -> np.ndarray:
    """Same identity as the audited chain: speaker|sentence_id, falling back
    to the sample id when a sentence id is absent."""
    keys: list[str] = []
    for record in records:
        speaker = str(record.get("speaker", "") or "unknown-speaker")
        sentence = str(record.get("sentence_id", "") or str(record["sample_id"]))
        keys.append(f"{speaker}|{sentence}")
    return np.asarray(keys, dtype=str)


def _reference_residualize(
    features: np.ndarray,
    keys: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    """Label-blind content residualization (audited chain recipe).

    Training rows are residualized against the leave-one-out centroid of their
    speaker|sentence group (falling back to the global centroid); evaluation
    rows are residualized against the stored training group centroids.  This
    removes speaker/content variation so the ANOVA selection and the model see
    emotion-discriminative coordinates only.
    """
    value = np.asarray(features, dtype=np.float32)
    train = value[train_indices]
    group_sum: dict[str, np.ndarray] = {}
    group_count: dict[str, int] = {}
    for row, key in zip(train, keys[train_indices]):
        group = str(key)
        if group not in group_sum:
            group_sum[group] = np.zeros(train.shape[1], dtype=np.float64)
            group_count[group] = 0
        group_sum[group] += row
        group_count[group] += 1
    global_sum = train.sum(axis=0, dtype=np.float64)
    global_centroid = global_sum / len(train)

    is_train = np.zeros(len(value), dtype=bool)
    is_train[train_indices] = True
    residual = np.empty_like(value, dtype=np.float32)
    for index, (row, key) in enumerate(zip(value, keys)):
        group = str(key)
        if is_train[index]:
            other_count = group_count[group] - 1
            if other_count > 0:
                reference = (group_sum[group] - row) / other_count
            else:
                reference = (global_sum - row) / (len(train) - 1)
        else:
            count = group_count.get(group, 0)
            if count:
                reference = group_sum[group] / count
            else:
                reference = global_centroid
        residual[index] = row - reference
    if not np.all(np.isfinite(residual)):
        raise RuntimeError("reference residualization produced non-finite values")
    return residual


def _stable_anova_rank(
    x: np.ndarray, y: np.ndarray, count: int
) -> np.ndarray:
    """Exact audited-chain ranking: sklearn f_classif + stable lexsort."""
    from sklearn.feature_selection import f_classif

    value = np.asarray(x, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            score, _ = f_classif(value, labels)
    finite = score[np.isfinite(score)]
    maximum = float(np.max(finite)) if finite.size else 0.0
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=max(1.0, maximum + abs(maximum) + 1.0),
        neginf=0.0,
    )
    indices = np.arange(value.shape[1], dtype=np.int64)
    return np.lexsort((indices, -score))[: min(int(count), value.shape[1])]


def _prepare_fold(
    features: np.ndarray,
    labels: np.ndarray,
    reference_keys: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    anova_selected: int | None = None,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    """Audited-chain preprocessing for one outer fold:
    reference-residual -> f_classif ANOVA -> training z-score.
    Returns (compact_train, compact_test, metadata); core = first 1024 cols.
    anova_selected 默认 cfg.ANOVA_SELECTED；特征维度不足时取 min（消融 F2/F3/F6）。
    """
    count = min(int(anova_selected or cfg.ANOVA_SELECTED), features.shape[1])
    residual = _reference_residualize(features, reference_keys, train_indices, test_indices)
    selected = _stable_anova_rank(
        residual[train_indices], labels[train_indices], count
    )
    if len(np.unique(selected)) != count:
        raise RuntimeError("ANOVA selection produced duplicate coordinates")
    train_compact = residual[train_indices][:, selected]
    mean = train_compact.mean(axis=0)
    std = np.maximum(train_compact.std(axis=0), 1e-6)
    compact_train = ((train_compact - mean) / std).astype(np.float32, copy=False)
    compact_test = ((residual[test_indices][:, selected] - mean) / std).astype(
        np.float32, copy=False
    )
    return compact_train, compact_test, {"selected_features": int(len(selected))}


def _loader(base: np.ndarray, core: np.ndarray, labels: np.ndarray, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(base, dtype=np.float32)),
        torch.from_numpy(np.asarray(core, dtype=np.float32)),
        torch.from_numpy(np.asarray(labels, dtype=np.int64)),
    )
    generator = torch.Generator().manual_seed(cfg.SEED)
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def _train_fold(
    model: V45SalienceStratifiedPerceiver,
    base: np.ndarray,
    core: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    *,
    run_name: str,
) -> list[Mapping[str, object]]:
    loader = _loader(base, core, labels, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    history: list[Mapping[str, object]] = []
    progress = tqdm(
        range(1, cfg.EPOCHS + 1),
        desc=f"[稳定专家 {run_name}]",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in progress:
        model.train()
        started = time.perf_counter()
        loss_sum = 0.0
        correct = rows = 0
        for batch_base, batch_core, batch_labels in loader:
            batch_base = batch_base.to(device, non_blocking=True)
            batch_core = batch_core.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_base, batch_core)
            loss = model.loss(output, batch_labels, label_smoothing=cfg.LABEL_SMOOTHING)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            optimizer.step()
            count = batch_labels.size(0)
            rows += count
            loss_sum += float(loss.item()) * count
            prediction = output["ensemble_probability"].argmax(dim=1)
            correct += int((prediction == batch_labels).sum().item())
        if rows != len(base):
            raise RuntimeError(f"{run_name} epoch did not cover all training rows")
        item = {
            "epoch": epoch,
            "training_rows": rows,
            "loss": loss_sum / rows,
            "ensemble_training_accuracy": correct / rows,
            "seconds": time.perf_counter() - started,
        }
        history.append(item)
        progress.set_postfix(
            loss=f"{item['loss']:.4f}", acc=f"{100 * item['ensemble_training_accuracy']:.2f}%",
            refresh=False,
        )
        if epoch % 25 == 0:
            progress.write(
                f"{run_name} epoch {epoch}/{cfg.EPOCHS}: "
                f"loss={item['loss']:.4f} acc={100 * item['ensemble_training_accuracy']:.2f}%"
            )
    progress.close()
    return history


def _predict_fold(
    model: V45SalienceStratifiedPerceiver,
    base: np.ndarray,
    core: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> Mapping[str, np.ndarray]:
    loader = _loader(base, core, labels, shuffle=False)
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_base, batch_core, _ in loader:
            batch_base = batch_base.to(device, non_blocking=True)
            batch_core = batch_core.to(device, non_blocking=True)
            output = model(batch_base, batch_core)
            probabilities.append(output["ensemble_probability"].float().cpu().numpy())
    probability = np.concatenate(probabilities, axis=0)
    prediction = probability.argmax(axis=1).astype(np.int64)
    return {"probability": probability, "prediction": prediction}


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> Mapping[str, object]:
    accuracy = float(np.mean(truth == prediction))
    macro_f1 = float(f1_score(truth, prediction, average="macro", zero_division=0))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "correct": int(np.sum(truth == prediction)),
        "samples": int(len(truth)),
    }


def run_stable_expert(
    dataset: str,
    records: Sequence[Mapping[str, str]],
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    device: torch.device,
    results_root: Path,
    label_order: Sequence[str] | None = None,
    anova_selected: int | None = None,
    seed: int | None = None,
) -> Mapping[str, object]:
    _enable_fast_path()
    if label_order is None:
        raise ValueError("label_order must be explicitly supplied")
    label_order = tuple(label_order)
    n_classes = len(label_order)
    anova_count = min(int(anova_selected or cfg.ANOVA_SELECTED), int(features.shape[1]))
    run_seed = int(seed) if seed is not None else cfg.SEED
    reference_keys = _reference_keys(records)
    oof_prediction = np.full(len(records), -1, dtype=np.int64)
    oof_probability = np.full((len(records), n_classes), np.nan, dtype=np.float64)
    fold_results: list[Mapping[str, object]] = []
    for fold in range(1, 6):
        test_indices = np.flatnonzero(folds == fold).astype(np.int64)
        train_indices = np.flatnonzero(folds != fold).astype(np.int64)
        if len(train_indices) * 4 != len(test_indices) * 16:
            raise RuntimeError(f"{dataset} fold {fold}: split is not exact 80/20")
        compact_train, compact_test, prep = _prepare_fold(
            features, labels, reference_keys, train_indices, test_indices,
            anova_selected=anova_count,
        )
        core_train = compact_train[:, : cfg.CORE_DIMENSION]
        core_test = compact_test[:, : cfg.CORE_DIMENSION]

        _seed_everything(run_seed)
        config = V45Config(
            classes=n_classes,
            base_dimension=int(anova_count),
            core_dimension=min(int(cfg.CORE_DIMENSION), int(anova_count)),
        )
        model = V45SalienceStratifiedPerceiver(
            variant="latent-cross-attention", config=config
        ).to(device)
        run_name = f"{dataset} fold {fold}/5"
        history = _train_fold(
            model, compact_train, core_train, labels[train_indices], device, run_name=run_name
        )
        truth = labels[test_indices]
        predicted = _predict_fold(model, compact_test, core_test, truth, device)
        prediction = predicted["prediction"]
        probability = predicted["probability"]
        metrics = _metrics(truth, prediction)
        oof_prediction[test_indices] = prediction
        oof_probability[test_indices] = probability
        fold_results.append(
            {
                "outer_fold": fold,
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "metrics": metrics,
                "selected_features": int(anova_count),
                "core_dimension": min(int(cfg.CORE_DIMENSION), int(anova_count)),
                "last_epoch": history[-1],
            }
        )
        tqdm.write(
            f"{dataset} fold {fold}/5 完成 | Accuracy={100 * metrics['accuracy']:.3f}% "
            f"| Macro-F1={100 * metrics['macro_f1']:.3f}% "
            f"| {metrics['correct']}/{metrics['samples']}"
        )
        del model
        torch.cuda.empty_cache()

    if np.any(oof_prediction < 0) or not np.all(np.isfinite(oof_probability)):
        raise RuntimeError(f"{dataset}: incomplete OOF coverage")
    pooled = _metrics(labels, oof_prediction)
    rows_out: list[dict[str, object]] = []
    for index, (record, prediction, probability) in enumerate(
        zip(records, oof_prediction, oof_probability)
    ):
        item: dict[str, object] = {
            "dataset": dataset,
            "outer_fold": int(folds[index]),
            "sample_id": str(record["sample_id"]),
            "relative_path": str(record["relative_path"]),
            "source_sha256": str(record["wav_sha256"]),
            "truth_index": int(labels[index]),
            "prediction_index": int(prediction),
        }
        for class_index, label in enumerate(label_order):
            item[f"probability_{label}"] = float(probability[class_index])
        rows_out.append(item)
    summary = {
        "dataset": dataset,
        "model": "V45 Salience-Stratified Latent Perceiver (CMUR-Fast)",
        "labels": list(label_order),
        "training_contract": {
            "epochs": cfg.EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay": cfg.WEIGHT_DECAY,
            "label_smoothing": cfg.LABEL_SMOOTHING,
            "gradient_clip": cfg.GRADIENT_CLIP,
            "seed": run_seed,
            "feature_dimension": cfg.TOTAL_FEATURE_DIMENSION,
            "anova_selected": cfg.ANOVA_SELECTED,
            "core_dimension": cfg.CORE_DIMENSION,
            "tf32": False,
            "amp": False,
        },
        "metrics": pooled,
        "folds": fold_results,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    oof_path = results_root / f"{dataset}_oof_predictions.csv"
    _atomic_csv(oof_path, rows_out)
    summary_path = results_root / f"{dataset}_summary.json"
    _atomic_json(summary_path, summary)
    return summary


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import csv

    if not rows:
        raise RuntimeError("cannot write empty predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
