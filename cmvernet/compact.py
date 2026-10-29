"""Compact expert: original fold-local preprocessing and RBF-SVM."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import torch

DEFAULT_K = 8192
DEFAULT_C = 8.0
DEFAULT_GAMMA_FACTOR = 0.40

def _anova_f_gpu(X: torch.Tensor, y: torch.Tensor, classes: Sequence[int]) -> torch.Tensor:
    """GPU 版 sklearn f_classif，复刻 Mac `_stable_rank` 语义：

    f = msb / msw；msw 不 clamp（常数特征 → inf/nan），随后 nan/inf/neginf
    一律置 0 —— 与 dev_kernel_screen._stable_rank 的 nan_to_num 一致。
    完全分离/常数特征排在最后，避免特征泄漏型过拟合。
    """
    n, dim = X.shape
    k = len(classes)
    df_between = max(k - 1, 1)
    df_within = max(n - k, 1)
    global_mean = X.mean(dim=0)
    ssb = torch.zeros(dim, device=X.device)
    ssw = torch.zeros(dim, device=X.device)
    for cls in classes:
        mask = y == cls
        count = int(mask.sum().item())
        if count < 1:
            continue
        part = X[mask]
        center = part.mean(dim=0)
        ssb = ssb + count * (center - global_mean).pow(2)
        if count > 1:
            ssw = ssw + (part - center).pow(2).sum(dim=0)
    f = (ssb / df_between) / (ssw / df_within)
    f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return f


def _speaker_center_gpu(
    X: torch.Tensor, speakers: np.ndarray, train_mask: np.ndarray
) -> torch.Tensor:
    """Subtract train-only per-speaker mean on GPU; OOS speakers use the global
    train mean (same convention as the frozen speaker-residual preprocessing)."""
    device = X.device
    speakers_t = torch.from_numpy(np.asarray(speakers, dtype=np.int64)).to(device)
    train_mask_t = torch.from_numpy(np.asarray(train_mask, dtype=bool)).to(device)
    global_mean = X[train_mask_t].mean(dim=0)
    centered = X.clone()
    for sp in np.unique(speakers):
        mask = speakers_t == int(sp)
        in_train = mask & train_mask_t
        if int(in_train.sum().item()) > 0:
            mu = X[in_train].mean(dim=0)
        else:
            mu = global_mean
        centered[mask] = X[mask] - mu
    return centered


def _fit_predict_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    c: float,
    gamma_factor: float,
) -> np.ndarray:
    """Fold-local ANOVA selection, standardization, and RBF-SVM prediction.

    Feature prep (ANOVA F-ranking, top-k selection, z-score) runs on CUDA; the
    RBF-SVM uses the exact libsvm solver (0.1-3 s/fold).  MLP committees and
    broad SVM voting were tested and *hurt* accuracy on small corpora
    (overfitting / vote dilution), so a single well-tuned configuration wins.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr = torch.from_numpy(np.asarray(X_train, dtype=np.float32)).to(device)
    y_tr = torch.from_numpy(np.asarray(y_train, dtype=np.int64)).to(device)
    X_te = torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(device)

    # GPU: one-way ANOVA F-ranking (exact sklearn f_classif semantics; inf/nan
    # constant features rank last to avoid leakage, matching the frozen chain).
    classes = [int(value) for value in np.unique(y_train)]
    f = _anova_f_gpu(X_tr, y_tr, classes)
    order = torch.argsort(-f, stable=True)
    selected = order[: min(k, order.shape[0])]

    # GPU: top-k selection + z-score (sklearn StandardScaler, biased ddof=0).
    X_tr_s = X_tr[:, selected]
    X_te_s = X_te[:, selected]
    center = X_tr_s.mean(dim=0)
    scale = X_tr_s.std(dim=0, unbiased=False).clamp_min(1e-8)
    X_tr_z = (X_tr_s - center) / scale
    X_te_z = (X_te_s - center) / scale

    from sklearn.svm import SVC

    gamma = float(gamma_factor) / int(X_tr_z.shape[1])
    classifier = SVC(
        C=float(c),
        gamma=gamma,
        class_weight="balanced",
        kernel="rbf",
        decision_function_shape="ovo",
        tol=1e-4,
        cache_size=2048,
    )
    classifier.fit(X_tr_z.cpu().numpy(), y_train)
    prediction = classifier.predict(X_te_z.cpu().numpy())
    return prediction


def predict_outer_fold(features, labels, speakers, train_indices, test_indices,
                       k=DEFAULT_K, c=DEFAULT_C, gamma_factor=DEFAULT_GAMMA_FACTOR):
    """Train the compact expert and predict one outer test fold.

    Speaker means, selection and standardization are fitted on training rows.
    Speaker identifiers may be strings or integers.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    speakers = np.asarray(speakers)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (len(features),) or speakers.shape != (len(features),):
        raise ValueError("features, labels and speakers must align")
    if not np.isfinite(features).all() or not len(train_indices) or not len(test_indices):
        raise ValueError("features must be finite and both folds non-empty")
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("training and test indices must not overlap")
    if np.any(train_indices < 0) or np.any(test_indices < 0) or np.any(train_indices >= len(features)) or np.any(test_indices >= len(features)):
        raise ValueError("fold indices are out of range")
    _, speaker_codes = np.unique(speakers, return_inverse=True)
    train_mask = np.zeros(len(features), dtype=bool)
    train_mask[train_indices] = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centered = _speaker_center_gpu(torch.as_tensor(features, device=device), speaker_codes, train_mask).cpu().numpy()
    return _fit_predict_gpu(centered[train_indices], labels[train_indices], centered[test_indices], k, c, gamma_factor)
