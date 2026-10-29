"""Pinned WavLM extraction routines from the original experiment code."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from . import config as cfg
from .acoustic_core import DEFAULT_CONFIG, load_waveform_views

def _load_frozen_encoder(device: torch.device):
    from transformers import AutoFeatureExtractor, WavLMModel
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        source = snapshot_download(
            repo_id=cfg.WAVLM_MODEL_ID, revision=cfg.WAVLM_REVISION, local_files_only=True
        )
        common: dict[str, object] = {"local_files_only": True}
    except LocalEntryNotFoundError:
        print(
            "Pinned WavLM revision not in the local HF cache; downloading the "
            "exact audited revision now.",
            flush=True,
        )
        source = cfg.WAVLM_MODEL_ID
        common = {"revision": cfg.WAVLM_REVISION}
    processor = AutoFeatureExtractor.from_pretrained(source, **common)
    model = WavLMModel.from_pretrained(source, use_safetensors=False, **common)
    if getattr(model.config, "_commit_hash", None) != cfg.WAVLM_REVISION:
        raise RuntimeError(
            f"unexpected WavLM revision: {getattr(model.config, '_commit_hash', None)!r}"
        )
    return processor, model.eval().to(device)


def _pool_hidden_states(model, hidden_states: Sequence[torch.Tensor], attention_mask: torch.Tensor) -> np.ndarray:
    hidden_length = int(hidden_states[0].shape[1])
    feature_mask = model._get_feature_vector_attention_mask(
        hidden_length, attention_mask
    ).to(hidden_states[0].dtype)
    weight = feature_mask.unsqueeze(-1)
    denominator = torch.clamp(weight.sum(dim=1), min=1.0)
    pooled_layers = []
    for hidden in hidden_states:
        mean = (hidden * weight).sum(dim=1) / denominator
        centered = hidden - mean.unsqueeze(1)
        variance = (centered.square() * weight).sum(dim=1) / denominator
        std = torch.sqrt(torch.clamp(variance, min=1e-8))
        pooled_layers.append(torch.cat([mean, std], dim=1))
    pooled = torch.stack(pooled_layers, dim=1)
    return pooled.detach().to("cpu", dtype=torch.float32).numpy()


def extract_batch(records, processor, model, device: torch.device):
    waveforms = []
    for record in records:
        path = Path(str(record["path"]))
        views = load_waveform_views(path, DEFAULT_CONFIG)
        waveforms.append(
            {name: np.asarray(views[name], dtype=np.float32) for name in cfg.WAVEFORM_VIEW_NAMES}
        )
    output: dict[str, list[np.ndarray]] = {name: [] for name in cfg.WAVEFORM_VIEW_NAMES}
    for name in cfg.WAVEFORM_VIEW_NAMES:
        encoded = processor(
            [views[name] for views in waveforms],
            sampling_rate=DEFAULT_CONFIG.sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values = encoded.input_values.to(device)
        attention_mask = encoded.attention_mask.to(device)
        # float32 计算 + TF32 张量核（无 bf16 autocast）：提取出的 embedding
        # 数值与审计版 float32 路径几乎一致，避免 bf16 舍入造成的特征漂移。
        with torch.inference_mode():
            result = model(input_values, attention_mask=attention_mask, output_hidden_states=True)
        pooled = _pool_hidden_states(model, result.hidden_states, attention_mask)
        if pooled.shape != (len(records), cfg.WAVLM_LAYER_COUNT, cfg.WAVLM_LAYER_DIMENSION):
            raise RuntimeError(f"unexpected batched WavLM output: {pooled.shape}")
        output[name] = [np.asarray(row, dtype=np.float32) for row in pooled]
    return [
        {name: output[name][index] for name in cfg.WAVEFORM_VIEW_NAMES}
        for index in range(len(records))
    ]
