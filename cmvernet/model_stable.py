"""V45 salience-stratified latent Perceiver residual for V43 SER evidence.

Core latent-cross-attention variant extracted from the original experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model_ensemble import (
    EfficientEnsembleLinear,
    IndependentEnsembleHead,
)


VARIANTS = ("latent-cross-attention",)


@dataclass(frozen=True)
class V45Config:
    base_dimension: int = 3072
    core_dimension: int = 1024
    hidden_dimensions: tuple[int, int] = (512, 128)
    classes: int = 6
    members: int = 8
    dropout: float = 0.10
    modulation_strength: float = 0.50
    additive_bypass: float = 0.25
    salience_tokens: int = 12
    coordinates_per_token: int = 256
    token_dimension: int = 96
    latent_tokens: int = 4
    attention_heads: int = 4
    feedforward_dimension: int = 192
    implementation_version: str = "v45-sspr-1.0"

    def validate(self) -> None:
        if self.salience_tokens * self.coordinates_per_token != self.base_dimension:
            raise ValueError("V45 salience tokenization must cover all selected coordinates")
        if self.token_dimension % self.attention_heads:
            raise ValueError("V45 token dimension must be divisible by attention heads")
        if self.members < 2 or self.classes < 2 or self.latent_tokens < 1:
            raise ValueError("V45 ensemble, classes, and latent count must be valid")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("V45 dropout must be in [0, 1)")


class V45SalienceStratifiedPerceiver(nn.Module):
    """V43 plus a zero-initialized latent correction over salience strata."""

    def __init__(
        self,
        *,
        variant: str,
        config: V45Config = V45Config(),
    ) -> None:
        super().__init__()
        config.validate()
        if variant not in VARIANTS:
            raise ValueError(f"unknown V45 variant: {variant}")
        self.variant = variant
        self.config = config
        first, second = config.hidden_dimensions

        # V43 backbone, built first to preserve its seeded initialization.
        self.base_first = EfficientEnsembleLinear(
            config.base_dimension, first, config.members, first_layer_random_signs=True
        )
        self.core_first = EfficientEnsembleLinear(
            config.core_dimension, first, config.members, first_layer_random_signs=True
        )
        self.fusion_normalization = nn.LayerNorm(first)
        self.second = EfficientEnsembleLinear(
            first, second, config.members, first_layer_random_signs=False
        )
        self.output = IndependentEnsembleHead(second, config.classes, config.members)
        self.dropout = nn.Dropout(config.dropout)

        # Salience strata are tokens, not acoustic time frames.  The same local
        # projector is used across ranks, with learned rank embeddings.
        self.token_projection = nn.Linear(
            config.coordinates_per_token, config.token_dimension
        )
        self.salience_position = nn.Parameter(
            torch.empty(config.salience_tokens, config.token_dimension)
        )
        self.latent_queries = nn.Parameter(
            torch.empty(config.latent_tokens, config.token_dimension)
        )
        self.cross_attention = nn.MultiheadAttention(
            config.token_dimension,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_normalization = nn.LayerNorm(config.token_dimension)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.correction_normalization = nn.LayerNorm(config.token_dimension)
        self.correction_head = nn.Linear(config.token_dimension, config.classes)
        self.reset_residual_parameters()

    def reset_residual_parameters(self) -> None:
        nn.init.trunc_normal_(self.salience_position, std=0.02)
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        # Exact V43 initial function while preserving non-zero gradients for
        # the correction head on the first update.
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def _v43_member_logits(self, base: Tensor, core: Tensor) -> Tensor:
        base_members = base[:, None, :].expand(-1, self.config.members, -1)
        core_members = core[:, None, :].expand(-1, self.config.members, -1)
        base_evidence = self.base_first(base_members)
        core_evidence = self.core_first(core_members)
        stable = F.relu(base_evidence)
        quotient = torch.tanh(core_evidence)
        fused = stable * (1.0 + self.config.modulation_strength * quotient)
        fused = fused + self.config.additive_bypass * F.relu(core_evidence)
        hidden = self.dropout(self.fusion_normalization(fused))
        hidden = self.dropout(F.relu(self.second(hidden)))
        return self.output(hidden)

    def _correction(self, base: Tensor) -> Tensor:
        batch = len(base)
        tokens = base.reshape(
            batch,
            self.config.salience_tokens,
            self.config.coordinates_per_token,
        )
        tokens = self.token_projection(tokens) + self.salience_position[None, :, :]
        latent = self.latent_queries[None, :, :].expand(batch, -1, -1)
        attended, _ = self.cross_attention(
            latent, tokens, tokens, need_weights=False
        )
        latent = self.cross_normalization(latent + attended)
        latent = self.latent_encoder(latent)
        pooled = self.correction_normalization(latent.mean(dim=1))
        return self.correction_head(pooled)

    def forward(self, base: Tensor, core: Tensor) -> Mapping[str, Tensor]:
        if base.ndim != 2 or base.shape[1] != self.config.base_dimension:
            raise ValueError("V45 base input shape is invalid")
        if core.ndim != 2 or core.shape[1] != self.config.core_dimension:
            raise ValueError("V45 core input shape is invalid")
        if len(base) != len(core):
            raise ValueError("V45 input rows do not align")
        if not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(core).all()):
            raise ValueError("V45 inputs must be finite")
        member_logits = self._v43_member_logits(base, core)
        correction = self._correction(base)
        member_logits = member_logits + correction[:, None, :]
        member_probability = F.softmax(member_logits, dim=-1)
        ensemble_probability = member_probability.mean(dim=1)
        return {
            "member_logits": member_logits,
            "member_probability": member_probability,
            "ensemble_probability": ensemble_probability,
            "correction_logits": correction,
            "logits": torch.log(ensemble_probability.clamp_min(1e-8)),
        }

    def loss(
        self,
        output: Mapping[str, Tensor],
        labels: Tensor,
        *,
        label_smoothing: float,
    ) -> Tensor:
        logits = output["member_logits"]
        repeated = labels[:, None].expand(-1, self.config.members).reshape(-1)
        return F.cross_entropy(
            logits.reshape(-1, self.config.classes),
            repeated,
            label_smoothing=float(label_smoothing),
        )

    def metadata(self) -> Mapping[str, object]:
        return {
            "name": "V45 Salience-Stratified Latent Perceiver Residual",
            "variant": self.variant,
            "configuration": asdict(self.config),
            "trainable_parameters": sum(int(p.numel()) for p in self.parameters()),
            "v43_exact_initial_function": True,
            "salience_rank_tokens_not_acoustic_time": True,
            "latent_cross_attention": self.variant == "latent-cross-attention",
            "zero_initialized_logit_residual": True,
            "contains_cnn": False,
            "contains_acoustic_time_rnn": False,
            "contains_bayesian_layer": False,
            "contains_graph_or_hypergraph": False,
        }


__all__ = ["VARIANTS", "V45Config", "V45SalienceStratifiedPerceiver"]
