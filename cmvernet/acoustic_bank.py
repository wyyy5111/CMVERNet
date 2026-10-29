"""Validated adapters for the unified V1 and V9 acoustic feature banks.

The project stores the expensive real-WAV feature families in component
caches.  This module defines the only supported concatenation rules used by
the new neural experiments.  It never fabricates, pads, imputes, or replaces a
missing/non-finite component: malformed evidence fails closed.

``v9`` is the audited 13,632-dimensional bank.  ``v1`` reproduces the original
3,902-dimensional definition from the same real-WAV caches:

* affective: the first 2,403 coordinates of ``source_affsig``;
* GFWP: coordinates 2,500:3,700 of ``gsed_gfwp_cspt``;
* temporal dynamics: all 146 coordinates;
* CEA: all 153 coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AcousticBlock:
    """One contiguous, semantically named block in a feature bank."""

    name: str
    dimension: int

    def validate(self) -> None:
        if not self.name or int(self.dimension) < 1:
            raise ValueError("acoustic block name/dimension is invalid")


V9_BLOCKS: tuple[AcousticBlock, ...] = (
    AcousticBlock("source_affsig", 5_636),
    AcousticBlock("gsed_gfwp_cspt", 4_800),
    AcousticBlock("temporal_dynamics", 146),
    AcousticBlock("gammatone_auditory", 231),
    AcousticBlock("cea", 153),
    AcousticBlock("pcf", 97),
    AcousticBlock("epac", 880),
    AcousticBlock("lemstcd", 1_689),
)

V1_BLOCKS: tuple[AcousticBlock, ...] = (
    AcousticBlock("affective", 2_403),
    AcousticBlock("gfwp", 1_200),
    AcousticBlock("temporal_dynamics", 146),
    AcousticBlock("cea", 153),
)

COMPONENT_DIMENSIONS: Mapping[str, int] = {
    "source_affsig": 5_636,
    "gsed_gfwp_cspt": 4_800,
    "temporal_dynamics": 146,
    "gammatone_auditory": 231,
    "cea": 153,
    "pcf": 97,
    "epac": 880,
    "lemstcd": 1_689,
}


def bank_blocks(bank: str) -> tuple[AcousticBlock, ...]:
    """Return the immutable layout for ``v1`` or ``v9``."""

    name = str(bank).strip().lower()
    if name == "v9":
        return V9_BLOCKS
    if name == "v1":
        return V1_BLOCKS
    raise ValueError("feature bank must be 'v1' or 'v9'")


def bank_dimension(bank: str) -> int:
    return int(sum(block.dimension for block in bank_blocks(bank)))


def _validated_component(
    components: Mapping[str, np.ndarray], name: str
) -> np.ndarray:
    if name not in components:
        raise KeyError(f"missing real acoustic component: {name}")
    vector = np.asarray(components[name], dtype=np.float32).reshape(-1)
    expected = int(COMPONENT_DIMENSIONS[name])
    if vector.shape != (expected,):
        raise ValueError(
            f"{name}: component dimension {vector.shape} differs from {(expected,)}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name}: non-finite real acoustic component")
    return vector


def assemble_acoustic_bank(
    components: Mapping[str, np.ndarray], bank: str
) -> np.ndarray:
    """Assemble one utterance without any fallback or synthetic coordinates."""

    name = str(bank).strip().lower()
    source_affsig = _validated_component(components, "source_affsig")
    gsed_gfwp_cspt = _validated_component(components, "gsed_gfwp_cspt")
    temporal = _validated_component(components, "temporal_dynamics")
    cea = _validated_component(components, "cea")

    if name == "v1":
        parts: Sequence[np.ndarray] = (
            source_affsig[:2_403],
            gsed_gfwp_cspt[2_500:3_700],
            temporal,
            cea,
        )
    elif name == "v9":
        parts = (
            source_affsig,
            gsed_gfwp_cspt,
            temporal,
            _validated_component(components, "gammatone_auditory"),
            cea,
            _validated_component(components, "pcf"),
            _validated_component(components, "epac"),
            _validated_component(components, "lemstcd"),
        )
    else:
        raise ValueError("feature bank must be 'v1' or 'v9'")

    result = np.concatenate(parts).astype(np.float32, copy=False)
    expected = bank_dimension(name)
    if result.shape != (expected,) or not np.all(np.isfinite(result)):
        raise RuntimeError(
            f"{name}: assembled acoustic bank is invalid: {result.shape}"
        )
    return result


def block_slices(bank: str) -> tuple[slice, ...]:
    """Return contiguous block slices in the exact concatenation order."""

    result: list[slice] = []
    cursor = 0
    for block in bank_blocks(bank):
        block.validate()
        result.append(slice(cursor, cursor + block.dimension))
        cursor += block.dimension
    if cursor != bank_dimension(bank):
        raise RuntimeError("acoustic bank block layout is internally inconsistent")
    return tuple(result)


__all__ = [
    "AcousticBlock",
    "COMPONENT_DIMENSIONS",
    "V1_BLOCKS",
    "V9_BLOCKS",
    "assemble_acoustic_bank",
    "bank_blocks",
    "bank_dimension",
    "block_slices",
]
