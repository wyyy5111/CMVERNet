"""Shared ensemble layers used by the stable expert.

Extracted from the original model_ensemble implementation.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn

class EfficientEnsembleLinear(nn.Module):
    """Shared dense map with member-specific rank-one factors and biases."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        members: int,
        *,
        first_layer_random_signs: bool,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.members = int(members)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.r = nn.Parameter(torch.empty(members, in_features))
        self.s = nn.Parameter(torch.empty(members, out_features))
        self.bias = nn.Parameter(torch.empty(members, out_features))
        self.first_layer_random_signs = bool(first_layer_random_signs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.first_layer_random_signs:
            with torch.no_grad():
                self.r.bernoulli_(0.5).mul_(2.0).sub_(1.0)
        else:
            nn.init.ones_(self.r)
        nn.init.ones_(self.s)
        bound = self.in_features**-0.5
        initial = torch.empty(
            self.out_features,
            dtype=self.weight.dtype,
            device=self.weight.device,
        ).uniform_(-bound, bound)
        with torch.no_grad():
            self.bias.copy_(initial.expand(self.members, -1))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.members, self.in_features):
            raise ValueError(
                "efficient ensemble input must have shape "
                f"B x {self.members} x {self.in_features}"
            )
        return (x * self.r) @ self.weight.T * self.s + self.bias


class IndependentEnsembleHead(nn.Module):
    def __init__(self, in_features: int, classes: int, members: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(members, classes, in_features))
        self.bias = nn.Parameter(torch.empty(members, classes))
        self.in_features = int(in_features)
        self.classes = int(classes)
        self.members = int(members)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = self.in_features**-0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.members, self.in_features):
            raise ValueError("independent head input shape is invalid")
        return torch.einsum("bkh,kch->bkc", x, self.weight) + self.bias
