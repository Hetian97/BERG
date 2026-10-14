"""Validated configuration for budget-constrained temporal refinement."""

from dataclasses import asdict, dataclass, fields
import math
import numbers


@dataclass(frozen=True)
class BERGConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int = 256
    num_anchors: int = 16
    deep_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    budget_ratio: float = 0.35
    neighbor_weight: float = 0.30
    neighbor_radius: int = 2
    selection_threshold: float = 0.5
    gate_temperature: float = 0.1
    anchor_sigma: float = 16.0
    interval_floor: float = 0.1
    pointer_count: int = 3
    review_threshold: float = 0.65
    boundary_threshold: float = 0.5
    step_seconds: float = 1.0
    boundary_weight: float = 1.0
    value_weight: float = 1.0
    budget_weight: float = 1.0
    coverage_weight: float = 1.0
    graph_weight: float = 1.0
    graph_consistency_weight: float = 0.1

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is int:
                minimum = 0 if field.name == "neighbor_radius" else 1
                if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
                    raise ValueError(f"{field.name} must be an integer of at least {minimum}")
                object.__setattr__(self, field.name, int(value))
            else:
                if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                    raise ValueError(f"{field.name} must be a finite real number")
                object.__setattr__(self, field.name, float(value))
        if self.num_classes < 2 or self.hidden_dim % self.num_heads:
            raise ValueError("At least two classes and divisible attention dimensions are required")
        for name in ("budget_ratio", "review_threshold", "boundary_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        for name in ("gate_temperature", "anchor_sigma", "interval_floor", "step_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("neighbor_weight", "selection_threshold", "boundary_weight",
                     "value_weight", "budget_weight", "coverage_weight", "graph_weight",
                     "graph_consistency_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)
