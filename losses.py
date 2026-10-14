"""Joint segmentation, value, budget, coverage, and interval supervision."""

import torch
from torch.nn import functional as F
from metrics import intervals_from_labels
from model import shifted


def position_targets(light_logits, labels, boundaries, valid):
    """Detached classification errors combined with supervised boundary indicators."""
    probabilities = light_logits.float().softmax(-1)
    safe = labels.masked_fill(~valid, 0).clamp(0, light_logits.shape[-1] - 1)
    error = 1 - probabilities.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    target = 1 - (1 - boundaries) * (1 - error.detach())
    return target * valid


def coverage_loss(soft_gates, boundaries, valid, radius):
    active = valid & (boundaries > 0)
    if not active.any():
        return soft_gates.sum() * 0
    gates = soft_gates.float().clamp(max=1 - 1e-7)
    log_uncovered = torch.zeros_like(gates)
    for offset in range(-radius, radius + 1):
        neighbor = shifted(gates, offset)
        mask = valid & shifted(valid, offset)
        log_uncovered += torch.log1p(-neighbor) * mask
    probability = -torch.expm1(log_uncovered)
    return -probability[active].clamp_min(1e-8).log().mean()


def joint_loss(model, output, labels, boundaries):
    """Apply supervision exclusively at observed, labelled positions."""
    config = model.config
    if labels.shape != output["valid_mask"].shape or boundaries.shape != labels.shape:
        raise ValueError("Supervision must match the model timeline")
    if labels.dtype != torch.long:
        raise ValueError("Labels must use torch.int64")
    if labels.device != output["logits"].device or boundaries.device != labels.device:
        raise ValueError("Supervision and predictions must share a device")
    observed = output["valid_mask"]
    if (labels[observed] < -1).any():
        raise ValueError("Unknown labels must use -1")
    if not torch.isfinite(boundaries[observed]).all() or not ((boundaries[observed] == 0) | (boundaries[observed] == 1)).all():
        raise ValueError("Observed boundaries must be finite and binary")
    valid = output["valid_mask"] & (labels >= 0)
    boundaries = boundaries.float().clone()
    boundaries[:, 0] = 0
    changes = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    boundaries[:, 1:] = torch.maximum(boundaries[:, 1:], changes.float())
    if (labels[valid] >= config.num_classes).any():
        raise ValueError("Class IDs must lie within the configured class vocabulary")
    zero = output["logits"].sum() * 0
    if valid.any():
        segmentation = F.cross_entropy(output["logits"][valid], labels[valid])
        segmentation += F.cross_entropy(output["light_logits"][valid], labels[valid])
        boundary = F.binary_cross_entropy_with_logits(output["boundary_logits"][valid], boundaries[valid])
        targets = position_targets(output["light_logits"], labels, boundaries, valid)
        value = F.binary_cross_entropy_with_logits(output["value_logits"][valid].float(), targets[valid].float())
    else:
        segmentation = boundary = value = zero
    counts = output["valid_mask"].sum(1).clamp_min(1)
    activation = output["soft_gates"].sum(1) / counts
    budget = (activation - config.budget_ratio).clamp_min(0).square().mean()
    coverage = coverage_loss(output["soft_gates"], boundaries, valid, config.neighbor_radius)
    graph_terms = []
    for row in range(labels.shape[0]):
        count = int(output["valid_mask"][row].sum().item())
        intervals = intervals_from_labels(labels[row, :count].detach().cpu().numpy(),
                                          boundaries[row, :count].detach().cpu().numpy())
        for interval in intervals:
            node = model.interval_node(output, row, interval["start"], interval["end"])
            truth = interval["label"]
            target = labels.new_tensor([truth])
            term = F.cross_entropy(node["logits"].unsqueeze(0), target)
            agreement = (node["weights"] * node["time_probabilities"][:, truth]).sum()
            graph_terms.append(term + config.graph_consistency_weight * (1 - agreement))
    graph = torch.stack(graph_terms).mean() if graph_terms else zero
    total = (segmentation + config.boundary_weight * boundary + config.value_weight * value
             + config.budget_weight * budget + config.coverage_weight * coverage
             + config.graph_weight * graph)
    return {"total": total, "segmentation": segmentation, "boundary": boundary,
            "value": value, "budget": budget, "coverage": coverage, "graph": graph}
