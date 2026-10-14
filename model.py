"""Sparse temporal refinement and interval-level process graph inference."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from config import BERGConfig


def shifted(values, offset):
    """Shift a batch of sequences without circular wraparound."""
    result = torch.zeros_like(values)
    length = values.shape[1]
    if abs(offset) >= length:
        return result
    if offset > 0:
        result[:, offset:] = values[:, :-offset]
    elif offset < 0:
        result[:, :offset] = values[:, -offset:]
    else:
        result = values
    return result


def neighbor_max(scores, valid, radius):
    """Maximum valid neighbor score, excluding the current position."""
    result = torch.zeros_like(scores)
    for offset in range(-radius, radius + 1):
        if offset:
            candidate = shifted(scores, offset) * shifted(valid, offset)
            result = torch.maximum(result, candidate)
    return result * valid


def select_steps(adjusted, valid, budget_ratio, threshold):
    """Stable thresholded top-K with a separate capacity for each sequence."""
    selected = torch.zeros_like(valid)
    for row in range(adjusted.shape[0]):
        capacity = math.ceil(budget_ratio * int(valid[row].sum().item()))
        candidates = torch.nonzero(valid[row] & (adjusted[row] >= threshold)).flatten()
        if capacity and candidates.numel():
            order = torch.argsort(adjusted[row, candidates], descending=True, stable=True)
            selected[row, candidates[order[:capacity]]] = True
    return selected


class DeepUpdate(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_dim
        self.query_norm = nn.LayerNorm(dim)
        self.anchor_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, config.num_heads,
                                             dropout=config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(4 * dim, dim))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, queries, anchors):
        context = self.anchor_norm(anchors)
        update, _ = self.attention(self.query_norm(queries), context, context, need_weights=False)
        queries = queries + self.dropout(update)
        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class BERG(nn.Module):
    """Offline temporal predictor operating on fixed visual feature sequences."""
    def __init__(self, config: BERGConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.projection = nn.Linear(config.input_dim, dim)
        self.light_conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.light_norm = nn.LayerNorm(dim)
        self.light_classifier = nn.Linear(dim, config.num_classes)
        self.value_mlp = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))
        self.deep = nn.ModuleList([DeepUpdate(config) for _ in range(config.deep_layers)])
        self.classifier = nn.Linear(dim, config.num_classes)
        self.boundary_head = nn.Linear(dim, 1)
        self.interval_query = nn.Linear(dim, dim, bias=False)
        self.interval_key = nn.Linear(dim, dim, bias=False)
        self.node_classifier = nn.Linear(dim, config.num_classes)

    def _check_input(self, features, valid):
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("features must have shape [batch, time, input_dim]")
        if features.shape[:2] != valid.shape or valid.dtype != torch.bool:
            raise ValueError("valid_mask must be a matching Boolean matrix")
        if features.shape[0] == 0 or features.shape[1] == 0 or not valid.any(dim=1).all():
            raise ValueError("Each sequence must contain at least one observed time step")
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Observed steps must form a prefix; split temporal gaps before batching")
        if not torch.isfinite(features[valid]).all():
            raise ValueError("Observed features must be finite")

    def _statistics(self, hidden, probabilities, valid):
        hidden = hidden.float()
        probabilities = probabilities.float()
        adjacent_distance = torch.zeros_like(valid, dtype=hidden.dtype)
        adjacent_prior = torch.zeros_like(adjacent_distance)
        adjacent_count = torch.zeros_like(adjacent_distance)
        local_sum = hidden.clone()
        local_count = valid.to(hidden.dtype)
        redundancy = torch.zeros_like(adjacent_distance)
        neighbor_count = torch.zeros_like(adjacent_distance)
        for offset in (-1, 1):
            mask = valid & shifted(valid, offset)
            similarity = F.cosine_similarity(hidden, shifted(hidden, offset), dim=-1)
            adjacent_distance += 0.5 * (1 - similarity) * mask
            adjacent_prior += 0.5 * (probabilities - shifted(probabilities, offset)).abs().sum(-1) * mask
            adjacent_count += mask
        for offset in range(-self.config.neighbor_radius, self.config.neighbor_radius + 1):
            if not offset:
                continue
            mask = valid & shifted(valid, offset)
            other = shifted(hidden, offset)
            local_sum += other * mask.unsqueeze(-1)
            local_count += mask
            redundancy += 0.5 * (1 + F.cosine_similarity(hidden, other, dim=-1)) * mask
            neighbor_count += mask
        mean = local_sum / local_count.clamp_min(1).unsqueeze(-1)
        deviation = 0.5 * (1 - F.cosine_similarity(hidden, mean, dim=-1))
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        entropy /= math.log(self.config.num_classes)
        redundancy = torch.where(neighbor_count > 0, redundancy / neighbor_count.clamp_min(1),
                                 torch.ones_like(redundancy))
        statistics = torch.stack((adjacent_distance / adjacent_count.clamp_min(1), deviation,
                                  entropy, 1 - redundancy,
                                  adjacent_prior / adjacent_count.clamp_min(1)), dim=-1)
        return statistics.clamp(0, 1) * valid.unsqueeze(-1)

    def _anchors(self, hidden, valid):
        """Exponential pooling over valid indices of each complete execution window."""
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device, dtype=torch.float32)
        centers = torch.linspace(0, 1, self.config.num_anchors, device=hidden.device,
                                 dtype=torch.float32)[None, :, None]
        centers = centers * (valid.sum(1).float() - 1)[:, None, None]
        logits = -(positions[None, None, :] - centers).abs() / self.config.anchor_sigma
        logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
        weights = logits.softmax(-1)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            anchors = weights @ hidden.float()
        return anchors.to(hidden.dtype), weights

    def forward(self, features, valid_mask):
        self._check_input(features, valid_mask)
        valid = valid_mask
        clean = features.masked_fill(~valid.unsqueeze(-1), 0)
        projected = self.projection(clean) * valid.unsqueeze(-1)
        light = self.light_norm(projected + F.gelu(self.light_conv(projected.transpose(1, 2)).transpose(1, 2)))
        light = light * valid.unsqueeze(-1)
        light_logits = self.light_classifier(light)
        probabilities = light_logits.float().softmax(-1)
        statistics = self._statistics(light, probabilities, valid)
        value_dtype = self.value_mlp[0].weight.dtype
        value_logits = self.value_mlp(statistics.to(value_dtype)).squeeze(-1).float()
        scores = value_logits.sigmoid() * valid
        adjusted = scores + self.config.neighbor_weight * neighbor_max(scores, valid, self.config.neighbor_radius)
        selected = select_steps(adjusted, valid, self.config.budget_ratio, self.config.selection_threshold)
        soft = ((adjusted - self.config.selection_threshold) / self.config.gate_temperature).sigmoid() * valid
        anchors, anchor_weights = self._anchors(light, valid)
        rows = []
        for row in range(features.shape[0]):
            positions = torch.nonzero(selected[row]).flatten()
            delta = torch.zeros_like(light[row])
            if positions.numel():
                refined = light[row, positions].unsqueeze(0)
                for layer in self.deep:
                    refined = layer(refined, anchors[row:row + 1])
                delta = delta.index_copy(0, positions, refined.squeeze(0) - light[row, positions])
            gate = selected[row].to(light.dtype)
            if self.training:
                gate = gate + (soft[row] - soft[row].detach())
            rows.append(light[row] + gate.unsqueeze(-1) * delta)
        fused = torch.stack(rows) * valid.unsqueeze(-1)
        return {"features": fused, "light_features": light, "light_logits": light_logits,
                "logits": self.classifier(fused), "boundary_logits": self.boundary_head(fused).squeeze(-1),
                "scores": scores, "value_logits": value_logits, "adjusted_scores": adjusted, "selected": selected,
                "soft_gates": soft, "anchors": anchors, "anchor_weights": anchor_weights,
                "valid_mask": valid, "executed_steps": selected.sum(1)}

    def decode_intervals(self, output, row):
        count = int(output["valid_mask"][row].sum().item())
        labels = output["logits"][row, :count].argmax(-1)
        boundaries = output["boundary_logits"][row, :count].float().sigmoid()
        starts = [0] + [t for t in range(1, count)
                        if labels[t] != labels[t - 1] or boundaries[t] >= self.config.boundary_threshold]
        return [{"start": a, "end": b, "label": int(labels[a].item())}
                for a, b in zip(starts, starts[1:] + [count])]

    def interval_node(self, output, row, start, end):
        """Use the same interval weights for classification and time-index selection."""
        hidden = output["features"][row, start:end]
        if not 0 <= start < end <= int(output["valid_mask"][row].sum().item()):
            raise ValueError("Interval endpoints must lie within the observed prefix")
        query = self.interval_query(hidden.mean(0))
        keys = self.interval_key(hidden)
        flags = output["selected"][row, start:end].float()
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            weights = (keys.float() @ query.float() / math.sqrt(self.config.hidden_dim)
                       + (flags + self.config.interval_floor).log()).softmax(0)
            pooled = weights @ hidden.float()
        logits = self.node_classifier(pooled.to(hidden.dtype))
        label = logits.argmax(-1)
        time_probabilities = output["logits"][row, start:end].float().softmax(-1)
        consistency = (weights * time_probabilities[:, label]).sum()
        return {"logits": logits, "weights": weights, "label": label,
                "consistency": consistency, "time_probabilities": time_probabilities}

    @torch.no_grad()
    def process_graph(self, output, row, timestamps, indices):
        nodes = []
        for interval in self.decode_intervals(output, row):
            start, end = interval["start"], interval["end"]
            node = self.interval_node(output, row, start, end)
            order = torch.argsort(node["weights"], descending=True, stable=True)
            pointers = start + order[:self.config.pointer_count]
            nodes.append({"start_index": int(indices[start]), "end_index": int(indices[end - 1]) + 1,
                          "start_time": float(timestamps[start]),
                          "end_time": float(timestamps[end - 1]) + self.config.step_seconds,
                          "label": int(node["label"]), "consistency": float(node["consistency"]),
                          "review": bool(node["consistency"] < self.config.review_threshold),
                          "time_indices": [int(indices[p]) for p in pointers],
                          "timestamps": [float(timestamps[p]) for p in pointers]})
        return {"nodes": nodes, "edges": [[i, i + 1] for i in range(len(nodes) - 1)]}
