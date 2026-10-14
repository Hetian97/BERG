"""Temporal segmentation metrics for independent, continuously labeled clips.

Scores use percentages in [0, 100]. Intervals use inclusive starts and exclusive
ends. Negative labels denote missing labels. ``SegmentationMeter`` separates
ground-truth gaps before scoring, so no segment or boundary can cross a gap.
Only NumPy and the Python standard library are required.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _labels(values: Any, name: str = "labels") -> np.ndarray:
    """Validate a one-dimensional sequence of integer class labels."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if array.dtype.kind not in "iubf":
        raise ValueError(f"{name} must contain integer class labels")
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{name} must contain finite integer class labels")
        if np.any(array < -(2.0 ** 63)) or np.any(array >= 2.0 ** 63):
            raise ValueError(f"{name} exceeds the int64 range")
    elif array.dtype.kind == "u" and np.any(array >= np.uint64(2 ** 63)):
        raise ValueError(f"{name} exceeds the int64 range")
    return array.astype(np.int64, copy=False)


def _paired_labels(pred_labels: Any, true_labels: Any) -> tuple[np.ndarray, np.ndarray]:
    pred = _labels(pred_labels, "pred_labels")
    true = _labels(true_labels, "true_labels")
    if pred.shape != true.shape:
        raise ValueError("pred_labels and true_labels must have equal lengths")
    return pred, true


def _positive_integer(value: Any, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _threshold(value: Any) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("IoU thresholds must be finite values in [0, 1]")
    return number


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 100.0 * (2 * tp) / denominator if denominator else 0.0


def _counts(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "f1": _f1(tp, fp, fn)}


def intervals_from_labels(
    labels: Any, boundaries: Any | None = None
) -> list[dict[str, int]]:
    """Convert class labels and optional event starts into labeled intervals.

    All nonnegative labels, including class zero, are retained. A negative label
    ends the current interval; equal labels on opposite sides of a gap remain
    separate intervals. Every class change splits an interval. When supplied,
    ``boundaries[t] > 0`` starts another interval at step t even if the class
    does not change. Ends are exclusive; a boundary at step zero has no effect.
    """
    values = _labels(labels)
    starts = np.zeros(len(values), dtype=bool)
    if boundaries is not None:
        boundary_values = np.asarray(boundaries)
        if boundary_values.ndim != 1 or boundary_values.shape != values.shape:
            raise ValueError("boundaries must be one-dimensional with the same length as labels")
        if boundary_values.dtype.kind not in "iubf" or not np.all(np.isfinite(boundary_values)):
            raise ValueError("boundaries must contain finite numeric values")
        starts = boundary_values > 0
    segments: list[dict[str, int]] = []
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or values[end] != values[start] or starts[end]:
            label = int(values[start])
            if label >= 0:
                segments.append({"start": start, "end": end, "label": label})
            start = end
    return segments


def labels_to_segments(labels: Any) -> list[dict[str, int]]:
    """Convert labels into intervals, keeping missing-label gaps separate."""
    return intervals_from_labels(labels)


def _validated_segments(
    segments: Sequence[dict[str, int]], length: int | None = None
) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for segment in segments:
        try:
            start = _positive_integer(segment["start"], "segment start", allow_zero=True)
            end = _positive_integer(segment["end"], "segment end")
            label = _positive_integer(segment["label"], "segment label", allow_zero=True)
        except (KeyError, TypeError) as error:
            raise ValueError("segments must contain start, end, and label integers") from error
        if end <= start:
            raise ValueError("each segment must have end greater than start")
        if length is not None and end > length:
            raise ValueError("segments must stay within the clip length")
        result.append({"start": start, "end": end, "label": label})
    result.sort(key=lambda segment: (segment["start"], segment["end"], segment["label"]))
    return result


def _segment_matches(
    pred: Sequence[dict[str, int]],
    true: Sequence[dict[str, int]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedily match same-class intervals in descending temporal IoU order."""
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred_segment in enumerate(pred):
        for true_index, true_segment in enumerate(true):
            if pred_segment["label"] != true_segment["label"]:
                continue
            intersection = max(
                0,
                min(pred_segment["end"], true_segment["end"])
                - max(pred_segment["start"], true_segment["start"]),
            )
            if not intersection:
                continue
            union = (
                pred_segment["end"] - pred_segment["start"]
                + true_segment["end"] - true_segment["start"]
                - intersection
            )
            iou = intersection / union
            if iou >= threshold:
                candidates.append((iou, pred_index, true_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_pred: set[int] = set()
    used_true: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pred_index, true_index in candidates:
        if pred_index not in used_pred and true_index not in used_true:
            used_pred.add(pred_index)
            used_true.add(true_index)
            matches.append((pred_index, true_index, iou))
    return matches


def interval_counts(
    pred_segments: Sequence[dict[str, int]],
    true_segments: Sequence[dict[str, int]],
    threshold: float,
) -> dict[str, int | float]:
    """Return same-class interval counts and percentage F1 at an IoU threshold.

    Explicit adjacent intervals may share a class; their separate starts are
    retained. Matching is one-to-one in descending IoU order. Duplicate or
    unmatched predicted intervals count as false positives.
    """
    pred = _validated_segments(pred_segments)
    true = _validated_segments(true_segments)
    threshold = _threshold(threshold)
    tp = len(_segment_matches(pred, true, threshold))
    return _counts(tp, len(pred) - tp, len(true) - tp)


def segment_f1(
    pred_labels: Any,
    true_labels: Any,
    thresholds: Sequence[float] = (0.1, 0.25, 0.5),
) -> dict[float, dict[str, int | float]]:
    """Return pooled segment counts and percentage F1 for each IoU threshold.

    Pairs require equal classes and positive temporal overlap. Matching is
    one-to-one, with descending IoU and then interval order breaking ties.
    Negative labels create gaps rather than segments. Two empty sequences have
    zero segment F1 because there are no observed segments.
    """
    pred, true = _paired_labels(pred_labels, true_labels)
    pred_segments = labels_to_segments(pred)
    true_segments = labels_to_segments(true)
    results: dict[float, dict[str, int | float]] = {}
    for value in thresholds:
        threshold = _threshold(value)
        results[threshold] = interval_counts(pred_segments, true_segments, threshold)
    return results


def _edit_from_segments(
    pred: Sequence[dict[str, int]], true: Sequence[dict[str, int]]
) -> float:
    pred_actions = [segment["label"] for segment in pred]
    true_actions = [segment["label"] for segment in true]
    length = max(len(pred_actions), len(true_actions))
    if not length:
        return 100.0
    previous = list(range(len(true_actions) + 1))
    for pred_index, pred_action in enumerate(pred_actions, 1):
        current = [pred_index]
        for true_index, true_action in enumerate(true_actions, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[true_index] + 1,
                    previous[true_index - 1] + (pred_action != true_action),
                )
            )
        previous = current
    return 100.0 * (1.0 - previous[-1] / length)


def edit_score(pred_labels: Any, true_labels: Any) -> float:
    """Return normalized Levenshtein similarity of segment-label sequences.

    Consecutive equal class labels collapse to one action, but equal actions
    separated by negative labels remain separate. Missing labels are excluded.
    The score is ``100 * (1 - distance / max_length)``. Two empty action
    sequences receive 100; the meter skips clips with no labeled observations.
    """
    pred, true = _paired_labels(pred_labels, true_labels)
    return _edit_from_segments(labels_to_segments(pred), labels_to_segments(true))


def _class_counts(
    pred: np.ndarray, true: np.ndarray, num_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = true >= 0
    observed_true = true[valid]
    observed_pred = pred[valid]
    if np.any(observed_true >= num_classes):
        raise ValueError("true_labels contain a class outside num_classes")
    if np.any(observed_pred >= num_classes):
        raise ValueError("pred_labels contain a class outside num_classes")
    predicted = observed_pred >= 0
    true_support = np.bincount(observed_true, minlength=num_classes)
    pred_support = np.bincount(observed_pred[predicted], minlength=num_classes)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    if np.any(predicted):
        joint = observed_true[predicted] * num_classes + observed_pred[predicted]
        confusion = np.bincount(joint, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
    return confusion, true_support, pred_support


def _macro_from_counts(
    confusion: np.ndarray, true_support: np.ndarray, pred_support: np.ndarray
) -> float:
    denominator = true_support + pred_support
    scores = np.zeros(len(denominator), dtype=np.float64)
    np.divide(
        2.0 * np.diag(confusion), denominator, out=scores, where=denominator > 0
    )
    return float(100.0 * scores.mean())


def macro_f1(pred_labels: Any, true_labels: Any, num_classes: int) -> float:
    """Return percentage Macro-F1 over every declared class.

    Classes absent from both prediction and truth have F1 zero and remain in the
    average. Ground-truth negative positions are ignored. Negative predictions
    at labeled positions count as missed labels, so they reduce true-class F1.
    """
    classes = _positive_integer(num_classes, "num_classes")
    pred, true = _paired_labels(pred_labels, true_labels)
    return _macro_from_counts(*_class_counts(pred, true, classes))


def _boundaries(labels: np.ndarray) -> np.ndarray:
    if len(labels) < 2:
        return np.empty(0, dtype=np.int64)
    changes = (labels[1:] != labels[:-1]) & (labels[1:] >= 0) & (labels[:-1] >= 0)
    return np.flatnonzero(changes) + 1


def _boundary_counts(
    pred_points: Sequence[int], true_points: Sequence[int], tolerance: int
) -> dict[str, int | float]:
    pred_index = true_index = tp = 0
    while pred_index < len(pred_points) and true_index < len(true_points):
        difference = int(pred_points[pred_index] - true_points[true_index])
        if difference < -tolerance:
            pred_index += 1
        elif difference > tolerance:
            true_index += 1
        else:
            tp += 1
            pred_index += 1
            true_index += 1
    return _counts(tp, len(pred_points) - tp, len(true_points) - tp)


def boundary_f1(
    pred_labels: Any, true_labels: Any, tolerance: int = 1
) -> dict[str, int | float]:
    """Return boundary counts and percentage F1 with a time-step tolerance.

    Boundaries are class transitions at interval starts. Clip starts and starts
    after missing labels are excluded. Two pointers on sorted boundary indices
    produce a maximum-cardinality one-to-one match within the tolerance.
    No observed boundaries produce zero F1, including constant identical clips.
    """
    tolerance = _positive_integer(tolerance, "tolerance", allow_zero=True)
    pred, true = _paired_labels(pred_labels, true_labels)
    return _boundary_counts(_boundaries(pred), _boundaries(true), tolerance)


def _short_from_segments(
    pred_segments: Sequence[dict[str, int]],
    true_segments: Sequence[dict[str, int]],
    duration: int,
    threshold: float,
) -> dict[str, int | float]:
    short_indices = {
        index
        for index, segment in enumerate(true_segments)
        if segment["end"] - segment["start"] <= duration
    }
    matches = _segment_matches(pred_segments, true_segments, threshold)
    tp = sum(true_index in short_indices for _, true_index, _ in matches)
    total = len(short_indices)
    return {"tp": tp, "total": total, "recall": 100.0 * tp / total if total else 0.0}


def short_recall(
    pred_labels: Any,
    true_labels: Any,
    max_duration_steps: int = 3,
    iou_threshold: float = 0.5,
) -> dict[str, int | float]:
    """Return matched short-event count, total count, and percentage recall.

    A short event is a ground-truth interval with duration at most the declared
    number of steps. Same-class matching uses all intervals, descending IoU,
    and one-to-one assignment before counting matched short events. No short
    events produce zero recall.
    """
    duration = _positive_integer(max_duration_steps, "max_duration_steps")
    threshold = _threshold(iou_threshold)
    pred, true = _paired_labels(pred_labels, true_labels)
    pred_segments = labels_to_segments(pred)
    true_segments = labels_to_segments(true)
    return _short_from_segments(pred_segments, true_segments, duration, threshold)


def _valid_runs(true: np.ndarray) -> list[tuple[int, int]]:
    valid = true >= 0
    padded = np.pad(valid.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _clip_segments(
    segments: Sequence[dict[str, int]], start: int, end: int
) -> list[dict[str, int]]:
    """Intersect clip-coordinate intervals with one run and use run coordinates."""
    clipped = []
    for segment in segments:
        clipped_start = max(start, segment["start"])
        clipped_end = min(end, segment["end"])
        if clipped_start < clipped_end:
            clipped.append(
                {
                    "start": clipped_start - start,
                    "end": clipped_end - start,
                    "label": segment["label"],
                }
            )
    return clipped


def _segment_boundaries(segments: Sequence[dict[str, int]]) -> list[int]:
    """Return distinct event starts except the first time step of a run."""
    return sorted({segment["start"] for segment in segments if segment["start"] > 0})


class SegmentationMeter:
    """Accumulate metrics without joining videos or ground-truth gaps.

    Segment, boundary, and short-event counts are pooled across labeled runs.
    Edit is the unweighted mean of per-run normalized similarities. Macro-F1
    pools labeled time steps across runs and includes all declared classes,
    assigning zero to absent classes. Computed scores are percentages.
    """

    thresholds = (0.1, 0.25, 0.5)

    def __init__(
        self,
        num_classes: int,
        short_duration_steps: int = 3,
        boundary_tolerance: int = 1,
    ) -> None:
        self.num_classes = _positive_integer(num_classes, "num_classes")
        self.short_duration_steps = _positive_integer(
            short_duration_steps, "short_duration_steps"
        )
        self.boundary_tolerance = _positive_integer(
            boundary_tolerance, "boundary_tolerance", allow_zero=True
        )
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.true_support = np.zeros(self.num_classes, dtype=np.int64)
        self.pred_support = np.zeros(self.num_classes, dtype=np.int64)
        self.segment_counts = {
            threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in self.thresholds
        }
        self.boundary_counts = {"tp": 0, "fp": 0, "fn": 0}
        self.short_counts = {"tp": 0, "total": 0}
        self.edit_sum = 0.0
        self.num_runs = 0
        self.num_observations = 0

    def update(
        self,
        pred_labels: Any,
        true_labels: Any,
        pred_segments: Sequence[dict[str, int]] | None = None,
        true_boundaries: Any | None = None,
    ) -> None:
        """Add one clip, optionally retaining explicit same-class event splits.

        Predicted intervals use coordinates relative to this input clip. They
        are clipped to each labeled truth run and shifted to its coordinates.
        Event metrics use these intervals and truth boundary starts; Macro-F1
        uses the dense labels. If intervals are omitted, class changes define
        the predicted events. Ground-truth gaps split independent scoring runs.
        """
        pred, true = _paired_labels(pred_labels, true_labels)
        true_intervals = intervals_from_labels(true, true_boundaries)
        pred_intervals = (
            intervals_from_labels(pred)
            if pred_segments is None
            else _validated_segments(pred_segments, length=len(pred))
        )
        runs = _valid_runs(true)
        run_intervals = [
            (start, end, _clip_segments(pred_intervals, start, end),
             _clip_segments(true_intervals, start, end))
            for start, end in runs
        ]
        if any(
            segment["label"] >= self.num_classes
            for _, _, pred_run, _ in run_intervals
            for segment in pred_run
        ):
            raise ValueError("pred_segments contain a class outside num_classes")
        confusion, true_support, pred_support = _class_counts(
            pred, true, self.num_classes
        )
        self.confusion += confusion
        self.true_support += true_support
        self.pred_support += pred_support
        for start, end, pred_run, true_run in run_intervals:
            for threshold in self.thresholds:
                counts = interval_counts(pred_run, true_run, threshold)
                for key in ("tp", "fp", "fn"):
                    self.segment_counts[threshold][key] += int(counts[key])
            boundaries = _boundary_counts(
                _segment_boundaries(pred_run),
                _segment_boundaries(true_run),
                self.boundary_tolerance,
            )
            for key in ("tp", "fp", "fn"):
                self.boundary_counts[key] += int(boundaries[key])
            short = _short_from_segments(
                pred_run, true_run, self.short_duration_steps, 0.5
            )
            for key in ("tp", "total"):
                self.short_counts[key] += int(short[key])
            self.edit_sum += _edit_from_segments(pred_run, true_run)
            self.num_runs += 1
            self.num_observations += end - start

    def compute(self) -> dict[str, float]:
        """Return the seven named metric families as percentage scores.

        A meter without labeled time steps returns zero for every metric.
        Counts with no events also return zero, preventing empty observations
        from being reported as perfect segmentation or boundary detection.
        """
        result = {
            "Macro-F1": 0.0,
            "F1@10": 0.0,
            "F1@25": 0.0,
            "F1@50": 0.0,
            "Edit": 0.0,
            "Boundary-F1": 0.0,
            "ShortRecall": 0.0,
        }
        if not self.num_observations:
            return result
        result["Macro-F1"] = _macro_from_counts(
            self.confusion, self.true_support, self.pred_support
        )
        for threshold, name in zip(self.thresholds, ("F1@10", "F1@25", "F1@50")):
            result[name] = _f1(**self.segment_counts[threshold])
        result["Edit"] = self.edit_sum / self.num_runs
        result["Boundary-F1"] = _f1(**self.boundary_counts)
        total = self.short_counts["total"]
        result["ShortRecall"] = 100.0 * self.short_counts["tp"] / total if total else 0.0
        return result


__all__ = [
    "intervals_from_labels",
    "labels_to_segments",
    "interval_counts",
    "segment_f1",
    "edit_score",
    "macro_f1",
    "boundary_f1",
    "short_recall",
    "SegmentationMeter",
]
