"""Validated feature sequences, split isolation, and variable-length batches."""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path, PureWindowsPath
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetError(ValueError):
    """A manifest or feature sequence violates the data contract."""


def _class_count(value: int | None) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 2
    ):
        raise DatasetError("num_classes must be an integer of at least two.")
    return None if value is None else int(value)


def _step_size(step_seconds: float) -> float:
    if isinstance(step_seconds, bool):
        raise DatasetError("step_seconds must be finite and positive.")
    try:
        value = float(step_seconds)
    except (TypeError, ValueError) as exc:
        raise DatasetError("step_seconds must be finite and positive.") from exc
    if not math.isfinite(value) or value <= 0:
        raise DatasetError("step_seconds must be finite and positive.")
    return value


def read_manifest(path: str | Path, num_classes: int | None = None) -> dict[str, Any]:
    """Read a schema-v1 manifest and reject teacher or video split leakage.

    Each record contains path, split, teacher_id, and video_id. NPZ paths must
    be relative to the manifest directory; returned record paths are absolute.
    """
    expected_classes = _class_count(num_classes)
    manifest_path = Path(path).expanduser().resolve()
    try:
        with manifest_path.open("r", encoding="utf-8-sig") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"Cannot read manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
        raise DatasetError("The manifest must be an object with schema_version=1.")
    if manifest["schema_version"] != 1:
        raise DatasetError("Only manifest schema_version=1 is supported.")
    names = manifest.get("class_names")
    if (
        not isinstance(names, list)
        or len(names) < 2
        or any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names)
        or len(set(names)) != len(names)
        or len({name.strip() for name in names}) != len(names)
    ):
        raise DatasetError("class_names must contain at least two distinct nonempty strings.")
    if expected_classes is not None and len(names) != expected_classes:
        raise DatasetError("num_classes does not match the manifest class_names.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise DatasetError("records must be a nonempty list.")
    teachers: dict[str, str] = {}
    videos: dict[str, str] = {}
    video_teachers: dict[str, str] = {}
    paths: set[Path] = set()
    resolved_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise DatasetError(f"Record {index} must be an object.")
        for key in ("path", "split", "teacher_id", "video_id"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise DatasetError(f"Record {index} requires a nonempty string {key}.")
            if record[key] != record[key].strip():
                raise DatasetError(f"Record {index} {key} must not have surrounding whitespace.")
        split = record["split"]
        if split not in {"train", "val", "test"}:
            raise DatasetError(f"Record {index} has an unsupported split: {split}.")
        relative = record["path"]
        windows_path = PureWindowsPath(relative)
        if Path(relative).is_absolute() or windows_path.drive or windows_path.root:
            raise DatasetError(f"Record {index} must use a relative NPZ path.")
        sequence_path = (manifest_path.parent / relative).resolve()
        if sequence_path.suffix.lower() != ".npz" or not sequence_path.is_file():
            raise DatasetError(f"Record {index} does not identify an existing NPZ: {sequence_path}.")
        if sequence_path in paths:
            raise DatasetError(f"Repeated feature file in manifest: {sequence_path}.")
        paths.add(sequence_path)
        teacher = record["teacher_id"]
        video = record["video_id"]
        for values, identifier, kind in ((teachers, teacher, "teacher_id"), (videos, video, "video_id")):
            if identifier in values and values[identifier] != split:
                raise DatasetError(f"{kind} {identifier!r} appears in multiple splits.")
            values[identifier] = split
        if video in video_teachers and video_teachers[video] != teacher:
            raise DatasetError(f"video_id {video!r} is assigned to multiple teachers.")
        video_teachers[video] = teacher
        resolved = dict(record)
        resolved["path"] = str(sequence_path)
        resolved_records.append(resolved)
    result = dict(manifest)
    result["records"] = resolved_records
    result["manifest_path"] = str(manifest_path)
    result["manifest_dir"] = str(manifest_path.parent)
    return result


def _feature_header(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    """Inspect feature shape and dtype without decoding the feature array."""
    try:
        with ZipFile(path) as archive:
            if archive.namelist().count("features.npy") != 1:
                raise DatasetError(f"{path} must contain exactly one features array.")
            with archive.open("features.npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise DatasetError(f"{path} uses an unsupported feature array format: {version}.")
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"Cannot inspect features in {path}: {exc}") from exc
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1 or not np.issubdtype(dtype, np.floating):
        raise DatasetError(f"{path}: features must be a nonempty floating-point [T, F] array.")
    return tuple(shape), dtype


def _metadata(path: Path, num_classes: int | None, require_labels: bool) -> dict[str, Any]:
    shape, dtype = _feature_header(path)
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if "timestamps" not in arrays:
                raise DatasetError(f"{path}: timestamps are required.")
            timestamps = arrays["timestamps"]
            if (
                timestamps.ndim != 1
                or timestamps.shape[0] != shape[0]
                or not np.issubdtype(timestamps.dtype, np.floating)
            ):
                raise DatasetError(f"{path}: timestamps must be a floating-point [T] array.")
            timestamps = timestamps.astype(np.float64, copy=True)
            if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
                raise DatasetError(f"{path}: timestamps must be finite and strictly increasing.")
            has_labels = "labels" in arrays
            if require_labels and not has_labels:
                raise DatasetError(f"{path}: labels are required for supervised data.")
            if has_labels:
                labels = arrays["labels"]
                if (
                    labels.ndim != 1
                    or labels.shape[0] != shape[0]
                    or not np.issubdtype(labels.dtype, np.integer)
                ):
                    raise DatasetError(f"{path}: labels must be an integer [T] array.")
                if np.any(labels < -1) or np.any(labels > np.iinfo(np.int64).max):
                    raise DatasetError(f"{path}: labels must be -1 or nonnegative int64 values.")
                if num_classes is not None and np.any(labels >= num_classes):
                    raise DatasetError(f"{path}: a label is outside [0, num_classes).")
                labels = labels.astype(np.int64, copy=True)
            else:
                labels = np.full(shape[0], -1, dtype=np.int64)
            if "boundaries" in arrays:
                boundaries = arrays["boundaries"]
                if (
                    boundaries.ndim != 1
                    or boundaries.shape[0] != shape[0]
                    or not (
                        np.issubdtype(boundaries.dtype, np.integer)
                        or np.issubdtype(boundaries.dtype, np.floating)
                    )
                    or not np.isfinite(boundaries).all()
                    or not np.all((boundaries == 0) | (boundaries == 1))
                ):
                    raise DatasetError(f"{path}: boundaries must be a finite binary numeric [T] array.")
                boundaries = boundaries.astype(np.float32, copy=True)
            else:
                boundaries = np.zeros(shape[0], dtype=np.float32)
                boundaries[1:] = (
                    (labels[1:] >= 0) & (labels[:-1] >= 0) & (labels[1:] != labels[:-1])
                ).astype(np.float32)
            boundaries[1:] = np.maximum(boundaries[1:], (
                (labels[1:] >= 0) & (labels[:-1] >= 0) & (labels[1:] != labels[:-1])
            ).astype(np.float32))
            boundaries[0] = 0
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"Cannot read sequence metadata from {path}: {exc}") from exc
    stat = path.stat()
    return {
        "shape": shape,
        "dtype": dtype,
        "timestamps": timestamps,
        "labels": labels,
        "boundaries": boundaries,
        "has_labels": has_labels,
        "file_stamp": (stat.st_size, stat.st_mtime_ns),
    }


def _features(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            features = arrays["features"]
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
        raise DatasetError(f"Cannot read features from {path}: {exc}") from exc
    if features.shape != shape or not np.issubdtype(features.dtype, np.floating):
        raise DatasetError(f"{path}: feature dimensions or dtype changed.")
    if not np.isfinite(features).all():
        raise DatasetError(f"{path}: features contain a nonfinite value.")
    with np.errstate(over="ignore", invalid="ignore"):
        features = np.ascontiguousarray(features, dtype=np.float32)
    if not np.isfinite(features).all():
        raise DatasetError(f"{path}: features exceed the finite float32 range.")
    return features


def _run_ranges(labels: np.ndarray, timestamps: np.ndarray, step_seconds: float, require_labels: bool):
    start = None
    for index in range(len(timestamps)):
        gap = index > 0 and timestamps[index] - timestamps[index - 1] > 1.5 * step_seconds
        adjacent_known = not require_labels or (index > 0 and labels[index] >= 0 and labels[index - 1] >= 0)
        if index > 0 and not gap and adjacent_known:
            difference = timestamps[index] - timestamps[index - 1]
            if not np.isclose(difference, step_seconds, rtol=1e-5, atol=1e-6):
                raise DatasetError("Within-run timestamp spacing must match step_seconds.")
        if gap and start is not None:
            yield start, index
            start = None
        if require_labels and labels[index] < 0:
            if start is not None:
                yield start, index
                start = None
        elif start is None:
            start = index
    if start is not None:
        yield start, len(timestamps)


class FeatureDataset(Dataset):
    """Supervised NPZ sequences split at unknown labels and timestamp gaps.

    Features are read on access. Labels and timestamps are cached during
    construction. max_steps optionally partitions training runs into adjacent,
    nonoverlapping windows; validation and test runs remain complete.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        num_classes: int | None = None,
        step_seconds: float = 1.0,
        max_steps: int | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise DatasetError("FeatureDataset split must be train, val, or test.")
        self.step_seconds = _step_size(step_seconds)
        if max_steps is not None and (
            isinstance(max_steps, bool) or not isinstance(max_steps, numbers.Integral) or max_steps < 1
        ):
            raise DatasetError("max_steps must be a positive integer or None.")
        if max_steps is not None and split != "train":
            raise DatasetError("max_steps can only partition the training split.")
        self.manifest = read_manifest(manifest_path, num_classes)
        self.class_names = list(self.manifest["class_names"])
        self.num_classes = len(self.class_names)
        self.split = split
        self.records = [record for record in self.manifest["records"] if record["split"] == split]
        if not self.records:
            raise DatasetError(f"The manifest contains no records for split {split!r}.")
        self._metadata = []
        self.runs: list[dict[str, int]] = []
        self.feature_dim = None
        for record_index, record in enumerate(self.records):
            metadata = _metadata(Path(record["path"]), self.num_classes, require_labels=True)
            dimension = metadata["shape"][1]
            if self.feature_dim is None:
                self.feature_dim = dimension
            elif dimension != self.feature_dim:
                raise DatasetError("All records in a split must have the same feature dimension.")
            self._metadata.append(metadata)
            for start, stop in _run_ranges(metadata["labels"], metadata["timestamps"], self.step_seconds, True):
                window = stop - start if max_steps is None else int(max_steps)
                for left in range(start, stop, window):
                    self.runs.append({"record": record_index, "start": left, "stop": min(left + window, stop)})
        if not self.runs:
            raise DatasetError(f"Split {split!r} has no labelled time steps.")
        self.record_count = len(self.records)
        self.run_count = len(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        run = self.runs[index]
        record = self.records[run["record"]]
        metadata = self._metadata[run["record"]]
        path = Path(record["path"])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != metadata["file_stamp"]:
            raise DatasetError(f"{path} changed after dataset construction; reopen the dataset.")
        start, stop = run["start"], run["stop"]
        features = _features(path, metadata["shape"])
        identifier = f"{record['video_id']}::{run['record']}::{start}:{stop}"
        boundaries = metadata["boundaries"][start:stop].copy()
        boundaries[0] = 0
        return {
            "features": torch.from_numpy(features[start:stop].copy()),
            "labels": torch.from_numpy(metadata["labels"][start:stop].copy()),
            "boundaries": torch.from_numpy(boundaries),
            "timestamps": torch.from_numpy(metadata["timestamps"][start:stop].copy()),
            "indices": torch.arange(start, stop, dtype=torch.int64),
            "id": identifier,
            "source_id": record["video_id"],
            "teacher_id": record["teacher_id"],
            "video_id": record["video_id"],
        }


def load_feature_sequence(
    path: str | Path,
    feature_dim: int | None = None,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Read an inference NPZ; absent labels become -1 without losing time indices."""
    sequence_path = Path(path).expanduser().resolve()
    classes = _class_count(num_classes)
    metadata = _metadata(sequence_path, classes, require_labels=False)
    if feature_dim is not None and (
        isinstance(feature_dim, bool)
        or not isinstance(feature_dim, numbers.Integral)
        or feature_dim < 1
        or metadata["shape"][1] != feature_dim
    ):
        raise DatasetError("feature_dim must match the positive NPZ feature dimension.")
    return {
        "features": torch.from_numpy(_features(sequence_path, metadata["shape"])),
        "labels": torch.from_numpy(metadata["labels"]),
        "boundaries": torch.from_numpy(metadata["boundaries"]),
        "timestamps": torch.from_numpy(metadata["timestamps"]),
        "indices": torch.arange(metadata["shape"][0], dtype=torch.int64),
        "id": sequence_path.stem,
        "source_id": sequence_path.stem,
        "teacher_id": "",
        "video_id": sequence_path.stem,
    }


def _validate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(sample, dict):
        raise DatasetError("Each sample must be a dictionary.")
    features = sample.get("features")
    timestamps = sample.get("timestamps")
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or features.shape[0] < 1
        or features.shape[1] < 1
        or not torch.is_floating_point(features)
        or not torch.isfinite(features).all().item()
    ):
        raise DatasetError("Sample features must be a finite nonempty floating-point [T, F] tensor.")
    count = features.shape[0]
    if (
        not isinstance(timestamps, torch.Tensor)
        or timestamps.ndim != 1
        or len(timestamps) != count
        or not torch.is_floating_point(timestamps)
        or not torch.isfinite(timestamps).all().item()
        or (count > 1 and not torch.all(timestamps[1:] > timestamps[:-1]).item())
    ):
        raise DatasetError("Sample timestamps must be finite, strictly increasing, and have shape [T].")
    integer_types = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    result = dict(sample)
    for key, default in (("labels", torch.full((count,), -1, dtype=torch.int64)), ("indices", torch.arange(count))):
        value = sample.get(key, default)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) != count or value.dtype not in integer_types:
            raise DatasetError(f"Sample {key} must be an integer [T] tensor.")
        result[key] = value.to(dtype=torch.int64)
    if torch.any(result["labels"] < -1).item():
        raise DatasetError("Sample labels must be -1 or nonnegative.")
    boundaries = sample.get("boundaries")
    if boundaries is None:
        boundaries = torch.zeros(count, dtype=torch.float32, device=result["labels"].device)
        boundaries[1:] = (
            (result["labels"][1:] >= 0)
            & (result["labels"][:-1] >= 0)
            & (result["labels"][1:] != result["labels"][:-1])
        ).to(torch.float32)
    if (
        not isinstance(boundaries, torch.Tensor)
        or boundaries.ndim != 1
        or len(boundaries) != count
        or (boundaries.dtype not in integer_types and not torch.is_floating_point(boundaries))
        or not torch.isfinite(boundaries).all().item()
        or not torch.all((boundaries == 0) | (boundaries == 1)).item()
    ):
        raise DatasetError("Sample boundaries must be a finite binary numeric [T] tensor.")
    result["boundaries"] = boundaries.to(dtype=torch.float32).clone()
    changes = ((result["labels"][1:] >= 0) & (result["labels"][:-1] >= 0)
               & (result["labels"][1:] != result["labels"][:-1])).to(torch.float32)
    result["boundaries"][1:] = torch.maximum(result["boundaries"][1:], changes)
    result["boundaries"][0] = 0
    indices = result["indices"]
    if torch.any(indices < 0).item() or (count > 1 and not torch.all(indices[1:] > indices[:-1]).item()):
        raise DatasetError("Sample indices must be nonnegative and strictly increasing.")
    result["features"] = features.to(dtype=torch.float32)
    result["timestamps"] = timestamps.to(dtype=torch.float64)
    if not torch.isfinite(result["features"]).all().item():
        raise DatasetError("Sample features exceed the finite float32 range.")
    return result


def split_feature_runs(
    sample: dict[str, Any], step_seconds: float = 1.0, require_labels: bool = False
) -> list[dict[str, Any]]:
    """Split timestamp gaps, optionally excluding unknown labels, while retaining indices."""
    step = _step_size(step_seconds)
    if not isinstance(sample, dict):
        raise DatasetError("Each sample must be a dictionary.")
    if require_labels and "labels" not in sample:
        raise DatasetError("Labels are required when require_labels=True.")
    sample = _validate_sample(sample)
    labels = sample["labels"].detach().cpu().numpy()
    timestamps = sample["timestamps"].detach().cpu().numpy()
    result = []
    identifier = str(sample.get("id", sample.get("video_id", "sequence")))
    for start, stop in _run_ranges(labels, timestamps, step, require_labels):
        run = dict(sample)
        for key in ("features", "labels", "boundaries", "timestamps", "indices"):
            run[key] = sample[key][start:stop].clone()
        run["boundaries"][0] = 0
        run["source_id"] = sample.get("source_id", identifier)
        run["id"] = f"{identifier}::{start}:{stop}"
        result.append(run)
    return result


def collate_features(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad sequences and retain true timestamps and original indices at valid positions."""
    if not samples:
        raise DatasetError("Cannot collate an empty sample list.")
    samples = [_validate_sample(sample) for sample in samples]
    dimensions = {sample["features"].shape[1] for sample in samples}
    devices = {sample[key].device for sample in samples for key in ("features", "labels", "boundaries", "timestamps", "indices")}
    if len(dimensions) != 1 or len(devices) != 1:
        raise DatasetError("Samples must share a feature dimension and tensor device.")
    device = samples[0]["features"].device
    lengths = torch.tensor([len(sample["labels"]) for sample in samples], dtype=torch.int64, device=device)
    batch, steps, dimension = len(samples), int(lengths.max().item()), dimensions.pop()
    features = torch.zeros((batch, steps, dimension), dtype=torch.float32, device=device)
    labels = torch.full((batch, steps), -1, dtype=torch.int64, device=device)
    boundaries = torch.zeros((batch, steps), dtype=torch.float32, device=device)
    timestamps = torch.zeros((batch, steps), dtype=torch.float64, device=device)
    indices = torch.full((batch, steps), -1, dtype=torch.int64, device=device)
    valid_mask = torch.zeros((batch, steps), dtype=torch.bool, device=device)
    for index, sample in enumerate(samples):
        length = len(sample["labels"])
        features[index, :length] = sample["features"]
        labels[index, :length] = sample["labels"]
        boundaries[index, :length] = sample["boundaries"]
        timestamps[index, :length] = sample["timestamps"]
        indices[index, :length] = sample["indices"]
        valid_mask[index, :length] = True
    return {
        "features": features,
        "labels": labels,
        "boundaries": boundaries,
        "timestamps": timestamps,
        "indices": indices,
        "valid_mask": valid_mask,
        "lengths": lengths,
        "ids": [str(sample.get("id", "")) for sample in samples],
        "teacher_ids": [str(sample.get("teacher_id", "")) for sample in samples],
        "video_ids": [str(sample.get("video_id", "")) for sample in samples],
    }
