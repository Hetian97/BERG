"""Command-line training, evaluation, process graph export, and profiling."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import BERGConfig
from data import FeatureDataset, collate_features, load_feature_sequence, read_manifest, split_feature_runs
from losses import joint_loss
from metrics import SegmentationMeter
from model import BERG


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_from_name(name):
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu and cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("CUDA device index is out of range")
    return device


def tensor_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def dataset_loader(dataset, batch_size, shuffle=False, seed=2022):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=collate_features, generator=generator)


def save_checkpoint(path, model, optimizer, epoch, best_score, class_names):
    temporary = Path(path).with_suffix(".tmp")
    torch.save({"schema_version": 1, "config": model.config.to_dict(),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_score": best_score, "class_names": class_names}, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    config = BERGConfig.from_dict(checkpoint["config"])
    model = BERG(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    names = checkpoint["class_names"]
    if (not isinstance(names, list) or len(names) != config.num_classes
            or any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Checkpoint class vocabulary is inconsistent")
    return model, names, checkpoint


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    meter = SegmentationMeter(model.config.num_classes)
    selected_total = observed_total = 0
    for raw_batch in loader:
        batch = tensor_batch(raw_batch, device)
        output = model(batch["features"], batch["valid_mask"])
        selected_total += int(output["executed_steps"].sum().item())
        observed_total += int(batch["valid_mask"].sum().item())
        for row in range(batch["features"].shape[0]):
            count = int(batch["valid_mask"][row].sum().item())
            predictions = output["logits"][row, :count].argmax(-1).cpu().numpy()
            truth = batch["labels"][row, :count].cpu().numpy()
            meter.update(predictions, truth, pred_segments=model.decode_intervals(output, row),
                         true_boundaries=batch["boundaries"][row, :count].cpu().numpy())
    result = meter.compute()
    result["Trigger"] = 100 * selected_total / max(1, observed_total)
    result["observed_steps"] = observed_total
    return result


def train_command(args):
    seed_everything(args.seed)
    device = device_from_name(args.device)
    manifest = read_manifest(args.manifest)
    overrides = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    step = overrides.get("step_seconds", 1.0)
    training = FeatureDataset(args.manifest, "train", step_seconds=step)
    validation = FeatureDataset(args.manifest, "val", step_seconds=step)
    if training.feature_dim != validation.feature_dim:
        raise ValueError("Training and validation feature dimensions differ")
    overrides.setdefault("input_dim", training.feature_dim)
    overrides.setdefault("num_classes", len(training.class_names))
    config = BERGConfig.from_dict(overrides)
    if config.input_dim != training.feature_dim or config.num_classes != len(training.class_names):
        raise ValueError("Model dimensions do not match the manifest")
    model = BERG(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("The output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"model": config.to_dict(), "class_names": manifest["class_names"],
                "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "gradient_clip": args.gradient_clip,
                "amp": use_amp, "device": str(device), "python": platform.python_version(),
                "cpu_threads": torch.get_num_threads(),
                "torch": str(torch.__version__), "numpy": np.__version__,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "training_runs": len(training), "validation_runs": len(validation)}
    write_json(output_dir / "run.json", metadata)
    best_score = -1.0
    history = []
    validation_loader = dataset_loader(validation, args.batch_size)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {}
        steps = 0
        loader = dataset_loader(training, args.batch_size, shuffle=True, seed=args.seed + epoch)
        for raw_batch in loader:
            batch = tensor_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(batch["features"], batch["valid_mask"])
                losses = joint_loss(model, output, batch["labels"], batch["boundaries"])
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            steps += 1
        validation_scores = evaluate_loader(model, validation_loader, device)
        record = {"epoch": epoch, "training_loss": {k: v / steps for k, v in totals.items()},
                  "validation": validation_scores}
        history.append(record)
        write_json(output_dir / "history.json", history)
        score = validation_scores["F1@50"]
        if score > best_score:
            best_score = score
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_score, manifest["class_names"])
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_score, manifest["class_names"])
        print(json.dumps(record, allow_nan=False), flush=True)


def evaluation_command(args):
    device = device_from_name(args.device)
    model, names, _ = load_checkpoint(args.checkpoint, device)
    dataset = FeatureDataset(args.manifest, args.split, num_classes=model.config.num_classes,
                             step_seconds=model.config.step_seconds)
    if dataset.class_names != names or dataset.feature_dim != model.config.input_dim:
        raise ValueError("Evaluation data do not match checkpoint features and class order")
    result = evaluate_loader(model, dataset_loader(dataset, args.batch_size), device)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


@torch.no_grad()
def prediction_command(args):
    device = device_from_name(args.device)
    model, names, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    sequence = load_feature_sequence(args.features, feature_dim=model.config.input_dim,
                                     num_classes=model.config.num_classes)
    runs = split_feature_runs(sequence, step_seconds=model.config.step_seconds, require_labels=False)
    results = []
    for run in runs:
        batch = tensor_batch(collate_features([run]), device)
        output = model(batch["features"], batch["valid_mask"])
        graph = model.process_graph(output, 0, batch["timestamps"][0], batch["indices"][0])
        for node in graph["nodes"]:
            node["class_name"] = names[node["label"]]
        graph["id"] = run["id"]
        graph["deep_update_indices"] = batch["indices"][0, output["selected"][0]].cpu().tolist()
        graph["step_labels"] = output["logits"][0].argmax(-1).cpu().tolist()
        results.append(graph)
    write_json(args.output, {"class_names": names, "runs": results})


@torch.no_grad()
def profile_command(args):
    device = device_from_name(args.device)
    model, _, _ = load_checkpoint(args.checkpoint, device)
    if args.fully_deep:
        model.config = replace(model.config, budget_ratio=1.0, selection_threshold=0.0)
    model.eval()
    sequence = load_feature_sequence(args.features, feature_dim=model.config.input_dim,
                                     num_classes=model.config.num_classes)
    runs = split_feature_runs(sequence, step_seconds=model.config.step_seconds, require_labels=False)
    if len(runs) != 1:
        raise ValueError("Profile one continuous execution window at a time")
    batch = tensor_batch(collate_features(runs), device)
    def iteration():
        output = model(batch["features"], batch["valid_mask"])
        model.process_graph(output, 0, batch["timestamps"][0], batch["indices"][0])
        return output
    for _ in range(args.warmup):
        iteration()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    latencies = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        output = iteration()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(1000 * (time.perf_counter() - start))
    result = {"device": str(device), "observed_steps": int(batch["valid_mask"].sum()),
              "selected_steps": int(output["executed_steps"].sum()),
              "parameters": sum(p.numel() for p in model.parameters()),
              "mean_latency_ms": float(np.mean(latencies)), "median_latency_ms": float(np.median(latencies)),
              "latency_std_ms": float(np.std(latencies, ddof=1)) if args.repeats > 1 else 0.0,
              "includes_process_graph": True, "includes_visual_encoding": False,
              "mode": "fully_deep" if args.fully_deep else "budget_constrained"}
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return value


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Value must be finite and positive")
    return value


def main():
    parser = argparse.ArgumentParser(description="BERG temporal refinement and process graph analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--config")
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=positive_int, default=50)
    training.add_argument("--batch-size", type=positive_int, default=4)
    training.add_argument("--learning-rate", type=positive_float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-2)
    training.add_argument("--gradient-clip", type=positive_float, default=1.0)
    training.add_argument("--seed", type=int, default=2022)
    training.add_argument("--amp", action="store_true")
    training.set_defaults(function=train_command)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    evaluation.add_argument("--batch-size", type=positive_int, default=4)
    evaluation.set_defaults(function=evaluation_command)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--features", required=True)
    prediction.set_defaults(function=prediction_command)
    profiling = commands.add_parser("profile")
    profiling.add_argument("--features", required=True)
    profiling.add_argument("--warmup", type=positive_int, default=10)
    profiling.add_argument("--repeats", type=positive_int, default=50)
    profiling.add_argument("--fully-deep", action="store_true")
    profiling.set_defaults(function=profile_command)
    for command in (evaluation, prediction, profiling):
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output", required=True)
    for command in (training, evaluation, prediction, profiling):
        command.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if hasattr(args, "weight_decay") and (not np.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("weight-decay must be finite and nonnegative")
    args.function(args)


if __name__ == "__main__":
    main()
