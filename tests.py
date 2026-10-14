"""Independent numerical, masking, data-isolation, and workflow checks."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from config import BERGConfig
from data import DatasetError, FeatureDataset, collate_features, load_feature_sequence, read_manifest, split_feature_runs
from losses import coverage_loss, joint_loss, position_targets
from metrics import SegmentationMeter, interval_counts, intervals_from_labels
from model import BERG, neighbor_max, select_steps
from train import load_checkpoint, save_checkpoint, seed_everything


def small_config(**kwargs):
    values = dict(input_dim=8, num_classes=3, hidden_dim=16, num_heads=4,
                  num_anchors=4, deep_layers=2, dropout=0.0)
    values.update(kwargs)
    return BERGConfig(**values)


def feature_fixture(path, labels, timestamps=None, boundaries=None):
    count = len(labels)
    values = {"features": np.arange(count * 8, dtype=np.float32).reshape(count, 8) / 40,
              "labels": np.asarray(labels, dtype=np.int64),
              "timestamps": np.asarray(timestamps if timestamps is not None else np.arange(count) + 20,
                                       dtype=np.float64)}
    if boundaries is not None:
        values["boundaries"] = np.asarray(boundaries, dtype=np.float32)
    np.savez(path, **values)


class SelectionTests(unittest.TestCase):
    def test_capacity_and_tie_order(self):
        scores = torch.tensor([[0.5] * 5, [0.9, 0.1, 0.0, 0.0, 0.0]])
        valid = torch.tensor([[1] * 5, [1, 1, 0, 0, 0]], dtype=torch.bool)
        mask = select_steps(scores, valid, 0.35, 0.2)
        self.assertEqual(mask.tolist(), [[True, True, False, False, False], [True, False, False, False, False]])
        self.assertFalse(select_steps(scores, valid, 0.0, 0.0).any())
        self.assertFalse(select_steps(scores, valid, 1.0, 2.0).any())

    def test_neighbor_excludes_current(self):
        scores = torch.tensor([[0.1, 0.9, 0.2]])
        valid = torch.ones_like(scores, dtype=torch.bool)
        torch.testing.assert_close(neighbor_max(scores, valid, 1), torch.tensor([[0.9, 0.2, 0.9]]))
        torch.testing.assert_close(neighbor_max(scores[:, :1], valid[:, :1], 2), torch.zeros(1, 1))

    def test_random_capacity_properties(self):
        for length in (1, 2, 7, 15):
            for beta in (0.0, 0.15, 0.35, 1.0):
                valid = torch.ones(2, length, dtype=torch.bool)
                scores = torch.linspace(0, 1, length).repeat(2, 1)
                mask = select_steps(scores, valid, beta, 0.4)
                self.assertTrue((mask.sum(1) <= int(np.ceil(beta * length))).all())
                self.assertTrue((scores[mask] >= 0.4).all())


class ModelTests(unittest.TestCase):
    def test_half_precision_boundary_threshold(self):
        model = BERG(small_config()).eval()
        logits = torch.zeros(1, 4, 3, dtype=torch.float16)
        logits[:, :, 0] = 2
        output = {"valid_mask": torch.ones(1, 4, dtype=torch.bool), "logits": logits,
                  "boundary_logits": torch.tensor([[0., -1e-4, 0., 1e-4]], dtype=torch.float16)}
        intervals = model.decode_intervals(output, 0)
        self.assertEqual([(item["start"], item["end"]) for item in intervals], [(0, 2), (2, 3), (3, 4)])

    def test_half_statistics_and_long_anchor_coordinates(self):
        model = BERG(small_config()).eval()
        hidden = torch.zeros(1, 4, 16, dtype=torch.float16)
        probabilities = torch.tensor([[[1., 0., 0.]] * 4], dtype=torch.float16)
        statistics = model._statistics(hidden, probabilities, torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(statistics.dtype, torch.float32)
        self.assertTrue(torch.isfinite(statistics).all())
        sequence = torch.linspace(-1, 1, 4096).reshape(1, 4096, 1).repeat(1, 1, 16).half()
        valid = torch.ones(1, 4096, dtype=torch.bool)
        _, half_weights = model._anchors(sequence, valid)
        _, full_weights = model._anchors(sequence.float(), valid)
        torch.testing.assert_close(half_weights, full_weights, atol=0, rtol=0)

    def test_cpu_autocast_prediction_finiteness(self):
        model = BERG(small_config()).train()
        for dtype in (torch.float16, torch.bfloat16):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=dtype):
                output = model(torch.randn(1, 5, 8), torch.ones(1, 5, dtype=torch.bool))
                losses = joint_loss(model, output, torch.tensor([[0, 0, 1, 1, 2]]), torch.zeros(1, 5))
            self.assertTrue(torch.isfinite(output["scores"]).all())
            self.assertTrue(torch.isfinite(losses["total"]))
            losses["total"].backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            node = model.interval_node(output, 0, 0, 5)
            self.assertEqual(node["weights"].dtype, torch.float32)
            self.assertEqual(node["consistency"].dtype, torch.float32)

    def setUp(self):
        seed_everything(2022)
        self.model = BERG(small_config()).eval()
        self.features = torch.arange(56, dtype=torch.float32).reshape(1, 7, 8) / 40
        self.valid = torch.ones(1, 7, dtype=torch.bool)

    def test_padding_invariance(self):
        first = self.model(self.features, self.valid)
        padded = torch.cat((self.features, torch.full((1, 4, 8), float("nan"))), dim=1)
        valid = torch.cat((self.valid, torch.zeros(1, 4, dtype=torch.bool)), dim=1)
        second = self.model(padded, valid)
        for key in ("logits", "scores", "features", "boundary_logits"):
            torch.testing.assert_close(first[key], second[key][:, :7], atol=2e-6, rtol=2e-5)
        self.assertEqual(first["selected"].tolist(), second["selected"][:, :7].tolist())

    def test_actual_sparse_execution(self):
        counts = []
        hook = self.model.deep[0].register_forward_pre_hook(lambda module, args: counts.append(args[0].shape[1]))
        output = self.model(self.features, self.valid)
        hook.remove()
        self.assertEqual(sum(counts), int(output["selected"].sum()))
        unselected = ~output["selected"]
        torch.testing.assert_close(output["features"][unselected], output["light_features"][unselected], atol=0, rtol=0)

    def test_zero_budget_single_step(self):
        model = BERG(small_config(budget_ratio=0.0)).eval()
        output = model(self.features[:, :1], self.valid[:, :1])
        self.assertEqual(int(output["executed_steps"].sum()), 0)
        node = model.interval_node(output, 0, 0, 1)
        torch.testing.assert_close(node["weights"], torch.ones(1))
        graph = model.process_graph(output, 0, torch.tensor([44.0]), torch.tensor([12]))
        self.assertEqual(graph["nodes"][0]["time_indices"], [12])
        self.assertEqual(graph["edges"], [])

    def test_anchor_formula(self):
        output = self.model(self.features, self.valid)
        hidden = output["light_features"][0]
        centers = torch.linspace(0, 6, 4)
        expected = torch.stack([(-torch.abs(torch.arange(7) - center) / 16).softmax(0) @ hidden
                                for center in centers])
        torch.testing.assert_close(output["anchors"][0], expected)

    def test_interval_formula_and_pointers(self):
        output = self.model(self.features, self.valid)
        node = self.model.interval_node(output, 0, 1, 6)
        hidden = output["features"][0, 1:6]
        q = self.model.interval_query(hidden.mean(0))
        logits = self.model.interval_key(hidden) @ q / 4
        flags = output["selected"][0, 1:6].float()
        weights = (logits.exp() * (flags + 0.1))
        weights = weights / weights.sum()
        torch.testing.assert_close(node["weights"], weights)
        expected = (weights * output["logits"][0, 1:6].softmax(-1)[:, node["label"]]).sum()
        torch.testing.assert_close(node["consistency"], expected)

    def test_training_gradient_finiteness(self):
        self.model.train()
        output = self.model(self.features, self.valid)
        labels = torch.tensor([[0, 0, 1, 1, 2, 2, 0]])
        boundaries = torch.tensor([[0, 0, 1, 0, 1, 0, 1]], dtype=torch.float32)
        losses = joint_loss(self.model, output, labels, boundaries)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(self.model.value_mlp[-1].weight.grad.abs().sum()), 0)

    def test_empty_supervision_is_finite(self):
        output = self.model(self.features, self.valid)
        losses = joint_loss(self.model, output, torch.full((1, 7), -1), torch.zeros(1, 7))
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["graph"]), 0)

    def test_prefix_mask_validation(self):
        with self.assertRaises(ValueError):
            self.model(self.features, torch.tensor([[1, 0, 1, 0, 0, 0, 0]], dtype=torch.bool))


class LossTests(unittest.TestCase):
    def test_loss_mask_and_supervision_validation(self):
        model = BERG(small_config()).eval()
        output = model(torch.ones(1, 4, 8), torch.tensor([[1, 1, 0, 0]], dtype=torch.bool))
        loss = joint_loss(model, output, torch.tensor([[0, 1, 999, 999]]), torch.zeros(1, 4))
        self.assertTrue(torch.isfinite(loss["total"]))
        with self.assertRaises(ValueError):
            joint_loss(model, output, torch.ones(1, 3, dtype=torch.long), torch.zeros(1, 3))
        with self.assertRaises(ValueError):
            joint_loss(model, output, torch.tensor([[0, 1, -1, -1]]), torch.tensor([[0., 0.5, 0., 0.]]))

    def test_extreme_value_logits_and_autocast(self):
        model = BERG(small_config()).train()
        features = torch.ones(1, 5, 8)
        valid = torch.ones(1, 5, dtype=torch.bool)
        labels = torch.tensor([[0, 0, 1, 1, 2]])
        boundaries = torch.tensor([[0, 0, 1, 0, 1]], dtype=torch.float32)
        with torch.no_grad():
            model.value_mlp[-1].weight.zero_()
            model.value_mlp[-1].bias.fill_(20)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(features, valid)
            losses = joint_loss(model, output, labels, boundaries)
        self.assertEqual(output["scores"].dtype, torch.float32)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(model.value_mlp[-1].bias.grad).all())

    def test_detached_value_target(self):
        logits = torch.randn(1, 3, 3, requires_grad=True)
        targets = position_targets(logits, torch.tensor([[0, 1, 2]]),
                                   torch.tensor([[0.0, 1.0, 0.0]]), torch.ones(1, 3, dtype=torch.bool))
        self.assertFalse(targets.requires_grad)
        self.assertEqual(float(targets[0, 1]), 1)

    def test_coverage_formula(self):
        gates = torch.tensor([[0.2, 0.3, 0.4]], requires_grad=True)
        boundaries = torch.tensor([[0.0, 1.0, 0.0]])
        actual = coverage_loss(gates, boundaries, torch.ones(1, 3, dtype=torch.bool), 1)
        expected = -torch.log(torch.tensor(1 - 0.8 * 0.7 * 0.6))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(gates.grad).all())
        self.assertEqual(float(coverage_loss(gates, torch.zeros_like(gates),
                                            torch.ones_like(gates, dtype=torch.bool), 1)), 0)


class DataMetricTests(unittest.TestCase):
    def test_configuration_types_and_metric_overflow(self):
        for key, value in (("budget_ratio", True), ("neighbor_radius", True),
                           ("dropout", False), ("value_weight", np.float32(np.nan))):
            with self.assertRaises(ValueError):
                small_config(**{key: value})
        for labels in (np.array([2 ** 64 - 1], dtype=np.uint64), np.array([2.0 ** 63])):
            with self.assertRaises(ValueError):
                intervals_from_labels(labels)

    def test_spacing_and_identifier_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            feature_fixture(folder / "clip.npz", [0, 0, 1], [10, 10.5, 11])
            sample = load_feature_sequence(folder / "clip.npz")
            with self.assertRaises(DatasetError):
                split_feature_runs(sample, step_seconds=1.0)
            self.assertEqual(len(split_feature_runs(sample, step_seconds=0.5)), 1)
            manifest = {"schema_version": 1, "class_names": ["a", "b"], "records": [
                {"path": "clip.npz", "split": "train", "teacher_id": "teacher ", "video_id": "a"}]}
            path = folder / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(DatasetError):
                read_manifest(path)

    def test_checkpoint_vocabulary_validation(self):
        model = BERG(small_config()).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pt"
            save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), 1, 0.0, "abc")
            with self.assertRaises(ValueError):
                load_checkpoint(path, torch.device("cpu"))

    def test_boundary_union_and_invalid_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clip.npz"
            feature_fixture(path, [0, 1, 1, 1], boundaries=[0, 0, 0, 1])
            sample = load_feature_sequence(path, num_classes=3)
            self.assertEqual(sample["boundaries"].tolist(), [0, 1, 0, 1])
        with self.assertRaises(DatasetError):
            split_feature_runs(None, require_labels=True)

    def test_temporal_gaps_and_original_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clip.npz"
            feature_fixture(path, [0, 0, -1, 1, 1, 2], [20, 21, 22, 25, 26, 30])
            sequence = load_feature_sequence(path, num_classes=3)
            runs = split_feature_runs(sequence, require_labels=True)
            self.assertEqual([run["indices"].tolist() for run in runs], [[0, 1], [3, 4], [5]])
            self.assertEqual([run["timestamps"].tolist() for run in runs], [[20, 21], [25, 26], [30]])
            batch = collate_features(runs)
            self.assertEqual(batch["labels"].tolist(), [[0, 0], [1, 1], [2, -1]])
            self.assertTrue((batch["boundaries"][:, 0] == 0).all())

    def test_manifest_split_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name in ("a.npz", "b.npz"):
                feature_fixture(folder / name, [0, 1])
            manifest = {"schema_version": 1, "class_names": ["a", "b"], "records": [
                {"path": "a.npz", "split": "train", "teacher_id": "same", "video_id": "a"},
                {"path": "b.npz", "split": "test", "teacher_id": "same", "video_id": "b"}]}
            path = folder / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(DatasetError):
                read_manifest(path)

    def test_explicit_same_class_boundary(self):
        segments = intervals_from_labels([0, 0, 0, 0], [0, 0, 1, 0])
        self.assertEqual([(s["start"], s["end"]) for s in segments], [(0, 2), (2, 4)])
        counts = interval_counts([{"start": 0, "end": 4, "label": 0}], segments, 0.5)
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (1, 0, 1))

    def test_perfect_and_wrong_metrics(self):
        meter = SegmentationMeter(3)
        meter.update(np.array([0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 2]))
        result = meter.compute()
        for metric in ("Macro-F1", "F1@10", "F1@25", "F1@50", "Edit", "Boundary-F1", "ShortRecall"):
            self.assertAlmostEqual(result[metric], 100.0, msg=metric)
        wrong = SegmentationMeter(3)
        wrong.update(np.array([2, 2]), np.array([0, 0]))
        self.assertEqual(wrong.compute()["F1@50"], 0)

    def test_checkpoint_round_trip(self):
        seed_everything(7)
        model = BERG(small_config()).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, model, optimizer, 1, 0.0, ["a", "b", "c"])
            restored, names, _ = load_checkpoint(path, torch.device("cpu"))
            restored.eval()
            features = torch.ones(1, 4, 8)
            valid = torch.ones(1, 4, dtype=torch.bool)
            torch.testing.assert_close(model(features, valid)["logits"], restored(features, valid)["logits"])
            self.assertEqual(names, ["a", "b", "c"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
