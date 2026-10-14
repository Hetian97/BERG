<div align="center">

# BERG
### Budget-Constrained Temporal Refinement for Teacher Behavior Analysis

**Full-timeline prediction · Sparse deep updates · Time-indexed process graphs**

Python 3.10+ &nbsp; | &nbsp; PyTorch 2.6+ &nbsp; | &nbsp; Fixed visual features

</div>

---

## 🔬 Method

BERG predicts behavior categories and boundaries from fixed visual feature sequences. A lightweight temporal branch covers every observed step. EVD estimates refinement demand from local representation statistics and class probabilities. BER combines neighboring scores, a score threshold, and a per-window update capacity. Only selected positions pass through the deep branch. EPG aggregates predicted intervals into chronologically connected behavior nodes with timestamp pointers and consistency flags.

```text
Fixed visual features
         │
         ▼
Lightweight temporal branch ────────────────┐
         │                                 │
         ├── EVD scores → BER selection    │
         │                    │            │
         └── Temporal anchors ▼            │
                      Sparse deep updates  │
                              └── Gated fusion
                                      │
                             Class / boundary heads
                                      │
                             Interval aggregation
                                      │
                       Time-indexed behavior process graph
```

**Scope.** This is an offline model. Both the bidirectional scoring statistics and the temporal anchors use complete execution windows. Visual feature extraction is outside the temporal network. Supervised inputs are split at unknown labels; all modes split temporal gaps into independent runs. Inference does not use label values to define execution windows. Graph edges never cross runs.

## 📂 Source layout

| File | Responsibility |
| :--- | :--- |
| `config.py` | Validated model and loss settings |
| `data.py` | NPZ validation, split isolation, temporal runs, and batching |
| `model.py` | EVD, BER, sparse refinement, prediction heads, and EPG |
| `losses.py` | Joint segmentation, boundary, value, budget, coverage, and graph losses |
| `metrics.py` | Segment matching, edit distance, boundary matching, and pooled evaluation |
| `train.py` | Training, held-out evaluation, graph export, and runtime profiling |
| `tests.py` | Numerical equations, sparse execution, masking, and workflow checks |

## ⚙️ Installation

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tests.py
```

The CPU validation environment uses Python 3.12, PyTorch 2.6.0+cpu, and NumPy 2.2.6. CUDA execution requires a PyTorch build compatible with the local driver. The source package contains no pretrained weights or benchmark data.

## 🗂️ Data contract

Each independent source clip is stored as a numeric NPZ archive. Arrays use a common temporal grid within each run.

| Key | Shape / dtype | Meaning |
| :--- | :--- | :--- |
| `features` | `[T, F]`, floating point | Fixed visual embeddings |
| `timestamps` | `[T]`, floating point | Original time coordinates in seconds, strictly increasing |
| `labels` | `[T]`, integer | Class IDs in `[0, C)`; `-1` marks unknown labels |
| `boundaries` | `[T]`, binary numeric, optional | A transition before step `t`; the first step of each run has value zero |

Training and evaluation require `labels`. Inference accepts archives without labels. Transitions between known class labels are mandatory boundaries; explicit boundaries add consecutive events with the same category. No pickle objects are accepted. Original temporal indices and timestamps remain attached to every run.

The JSON manifest specifies the class order and experimental splits:

```json
{
  "schema_version": 1,
  "class_names": ["class_0", "class_1", "class_2"],
  "records": [
    {"path": "features/train_001.npz", "split": "train", "teacher_id": "teacher_a", "video_id": "video_a"},
    {"path": "features/val_001.npz", "split": "val", "teacher_id": "teacher_b", "video_id": "video_b"},
    {"path": "features/test_001.npz", "split": "test", "teacher_id": "teacher_c", "video_id": "video_c"}
  ]
}
```

Paths are relative to the manifest directory. A teacher or source video must occur in exactly one split. Repeated feature files and identifiers with surrounding whitespace are rejected. Supervised samples are split at unknown-label gaps and timestamp differences exceeding `1.5 * step_seconds`. The default temporal step is one second. Within-run timestamp differences must match `step_seconds`, allowing a relative tolerance of `1e-5` and an absolute tolerance of `1e-6` seconds.

**Benchmark access.** TBU currently exposes examples rather than its complete dataset. MM-TBA releases anonymized visual feature files and action annotations. Obtain data through the maintainers, preserve their usage conditions, and convert numeric feature arrays into the NPZ contract before running experiments. This package does not distribute classroom recordings or infer access permissions.

```text
TBU: https://github.com/cai-KU/TBU
MM-TBA: https://doi.org/10.6084/m9.figshare.28942487
```

## 🚀 Experimental workflow

### Train

```bash
python train.py train --manifest data/manifest.json --output runs/berg_seed2022 --seed 2022 --epochs 50 --batch-size 4 --device cpu
```

The output directory must be empty. Checkpoints `best.pt` and `last.pt`, configuration and environment metadata, and the epoch history are written to this directory. Model selection uses validation F1@50 only. Test records are excluded from training and checkpoint selection. For CUDA, set `--device cuda`; `--amp` enables automatic mixed precision for training on CUDA.

Optional settings can be supplied with `--config settings.json`. `input_dim` and `num_classes` are inferred from the data unless explicitly supplied; conflicting dimensions are rejected.

```json
{
  "hidden_dim": 256,
  "num_anchors": 16,
  "deep_layers": 3,
  "budget_ratio": 0.35,
  "neighbor_weight": 0.30,
  "neighbor_radius": 2,
  "review_threshold": 0.65
}
```

### Evaluate a held-out split

```bash
python train.py evaluate --manifest data/manifest.json --checkpoint runs/berg_seed2022/best.pt --split test --output runs/berg_seed2022/test.json --device cpu
```

Metrics include Macro-F1, F1@10/25/50, Edit, Boundary-F1, ShortRecall, and the deep-update activation percentage (`Trigger`). Segment matching uses same-class IoU and one-to-one matching. Explicit predicted boundaries are retained even when adjacent events have the same class. Boundary matching allows one time step of error. ShortRecall uses events of at most three steps and an IoU threshold of 0.5. Macro-F1 includes all declared classes; classes without support contribute zero. Segment and boundary counts are pooled across runs; Edit is averaged over runs. Empty evidence is reported as zero rather than a perfect score.

### Export a process graph

```bash
python train.py predict --features data/features/test_001.npz --checkpoint runs/berg_seed2022/best.pt --output runs/berg_seed2022/graph.json --device cpu
```

Each run has its own node and edge list. Nodes contain half-open index and time intervals, a class label, consistency score, review flag, and timestamp pointers. Low-consistency nodes remain in the graph. Pointer ordering follows descending interval weight with stable temporal-index tie breaking. The graph node classifier and the framewise classifier are distinct heads.

### Measure temporal-network runtime

```bash
python train.py profile --features data/features/test_001.npz --checkpoint runs/berg_seed2022/best.pt --output runs/berg_seed2022/profile.json --warmup 10 --repeats 50 --device cpu
python train.py profile --features data/features/test_001.npz --checkpoint runs/berg_seed2022/best.pt --output runs/berg_seed2022/profile_full.json --fully-deep --device cpu
```

Runtime includes temporal prediction and process graph construction; feature encoding and disk access are excluded. The fully deep mode forces every observed position through the same trained deep branch. It is a computational control, not a separately trained baseline. Increasing the budget to one without removing the score threshold does not force full execution. Empirical latency depends on sequence length, hardware, and tensor-kernel overhead; a lower update count alone is not evidence of lower wall-clock latency.

## 📐 Equation-to-code alignment

| Manuscript equations | Implemented computation |
| :--- | :--- |
| (1) | Detached lightweight error and boundary target; five-feature sigmoid position scorer |
| (2) | Neighbor maximum excluding self, threshold screening, and `ceil(beta * T)` stable top-K capacity |
| (3) | Selected deep updates with unchanged lightweight representations elsewhere |
| (4) | Masked exponential temporal pooling with equally spaced anchor centers |
| (5)–(7) | Interval attention with positive selection floor; shared weights for node labels, consistency, and pointers |
| (8) | Chronological edges within each independent run |
| (9)–(10) | Per-window soft-budget penalty and numerically stable boundary coverage probability |
| (11)–(12) | Ground-truth interval supervision and weighted joint objective |

### Explicit implementation settings

The manuscript leaves several operators and hyperparameters unspecified. The following choices make the computational contract complete and inspectable:

| Component | Setting |
| :--- | :--- |
| Lightweight branch | Linear input projection; residual temporal convolution of width 3; GELU and LayerNorm |
| EVD statistics | Adjacent cosine distance, local-mean cosine deviation, normalized class entropy, mapped neighbor cosine redundancy, and bidirectional probability distance |
| Boundary target | Binary annotated interval starts, excluding the first step of each run |
| Soft gate | `sigmoid((adjusted_score - selection_threshold) / gate_temperature)` |
| Sparse training gradient | Straight-through selection multiplier on selected updates; zero deep delta at all unselected positions |
| Deep branch | Three pre-normalized selected-query-to-anchor attention blocks with feed-forward expansion 4 |
| Anchor coordinates | Float32 valid time-step indices, centers, and normalized weights; default pooling scale 16 steps |
| Interval query | Linear projection of the mean interval representation |
| Interval decoding | Float32 sigmoid thresholding; split at every class change or boundary probability at least 0.5 |
| Selection | Threshold 0.5; temperature 0.1; stable ties resolved by lower temporal index |
| Graph pooling | Positive floor 0.1; up to three timestamp pointers per interval |
| Joint losses | Unit weights; graph consistency coefficient 0.1; lightweight and fused class cross-entropy |
| Optimization | AdamW; learning rate 0.0003; weight decay 0.01; gradient-norm clipping at 1 |

All settings are recorded in `run.json` and checkpoints. Reported manuscript scores and parameter counts are not asserted by this source package. Establish benchmark performance from authorized datasets, documented splits, and independently executed multi-seed experiments. Graph output is implemented; the manuscript's Node-F1, Edge-F1, Coverage, and UnsupportedRate protocols are not included in the metric module because their full matching conventions are not specified.

## ✅ Verification and reproducibility

`tests.py` checks equations against direct numerical calculations, update capacities and thresholds, actual deep-token execution, padding invariance, detached score targets, finite gradients, explicit same-class boundaries, original timestamps, split isolation, and checkpoint round trips. It uses temporary numerical fixtures solely to exercise software contracts. These checks are not benchmark experiments.

Three independent review-and-repair cycles were completed across the method, data pipeline, evaluation workflow, and documentation:

| Cycle | Findings addressed | Verification |
| :--- | :--- | :--- |
| 1 | AMP-safe position loss, exact straight-through arithmetic, loss masks, consistent boundary targets, CUDA workspace configuration, and invalid sample handling | 21 numerical and workflow checks passed after repair |
| 2 | Float32 EVD statistics and long-sequence anchor coordinates, group-ID validation, sampling-grid validation, scalar types, label overflow, and checkpoint vocabulary | 26 regression checks passed; training, held-out evaluation, label-free prediction, and profiling completed |
| 3 | Float32 boundary thresholding and graph consistency arithmetic; complete gradient-clipping metadata | 27 final regression checks and 56 independent data checks passed; repeated CPU training states and histories matched exactly |

Verification used the CPU environment listed above, including CPU float16 and bfloat16 autocast numerical paths. CUDA execution and empirical benchmark scores have not been validated in this environment. Runtime measurements from numerical fixtures are excluded from scientific performance claims.

For empirical studies, retain the full feature preprocessing description, teacher/video identifiers, label vocabulary, manifest, random seeds, and runtime environment. The manuscript lists seeds 2022–2026. Repeat independent training runs with those seeds and report measured variability. Deterministic PyTorch operations are requested and the cuBLAS workspace setting is recorded, but identical results across different PyTorch versions and hardware are not guaranteed. Position-value supervision operates on logits, while selection scores, coverage gates, interval weights, consistency probabilities, and boundary thresholding remain in float32 during mixed-precision training.

---

<div align="center">
<sub>BERG · Temporal refinement under explicit computational budgets</sub>
</div>
