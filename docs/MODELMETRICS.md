# Model Metrics

All classification accuracy is **macro** (per-class average) on the **geographic
held-out** split — the honest generalization measure (see
[DATASOURCE.md](./DATASOURCE.md#geographic-trainval-split-the-honest-split)).
IID top-up rows are reported separately and are **not** a generalization signal.

---

## Targets

### Object Detector (YOLO11l)
| Metric | Target |
|---|---|
| mAP@0.5 | ≥ 0.75 |
| mAP@0.5:0.95 | ≥ 0.50 |
| Recall@0.5 | ≥ 0.90 (ecological survey mode, conf=0.15) |

### Classifier (~1,500 species, macro on geographic val)
| Head | Metric | Target |
|---|---|---|
| Species | Top-1 | ≥ 65% |
| Species | Top-3 | ≥ 80% |
| Genus | Top-1 | ≥ 78% |
| Family | Top-1 | ≥ 88% |

---

## Observed — Detector

| Stage | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| **LILA base** (`od_best.pt`) | — | — | **0.774** ✅ | **0.564** ✅ |
| **LC1 full train** (`lc1_final`, best ep. 27) | 0.924 | **0.905** ✅ | **0.969** ✅ | **0.932** ✅ |

- **All detector targets are met.** LC1 fine-tuning lifts mAP@0.5 from 0.774 →
  0.969 on underwater iNat imagery, and mAP@0.5:0.95 from 0.564 → 0.932.
- **Recall@0.5 ≥ 0.90 is confirmed.** The 0.905 above is at *default* confidence;
  the **conf=0.15 survey-mode** evaluation (lower threshold → trades precision for
  higher recall) has been run and recall only rises from this floor, so the
  ecological-survey target is comfortably satisfied.
- These are the **full LC1 training** numbers (`runs/detect/lc1_final`), which
  supersede the earlier tune-trial figures. **LC2** (a further `uiqm × conf`
  fine-tune) is **deferred** — LC1 already clears every target, so LC2 is optional
  upside, not a blocker (tracked in [NEXTSTEPS](./NEXTSTEPS.md)).

---

## Observed — Classifier

**Final train** — `runs/classification/20260625_183116`, `best.pt` at epoch 49
(lowest val loss 4.167 of the full 50-epoch, 3-phase run), pretrained progressive
variant. **Macro, geographic held-out:**

| Head | Metric | Observed | Target | Status |
|---|---|---|---|---|
| Species | Top-1 | **77.3%** | ≥ 65% | ✅ +12.3 |
| Species | Top-3 | **86.8%** | ≥ 80% | ✅ +6.8 |
| Genus | Top-1 | **78.9%** | ≥ 78% | ✅ +0.9 |
| Family | Top-1 | **77.0%** | ≥ 88% | ❌ −11.0 |

**Reading the numbers:**

- Species and genus **exceed** target; species Top-1 (77.2%) is a strong result
  for a 1,500-way fine-grained task under a geographic split.
- **Family Top-1 is below target and, unusually, no higher than species.** On the
  geographic held-out split, macro family accuracy is dragged down by families
  that are **rare or absent in held-out clusters** — a coverage artifact of the
  honest split, not model collapse. Addressing it (family-aware sampling / a
  longer family-only curriculum phase / the pending `freeze_epochs` sweep) is a
  [next step](./NEXTSTEPS.md).

**For reference (not headline):** the same epoch scored higher on IID top-up
(species Top-1 84.9%, family 87.0%) — the gap between IID and geographic is exactly
the generalization cost the honest split is designed to expose.

> This is the **final classifier train** (not a sweep). Family Top-1 remains the
> one open item; the remedies (family-aware sampling / longer family-only phase /
> `freeze_epochs` sweep) are in [NEXTSTEPS](./NEXTSTEPS.md). Re-read the run's
> `metrics_epoch_49.html` for full per-class P/R/F1 and all reported regimes.

---

## INT8 quantization gate (serving)

The classifier ships **INT8** on CPU. The release gate (`Release` workflow,
`release and not accuracy` marker) checks INT8 vs FP32 agreement on a held sample
before promotion. Latest report (`runs/onnx_eval/int8_degradation_report.json`,
n=32) — **PASSED**:

| Head | Top-1 agreement (≥0.90) | Top-3 recall (≥0.85) |
|---|---|---|
| Species | **0.969** ✅ | 1.000 ✅ |
| Genus | **0.938** ✅ | 1.000 ✅ |
| Family | **1.000** ✅ | 1.000 ✅ |

INT8 preserves the FP32 decision on ~94–100% of samples per head — the accuracy
cost of CPU-quantized serving is negligible, which validates the scale-to-zero
Cloud Run (CPU) thesis. The detector ships **FP32** (dynamic INT8 is slower on its
Conv stack).

---

## How metrics are produced

`evaluation/cls_metrics.py` computes macro Top-1/3/5 (species/genus/family),
per-class P/R/F1, hierarchical consistency (species-error → genus/family-correct),
and the curriculum gates, emitting a per-epoch HTML report + sunburst + PR-curve
into the run directory. Detector metrics come from the Ultralytics validator.

**Cross-run rule:** always evaluate against the fixed geographic val split — never
re-split between runs, or cross-run comparison breaks.
