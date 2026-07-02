"""Model-release gate suite (items 3, 4, 6, 6b) — the tests CI must pass before
an ONNX artifact is promoted into the serving image.

All `release`-marked: they need the real trained weights (and, for the detector
accuracy gate, the DB val split), so they run in the `v*` release job, not on
every push. Structure:

  * TestDetectorAccuracyGate  — Recall@0.5 / mAP targets on the checkpoint (item 3)
  * TestClassifierFp32Parity  — ONNX-FP32 == torch: logits + top-1/top-3 (item 4)
  * TestClassifierInt8Agree   — ONNX-INT8 vs ONNX-FP32 top-1/top-3 rate  (item 4)
  * TestInt8Degradation       — accuracy diff + JSON artifact for review (item 6/6b)

onnxruntime is imported lazily so `--collect-only` works even where the runtime
can't load. Parity runs torch on CPU so it compares against CPU onnxruntime math.

NOTE: the sample batch is seeded synthetic input — it exercises the full compare
machinery and produces the artifact anywhere the checkpoint loads. For the most
representative certification, point `sample_batch` at real val crops (a batch from
`class_dataloader(split="val")`); the agreement math is identical.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from whatsthatfish.models.classifier import Classifier
from whatsthatfish.models.detection import Detector, Dataset

pytestmark = pytest.mark.release

_WEIGHTS = Path(__file__).parents[1] / "src" / "whatsthatfish" / "weights"
CKPT = _WEIGHTS / "classifier_best.pt"

# ── gate thresholds (tune to your quality bar) ──────────────────────────────
FP32_LOGIT_ATOL = 1e-3       # ONNX-FP32 vs torch: export must be near-exact
TOP1_AGREEMENT_MIN = 0.99    # INT8 vs FP32 top-1 must still match this often
TOP3_AGREEMENT_MIN = 0.99    # INT8 vs FP32 top-3 SET must match this often
N_SAMPLES = 32

ARTIFACT = Path(__file__).parents[1] / "runs" / "onnx_eval" / "int8_degradation_report.json"


def _session(path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """(N, k) top-k class indices per row, descending — matches the serving top-k."""
    return np.argsort(-logits, axis=1)[:, :k]


def _top1_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((_top_k(a, 1)[:, 0] == _top_k(b, 1)[:, 0]).mean())


def _top3_set_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of rows whose top-3 SETS are identical (order-insensitive, as the
    frontend shows an unordered trio and near-ties may legitimately swap)."""
    ta, tb = _top_k(a, 3), _top_k(b, 3)
    return float(np.mean([set(ra) == set(rb) for ra, rb in zip(ta, tb)]))


@pytest.fixture(scope="module")
def sample_batch() -> np.ndarray:
    g = torch.Generator().manual_seed(0)
    return torch.rand(N_SAMPLES, 5, 320, 320, generator=g).numpy().astype(np.float32)


@pytest.fixture(scope="module")
def torch_model():
    clf = Classifier()
    clf.device = torch.device("cpu")  # CPU so parity compares against CPU onnxruntime
    return clf._load_for_predict(CKPT)


@pytest.fixture(scope="module")
def onnx_paths(tmp_path_factory):
    out = tmp_path_factory.mktemp("release_onnx") / "classifier.onnx"
    return Classifier().export(weights=CKPT, out=out, int8=True)


@pytest.fixture(scope="module")
def torch_logits(torch_model, sample_batch):
    with torch.no_grad():
        outs = torch_model(torch.from_numpy(sample_batch))
    return [o.cpu().numpy() for o in outs]  # [species, genus, family]


@pytest.fixture(scope="module")
def fp32_logits(onnx_paths, sample_batch):
    return _session(onnx_paths["fp32"]).run(None, {"input": sample_batch})


@pytest.fixture(scope="module")
def int8_logits(onnx_paths, sample_batch):
    return _session(onnx_paths["int8"]).run(None, {"input": sample_batch})


HEADS = ("species", "genus", "family")


# ── item 3: detector accuracy gate ──────────────────────────────────────────
@pytest.mark.accuracy  # needs the full val DB split → data-bearing env, not vanilla CI
class TestDetectorAccuracyGate:
    def test_recall_and_map_targets(self):
        _, passed = Detector(dataset=Dataset.LC1).evaluate()
        failed = [k for k, ok in passed.items() if not ok]
        assert not failed, f"Detector missed release targets: {failed}"


# ── item 4: FP32 parity (ONNX vs torch) ─────────────────────────────────────
class TestClassifierFp32Parity:
    def test_logits_match_torch(self, torch_logits, fp32_logits):
        for name, t, o in zip(HEADS, torch_logits, fp32_logits):
            np.testing.assert_allclose(
                o, t, atol=FP32_LOGIT_ATOL, err_msg=f"{name} logits diverge"
            )

    def test_top1_and_top3_match_torch(self, torch_logits, fp32_logits):
        for name, t, o in zip(HEADS, torch_logits, fp32_logits):
            assert _top1_agreement(t, o) == 1.0, f"{name} top-1 mismatch"
            assert _top3_set_agreement(t, o) == 1.0, f"{name} top-3 set mismatch"


# ── item 4: INT8 agreement (ONNX-INT8 vs ONNX-FP32) ─────────────────────────
class TestClassifierInt8Agreement:
    def test_top1_agreement(self, fp32_logits, int8_logits):
        for name, f, q in zip(HEADS, fp32_logits, int8_logits):
            rate = _top1_agreement(f, q)
            assert rate >= TOP1_AGREEMENT_MIN, f"{name} top-1 agreement {rate:.4f}"

    def test_top3_agreement(self, fp32_logits, int8_logits):
        for name, f, q in zip(HEADS, fp32_logits, int8_logits):
            rate = _top3_set_agreement(f, q)
            assert rate >= TOP3_AGREEMENT_MIN, f"{name} top-3 agreement {rate:.4f}"


# ── item 6 + 6b: INT8 degradation report (asserts + reviewable artifact) ─────
class TestInt8Degradation:
    def test_degradation_within_bounds_and_write_artifact(
        self, fp32_logits, int8_logits
    ):
        report = {
            "thresholds": {
                "top1_agreement_min": TOP1_AGREEMENT_MIN,
                "top3_agreement_min": TOP3_AGREEMENT_MIN,
            },
            "n_samples": N_SAMPLES,
            "heads": {},
        }
        ok = True
        for name, f, q in zip(HEADS, fp32_logits, int8_logits):
            t1 = _top1_agreement(f, q)
            t3 = _top3_set_agreement(f, q)
            diff = np.abs(f - q)
            report["heads"][name] = {
                "top1_agreement": t1,
                "top3_agreement": t3,
                "mean_abs_logit_diff": float(diff.mean()),  # diagnostic, not a gate
                "max_abs_logit_diff": float(diff.max()),
            }
            ok = ok and t1 >= TOP1_AGREEMENT_MIN and t3 >= TOP3_AGREEMENT_MIN
        report["passed"] = ok

        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(report, indent=2))

        assert ok, f"INT8 degradation exceeded bounds — see {ARTIFACT}"
