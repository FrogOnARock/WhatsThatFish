"""Post-train promotion gate: does a freshly trained model earn the GCS upload?

Two conditions, composed (see the design discussion):
  * absolute FLOOR — never promote a model below the objective bar. On the FIRST
    run (no incumbent) this is the whole gate (bootstrap).
  * NO-REGRESSION — with an incumbent, every gated metric must be >= incumbent
    minus a small epsilon (promotes lateral/beneficial updates, blocks backslides
    and noise-level thrash).

Cross-run comparison is valid ONLY because both models are evaluated on the FIXED
geographic val split (CLAUDE.md: never re-split). Metrics are stored alongside the
weights so the next run reads the incumbent's numbers without re-running it.

Higher-is-better is assumed for every metric here (accuracies, mAP, recall).
"""

import json
import shutil
from pathlib import Path

from ..config import _get_logger

logger = _get_logger(__name__)

DEFAULT_EPSILON = 0.005  # 0.5pp regression tolerance (noise guard)

# ── Detector: the metrics we gated on pre-test-suite (targets == absolute floor) ──
DETECTOR_FLOOR = {"mAP@0.5": 0.75, "mAP@0.5:0.95": 0.50, "Recall@0.5": 0.90}
DETECTOR_KEYS = list(DETECTOR_FLOOR)

# ── Classifier: six macro GEOGRAPHIC held-out metrics; bootstrap floor is species
#    top-1 alone (headline target 0.65 — bump to 0.68 to tighten). ──
CLASSIFIER_SPECIES_TOP1_FLOOR = 0.65
CLASSIFIER_FLOOR = {"geo_top1_species": CLASSIFIER_SPECIES_TOP1_FLOOR}
CLASSIFIER_KEYS = [
    "geo_top1_species",
    "geo_top3_species",
    "geo_top1_genus",
    "geo_top3_genus",
    "geo_top1_family",
    "geo_top3_family",
]


def _eps(epsilon, key):
    return epsilon.get(key, 0.0) if isinstance(epsilon, dict) else epsilon


def should_promote(new, incumbent, floor, regression_keys, epsilon=DEFAULT_EPSILON):
    """Decide promotion. Pure — no I/O. Returns (promote: bool, reasons: dict).

    `new` / `incumbent` are metric->value dicts (incumbent None on the first run).
    Floor is checked always; no-regression only when an incumbent exists.
    """
    reasons = {}
    ok = True

    for key, min_val in floor.items():
        val = new.get(key)
        passed = val is not None and val >= min_val
        reasons[f"floor:{key}"] = {"value": val, "min": min_val, "pass": passed}
        ok = ok and passed

    if incumbent is None:
        reasons["bootstrap"] = True  # no incumbent → floor-only gate
        return ok, reasons

    for key in regression_keys:
        val, prev = new.get(key), incumbent.get(key)
        # A missing incumbent metric imposes no constraint for that key.
        passed = val is not None and (prev is None or val >= prev - _eps(epsilon, key))
        reasons[f"regress:{key}"] = {
            "new": val,
            "incumbent": prev,
            "eps": _eps(epsilon, key),
            "pass": passed,
        }
        ok = ok and passed

    return ok, reasons


class PromotionStore:
    """Reads the incumbent's metrics and promotes a new checkpoint + metrics.

    Local-filesystem default: ``root/<model>/`` holds the weights + metrics.json.
    On the training VM, inject a GCS-backed subclass overriding load_metrics /
    promote with `gcloud storage` equivalents (bumping the 'latest-good' object).
    """

    def __init__(self, root):
        self.root = Path(root)

    def _dir(self, model_name):
        return self.root / model_name

    def load_metrics(self, model_name):
        p = self._dir(model_name) / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else None

    def promote(self, model_name, weights_path, metrics):
        d = self._dir(model_name)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(weights_path, d / Path(weights_path).name)
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
        logger.info("Promoted %s → %s", model_name, d)


def gate_and_promote(
    model_name,
    new_metrics,
    weights_path,
    floor,
    regression_keys,
    store,
    epsilon=DEFAULT_EPSILON,
):
    """Load incumbent → decide → promote-or-skip. Returns the promote decision."""
    incumbent = store.load_metrics(model_name)
    promote, reasons = should_promote(
        new_metrics, incumbent, floor, regression_keys, epsilon
    )
    for k, v in reasons.items():
        logger.info("gate %s: %s", k, v)
    if promote:
        gated = {k: new_metrics.get(k) for k in [*floor, *regression_keys]}
        store.promote(model_name, weights_path, gated)
    else:
        logger.warning("%s NOT promoted — keeping incumbent", model_name)
    return promote
