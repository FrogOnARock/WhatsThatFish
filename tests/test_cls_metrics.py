"""
Tests for ClassificationMetrics: top-k accuracy, hierarchical consistency,
accumulate/reset lifecycle, and per-epoch report files.

Pure — the DB-backed taxonomy lookup is injected as a DataFrame via
object.__new__, skipping __init__.
"""

import pandas as pd
import torch

from whatsthatfish.evaluation.cls_metrics import ClassificationMetrics

# 5 species → 4 genera → 3 subfamilies (≥3 per head so label_binarize
# in the PR curves never hits the binary single-column special case)
TAXONOMY = pd.DataFrame(
    {
        "species_idx": [0, 1, 2, 3, 4],
        "genus_idx": [0, 0, 1, 2, 3],
        "family_idx": [0, 0, 0, 1, 2],
        "species_name": ["sp_a", "sp_b", "sp_c", "sp_d", "sp_e"],
        "genus_name": ["ge_a", "ge_a", "ge_b", "ge_c", "ge_d"],
        "family_name": ["sf_a", "sf_a", "sf_a", "sf_b", "sf_c"],
    }
)


def _bare_metrics(tmp_path) -> ClassificationMetrics:
    m = object.__new__(ClassificationMetrics)
    m.output_dir = tmp_path
    m._taxonomy = TAXONOMY
    # plot_every=1 → write the heavy reports every epoch (the lifecycle tests
    # assert the files appear); normally set from config (default 5).
    m.plot_every = 1
    m.reset()
    return m


def _one_hot_logits(labels: list[int], num_classes: int) -> torch.Tensor:
    """Logits whose argmax is exactly `labels` — fully deterministic preds."""
    logits = torch.full((len(labels), num_classes), -5.0)
    for i, lbl in enumerate(labels):
        logits[i, lbl] = 5.0
    return logits


# ════════════════════════════════════════════════════════════════════════════════
# _top_k_accuracy
# ════════════════════════════════════════════════════════════════════════════════


class TestTopKAccuracy:
    def test_top1_exact(self, tmp_path):
        m = _bare_metrics(tmp_path)
        logits = _one_hot_logits([0, 1, 2, 2], num_classes=5)
        targets = torch.tensor([0, 1, 2, 3])  # 3 of 4 correct
        assert m._top_k_accuracy(logits, targets, 1) == 0.75

    def test_topk_counts_target_anywhere_in_top_k(self, tmp_path):
        m = _bare_metrics(tmp_path)
        # target class 1 always second-highest → top-1 = 0, top-3 = 1
        logits = torch.tensor([[5.0, 3.0, 0.0, -1.0, -2.0]] * 4)
        targets = torch.tensor([1, 1, 1, 1])
        assert m._top_k_accuracy(logits, targets, 1) == 0.0
        assert m._top_k_accuracy(logits, targets, 3) == 1.0


# ════════════════════════════════════════════════════════════════════════════════
# _hierarchical_consistency
# ════════════════════════════════════════════════════════════════════════════════


class TestHierarchicalConsistency:
    def test_rates_use_error_set_as_denominator(self, tmp_path):
        m = _bare_metrics(tmp_path)
        # 4 samples: species wrong on the last two; genus right on exactly
        # one of those two → species_wrong_genus_right = 0.5
        logits = {
            "species": _one_hot_logits([0, 1, 0, 0], 5),
            "genus": _one_hot_logits([0, 0, 1, 3], 4),
            "family": _one_hot_logits([0, 0, 0, 0], 3),
        }
        targets = {
            "species": torch.tensor([0, 1, 2, 3]),
            "genus": torch.tensor([0, 0, 1, 2]),
            "family": torch.tensor([0, 0, 0, 1]),
        }
        out = m._hierarchical_consistency(logits, targets)
        assert out["species_wrong_genus_right"] == 0.5

    def test_no_errors_returns_one(self, tmp_path):
        m = _bare_metrics(tmp_path)
        labels = [0, 1, 2]
        logits = {
            "species": _one_hot_logits(labels, 5),
            "genus": _one_hot_logits(labels, 4),
            "family": _one_hot_logits(labels, 3),
        }
        targets = {k: torch.tensor(labels) for k in logits}
        out = m._hierarchical_consistency(logits, targets)
        assert out["species_wrong_genus_right"] == 1.0
        assert out["genus_wrong_family_right"] == 1.0


# ════════════════════════════════════════════════════════════════════════════════
# update / compute lifecycle
# ════════════════════════════════════════════════════════════════════════════════


def _feed_two_batches(m):
    for labels in ([0, 1, 2], [3, 4, 0]):
        m.update(
            out_species=_one_hot_logits(labels, 5),
            out_genus=_one_hot_logits([min(l, 3) for l in labels], 4),
            out_family=_one_hot_logits([min(l, 2) for l in labels], 3),
            target={
                "species": torch.tensor(labels),
                "genus": torch.tensor([min(l, 3) for l in labels]),
                "family": torch.tensor([min(l, 2) for l in labels]),
            },
        )


class TestComputeLifecycle:
    def test_compute_returns_topk_and_consistency_keys(self, tmp_path):
        m = _bare_metrics(tmp_path)
        _feed_two_batches(m)
        out = m.compute(epoch=0)
        assert out["top3_species"] == 1.0  # all predictions exact
        assert {
            "top3_species",
            "top5_species",
            "top3_genus",
            "species_wrong_genus_right",
            "species_wrong_family_right",
            "genus_wrong_family_right",
        } <= set(out)

    def test_compute_writes_all_three_report_files(self, tmp_path):
        m = _bare_metrics(tmp_path)
        _feed_two_batches(m)
        m.compute(epoch=7)
        assert (tmp_path / "metrics_epoch_7.html").exists()
        assert (tmp_path / "sunburst_epoch_7.html").exists()
        assert (tmp_path / "pr_curves_epoch_7.png").exists()

    def test_compute_resets_state_for_next_epoch(self, tmp_path):
        """No cross-epoch leakage — accumulators must be empty after compute."""
        m = _bare_metrics(tmp_path)
        _feed_two_batches(m)
        m.compute(epoch=0)
        assert all(len(v) == 0 for v in m._logits.values())
        assert all(len(v) == 0 for v in m._targets.values())

    def test_update_accumulates_across_batches(self, tmp_path):
        m = _bare_metrics(tmp_path)
        _feed_two_batches(m)
        assert len(m._logits["species"]) == 2
        assert torch.cat(m._logits["species"]).shape == (6, 5)
