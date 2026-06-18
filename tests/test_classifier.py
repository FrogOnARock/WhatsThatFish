"""
Tests for Classifier bookkeeping: checkpointing, loss CSV,
experiment registry, and best-model tracking in the train loop.

Pure — no DB or dataloaders. The trainer is constructed via object.__new__
to skip the DB-dependent __init__, then populated with a tiny model.
"""

import csv

import pytest
import torch
from torch import optim

from whatsthatfish.models.architecture.custom_resnet import BasicBlock, CustomResnet
from whatsthatfish.models.classifier import Classifier

NUM_LABELS = [5, 4, 3]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _bare_trainer(tmp_path) -> Classifier:
    """Trainer with all attributes train()/_save_checkpoint need, no DB."""
    t = object.__new__(Classifier)
    t.device = torch.device("cpu")
    t.model = CustomResnet(block=BasicBlock, layers=[1, 1, 1, 1], num_class=NUM_LABELS)
    t.optimizer = optim.AdamW(t.model.parameters(), lr=1e-3)
    t.lr_scheduler = optim.lr_scheduler.StepLR(t.optimizer, step_size=1)
    t.num_labels = NUM_LABELS
    t.model_version = "test_run"
    t.output_dir = tmp_path / "run"
    t.output_dir.mkdir()
    t.experiments_csv = tmp_path / "experiments.csv"
    t.lr = 1e-3
    t.max_lr = 1e-2
    t.weight_decay = 0.01
    t.epochs = 3
    t.batch_size = 2
    t.loss_weights = [0.6, 0.3, 0.1]
    # Variant-C defaults: no pretrained backbone, so train()'s progressive
    # freeze/unfreeze branches are no-ops for these checkpoint/CSV tests.
    t.pretrained = False
    t.freeze_epochs = 0
    # Discriminative-LR + topology + clip fields the registry now records.
    t.head_lr = 1e-3
    t.backbone_lr = 1e-4
    t.head_mode = "progressive"
    t.grad_clip = 1.0
    return t


def _stub_train_loop(trainer, val_losses: list[float]):
    """Replace the heavy per-epoch methods with deterministic stubs."""
    trainer.train_one_epoch = lambda epoch: (0.1, 0.1, 0.1, 0.3)
    it = iter(enumerate(val_losses))

    def fake_eval(epoch):
        i, loss = next(it)
        return {"val_loss": loss, "top3_species": 0.5 + i / 10}

    trainer.eval_one_epoch = fake_eval


# ════════════════════════════════════════════════════════════════════════════════
# _save_checkpoint
# ════════════════════════════════════════════════════════════════════════════════


class TestSaveCheckpoint:
    def test_writes_file(self, tmp_path):
        t = _bare_trainer(tmp_path)
        t._save_checkpoint(epoch=0, val_loss=1.23, filename="last.pt")
        assert (t.output_dir / "last.pt").exists()

    def test_checkpoint_contains_full_training_state(self, tmp_path):
        t = _bare_trainer(tmp_path)
        t._save_checkpoint(epoch=4, val_loss=0.5, filename="best.pt")
        ckpt = torch.load(t.output_dir / "best.pt", weights_only=False)
        assert set(ckpt) == {
            "epoch",
            "val_loss",
            "model",
            "optimizer",
            "lr_scheduler",
            "num_labels",
            "model_version",
        }
        assert ckpt["epoch"] == 4
        assert ckpt["val_loss"] == 0.5
        assert ckpt["num_labels"] == NUM_LABELS

    def test_model_state_loads_into_fresh_model(self, tmp_path):
        """num_labels in the checkpoint is enough to rebuild the model for serving."""
        t = _bare_trainer(tmp_path)
        t._save_checkpoint(epoch=0, val_loss=1.0, filename="best.pt")
        ckpt = torch.load(t.output_dir / "best.pt", weights_only=False)
        fresh = CustomResnet(
            block=BasicBlock, layers=[1, 1, 1, 1], num_class=ckpt["num_labels"]
        )
        fresh.load_state_dict(ckpt["model"])  # raises on any mismatch


# ════════════════════════════════════════════════════════════════════════════════
# train() best-model tracking
# ════════════════════════════════════════════════════════════════════════════════


class TestTrainLoopCheckpointing:
    def test_best_pt_keeps_lowest_val_loss(self, tmp_path):
        t = _bare_trainer(tmp_path)
        t._init_loss_csv()
        _stub_train_loop(t, val_losses=[1.0, 0.4, 0.7])
        t._train_loop()

        best = torch.load(t.output_dir / "best.pt", weights_only=False)
        last = torch.load(t.output_dir / "last.pt", weights_only=False)
        assert best["val_loss"] == 0.4
        assert best["epoch"] == 1
        assert last["val_loss"] == 0.7
        assert last["epoch"] == 2

    def test_registered_metrics_come_from_best_epoch_not_last(self, tmp_path):
        t = _bare_trainer(tmp_path)
        t._init_loss_csv()
        _stub_train_loop(t, val_losses=[1.0, 0.4, 0.7])
        t._train_loop()

        with open(t.experiments_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        # epoch 1 (val_loss 0.4) produced top3_species = 0.5 + 1/10
        assert float(rows[0]["top3_species"]) == 0.6
        assert float(rows[0]["best_val_loss"]) == 0.4

    def test_loss_csv_has_one_row_per_epoch(self, tmp_path):
        t = _bare_trainer(tmp_path)
        t._init_loss_csv()
        _stub_train_loop(t, val_losses=[1.0, 0.4, 0.7])
        t._train_loop()

        with open(t.output_dir / "losses.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert [float(r["val_loss"]) for r in rows] == [1.0, 0.4, 0.7]


# ════════════════════════════════════════════════════════════════════════════════
# CSV bookkeeping
# ════════════════════════════════════════════════════════════════════════════════


class TestCsvBookkeeping:
    def test_init_loss_csv_writes_header_once(self, tmp_path):
        t = _bare_trainer(tmp_path)
        t._init_loss_csv()
        t._init_loss_csv()  # idempotent — must not truncate or duplicate
        with open(t.output_dir / "losses.csv") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert lines[0].startswith("epoch,lr,")

    def test_register_experiment_appends_without_reheader(self, tmp_path):
        t = _bare_trainer(tmp_path)
        t._register_experiment({"top3_species": 0.6}, best_val_loss=0.4)
        t._register_experiment({"top3_species": 0.7}, best_val_loss=0.3)
        with open(t.experiments_csv) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows

    def test_register_experiment_logs_pretrained_lr_knobs(self, tmp_path):
        """The registry must record the LRs that actually drove the run."""
        t = _bare_trainer(tmp_path)
        t.pretrained = True
        t.head_lr = 0.0024
        t.backbone_lr = 0.0003
        t._register_experiment({"top1_species": 0.5}, best_val_loss=0.4)
        with open(t.experiments_csv) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["pretrained"] == "True"
        assert rows[0]["head_lr"] == "0.0024"
        assert rows[0]["backbone_lr"] == "0.0003"
        assert rows[0]["grad_clip"] == "1.0"
        assert rows[0]["top1_species"] == "0.5"

    def test_register_experiment_archives_old_schema(self, tmp_path):
        """A pre-existing registry with a different header is archived, not mangled."""
        t = _bare_trainer(tmp_path)
        # Simulate the legacy schema (old no-op-LR columns).
        t.experiments_csv.write_text(
            "model_version,timestamp,epochs,lr,max_lr,weight_decay\n"
            "old_run,2026-06-13,20,1e-5,1e-3,0.01\n"
        )
        t._register_experiment({"top1_species": 0.5}, best_val_loss=0.4)

        # Old file preserved under an archive name; new file has the new header.
        archives = list(tmp_path.glob("experiments_legacy_*.csv"))
        assert len(archives) == 1
        assert "old_run" in archives[0].read_text()
        with open(t.experiments_csv) as f:
            lines = f.readlines()
        assert lines[0].startswith("model_version,timestamp,epochs,pretrained,head_mode,")
        assert len(lines) == 2  # new header + the one new row


# ════════════════════════════════════════════════════════════════════════════════
# _curriculum_loss — the progressive loss-weight scheduler
#
# Pure arithmetic over (epoch, loss_phase, gate). These tests pin the behaviour
# that several subtle bugs broke during development:
#   - phase-local time must be (epoch - phase_start)/tc, not epoch/tc*phase
#   - the gate counter must RESET on a miss (consecutive, not cumulative)
#   - B must be clamped to [0, 1] so high coverage can't extrapolate past END
#   - val/best.pt uses fixed end-state weights (covered in eval tests, not here)
#   - early gate-driven transitions must re-anchor the next phase's clock
# ════════════════════════════════════════════════════════════════════════════════


# Schedule milestones (species, genus, family), matching the trainer defaults.
P1 = [0.0, 0.0, 1.0]  # start: pure family
P2 = [0.0, 0.6, 0.4]  # genus-focus
P3 = [0.6, 0.3, 0.1]  # species-dominant end state


class _FakeMetrics:
    """Stand-in for ClassificationMetrics exposing only the two gate fields the
    scheduler reads (per-class coverage, updated after each eval)."""

    def __init__(self, fam_gate: float = 0.0, genus_gate: float = 0.0):
        self.fam_gate = fam_gate
        self.genus_gate = genus_gate


def _curriculum_trainer(min_time_per_phase: int = 5) -> Classifier:
    """Minimal trainer carrying only the attributes _curriculum_loss touches.

    `min_time_per_phase` is now a trainer attribute (read as self.min_time_per_phase),
    not a _curriculum_loss kwarg — set it here per test.
    """
    t = object.__new__(Classifier)
    t.loss_phase = 1
    t.consecutive_epochs = 0
    t.phase_start_epoch = 0
    t.epochs = 40
    t.min_time_per_phase = min_time_per_phase
    t.loss_weights = list(P1)
    t.loss_weights_2 = list(P2)
    t.loss_weights_3 = list(P3)
    t.metrics = _FakeMetrics()
    return t


class TestCurriculumLoss:
    def test_phase1_starts_at_start_weights(self):
        t = _curriculum_trainer()
        w = t._curriculum_loss(epoch=0, time_constraint=20)
        assert w == pytest.approx(P1)

    def test_phase1_ramps_linearly_toward_p2(self):
        """Halfway through phase 1's time budget (no gate) → midpoint of P1→P2."""
        t = _curriculum_trainer()
        w = t._curriculum_loss(epoch=10, time_constraint=20)
        assert w == pytest.approx([0.0, 0.3, 0.7])  # 0.5*P1 + 0.5*P2

    def test_weights_sum_to_one_across_sweep(self):
        """Invariant: lerp of two sum-1 vectors stays sum-1, and clamping holds it."""
        t = _curriculum_trainer(min_time_per_phase=99)
        for e in range(0, 20):
            t.metrics.fam_gate = 0.5 if e % 2 else 0.0
            w = t._curriculum_loss(epoch=e, time_constraint=50)
            assert sum(w) == pytest.approx(1.0)
            assert all(0.0 <= x <= 1.0 for x in w)

    def test_transition_by_time_when_gate_cold(self):
        """Gate never fires → phase advances on the time budget alone (elapsed=tc)."""
        t = _curriculum_trainer()
        for e in range(0, 6):
            t._curriculum_loss(epoch=e, time_constraint=10)
        assert t.loss_phase == 1  # B_time < 1 before elapsed=10
        out = None
        for e in range(6, 11):
            out = t._curriculum_loss(epoch=e, time_constraint=10)
        assert t.loss_phase == 2  # crossed at epoch 10 (B_time>=1, elapsed>5)
        assert out == pytest.approx(P2)  # transition returns END

    def test_early_transition_by_sustained_gate(self):
        """3 consecutive gate passes (past min_time) advance the phase well before
        the time budget (tc=20) would have."""
        t = _curriculum_trainer()
        t.metrics.fam_gate = 0.95
        for e in range(0, 6):
            t._curriculum_loss(epoch=e, time_constraint=20)
        assert t.loss_phase == 1  # elapsed not yet > min_time at epoch 5
        t._curriculum_loss(epoch=6, time_constraint=20)
        assert t.loss_phase == 2  # elapsed=6>5 and consecutive>=3

    def test_non_consecutive_gate_does_not_transition(self):
        """Alternating pass/fail must never reach a streak of 3 (the +=/= bug)."""
        t = _curriculum_trainer(min_time_per_phase=2)
        for e in range(0, 20):
            t.metrics.fam_gate = 0.95 if e % 2 == 0 else 0.0  # never two in a row
            t._curriculum_loss(epoch=e, time_constraint=50)
        assert t.loss_phase == 1
        assert t.consecutive_epochs <= 1

    def test_high_coverage_does_not_extrapolate_past_end(self):
        """coverage=1.0 → gate/0.9≈1.11; B must clamp to 1 so no negative weights."""
        t = _curriculum_trainer(min_time_per_phase=10)
        t.metrics.fam_gate = 1.0
        w = t._curriculum_loss(epoch=0, time_constraint=20)
        assert w == pytest.approx(P2)  # exactly END, not beyond it
        assert all(x >= 0.0 for x in w)

    def test_early_exit_reanchors_next_phase_clock(self):
        """The phase_start_epoch fix: after an early phase-1 exit, phase 2's clock
        restarts from the transition epoch — so a hot genus gate can also exit
        early. With the old nominal offset, phase 2's elapsed would go negative and
        stall until the original fixed schedule caught up."""
        t = _curriculum_trainer()
        t.metrics.fam_gate = 0.95
        for e in range(0, 7):
            t._curriculum_loss(epoch=e, time_constraint=20)
        assert t.loss_phase == 2
        assert t.phase_start_epoch == 6  # anchored to the actual transition epoch

        t.metrics.fam_gate = 0.0
        t.metrics.genus_gate = 0.95
        for e in range(7, 13):
            t._curriculum_loss(epoch=e, time_constraint=20)
        # elapsed measured from epoch 6, so by epoch 12 (elapsed=6>5, streak>=3)
        # phase 2 also exits early — acceleration compounds instead of stalling.
        assert t.loss_phase == 3

    def test_phase3_short_circuits_to_end_state(self):
        t = _curriculum_trainer()
        t.loss_phase = 3
        w = t._curriculum_loss(epoch=0, time_constraint=10)
        assert w == pytest.approx(P3)
