"""
Tests for CustomResnetTrainer bookkeeping: checkpointing, loss CSV,
experiment registry, and best-model tracking in the train loop.

Pure — no DB or dataloaders. The trainer is constructed via object.__new__
to skip the DB-dependent __init__, then populated with a tiny model.
"""

import csv

import torch
from torch import optim

from whatsthatfish.models.c_custom_resnet import BasicBlock, CustomResnet
from whatsthatfish.training.oc_training import CustomResnetTrainer

NUM_LABELS = [5, 4, 3]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _bare_trainer(tmp_path) -> CustomResnetTrainer:
    """Trainer with all attributes train()/_save_checkpoint need, no DB."""
    t = object.__new__(CustomResnetTrainer)
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
    return t


def _stub_train_loop(trainer, val_losses: list[float]):
    """Replace the heavy per-epoch methods with deterministic stubs."""
    trainer.train_one_epoch = lambda: (0.1, 0.1, 0.1, 0.3)
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
        t.train()

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
        t.train()

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
        t.train()

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
