"""Train and tune the 5-channel ResNet-34 hierarchical fish classifier.

CustomResnetTrainer runs a 3-phase taxonomic curriculum (family → +genus → all
three heads), with discriminative LRs (fast heads/stem, ~10x slower backbone),
OneCycleLR, and a progressive backbone unfreeze after a warmup. CustomResnetTuner
wraps it in an in-process random search over the param-space YAML (no Ray workers,
to stay within a single GPU's memory).
"""

from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from torch import nn
from torch import optim
import torch
import csv
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func
from collections import defaultdict
import time
import yaml
from ray import tune

from ..database.models import InatClassificationDataset
from ..models.loaders.c_dataloader import class_dataloader
from ..models.c_custom_resnet import CustomResnet, BasicBlock
from ..database.config import get_session_factory
from ..evaluation.cls_metrics import ClassificationMetrics
from ..config import _get_logger

logger = _get_logger(__name__)


class CustomResnetTrainer:
    """Owns the full classifier training loop, model, optimizer and metrics.

    Reads its hyperparameters from the YAML config (overridable by kwargs during
    tuning), builds the class-weighted losses and discriminative optimizer, and
    drives the curriculum, checkpointing and the per-run/global CSV registries.
    """

    def __init__(
        self,
        config: Path = Path(__file__).parents[1] / "config/cls_model_config.yaml",
        session_maker: sessionmaker = None,
        tuning: bool = False,
        **kwargs,
    ):
        # Lazy default — get_session_factory() at definition time would make the
        # module unimportable without DATABASE_URL set.
        session_maker = session_maker or get_session_factory()

        if config:
            if config.suffix not in (".yaml", ".yml"):
                raise ValueError("Config must be a yaml file.")
            with open(config, "r") as f:
                data = yaml.safe_load(f)

            # Tunable parameters
            self.lr = kwargs.get("lr", data.get("lr", 0.001))
            self.weight_decay = kwargs.get(
                "weight_decay", data.get("weight_decay", 0.01)
            )
            self.epochs = 20 if tuning else data.get("epochs", 50)
            self.warmup_pct = kwargs.get("warmup_pct", data.get("warmup_pct", 0.05))
            self.max_lr = kwargs.get("max_lr", data.get("max_lr", 0.01))
            self.min_time_per_phase = kwargs.get(
                "min_time_per_phase", data.get("min_time_per_phase", 10)
            )

            # Tunable parameters - pretrained
            self.backbone_lr = kwargs.get("backbone_lr", data.get("backbone_lr", 1e-4))
            self.head_lr = kwargs.get("head_lr", data.get("head_lr", 1e-3))
            # Global grad-norm clip — caps the per-step update so a hot-LR trial
            # degrades gracefully instead of exploding (the species-loss=577 blow-up
            # at the backbone unfreeze). <= 0 disables clipping.
            self.grad_clip = kwargs.get("grad_clip", data.get("grad_clip", 1.0))

            # Fixed parameters
            self.loss_weights = data.get("loss_weights", [0.0, 0.0, 1.0])
            self.loss_weights_2 = data.get("loss_weights_p2", [0.0, 0.6, 0.4])
            self.loss_weights_3 = data.get("loss_weights_p3", [0.6, 0.3, 0.1])
            self.batch_size = data.get("batch_size", 16)
            # Model variant (A/B/C harness) + fine-tune schedule
            self.pretrained = data.get("pretrained", False)
            self.in_dim = data.get("in_dim", 5)
            self.layers = data.get("layers", [8, 8, 12, 6])
            self.freeze_epochs = kwargs.get(
                "freeze_epochs", data.get("freeze_epochs", 0)
            )
            # Head topology — tunable so the progressive-vs-parallel ablation can run
            # through the same search path as the LR sweep.
            self.head_mode = kwargs.get(
                "head_mode", data.get("head_mode", "progressive")
            )
            self.model_version = data.get(
                "model_version", datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        col_set = {
            "species": InatClassificationDataset.zero_indexed_species,
            "genus": InatClassificationDataset.zero_indexed_genus,
            "family": InatClassificationDataset.zero_indexed_family,
        }

        with session_maker() as session:
            rows = session.execute(
                select(
                    func.max(InatClassificationDataset.zero_indexed_species),
                    func.max(InatClassificationDataset.zero_indexed_genus),
                    func.max(InatClassificationDataset.zero_indexed_family),
                )
            )
            self.num_labels = [[r[0] + 1, r[1] + 1, r[2] + 1] for r in rows][0]

            weight_dict = defaultdict(list)
            for lbl, col in col_set.items():
                rows = session.execute(
                    select(
                        col,
                        func.round(
                            (func.sum(func.count()).over() / func.count(col).over())
                            / func.count(),
                            4,
                        ).label("weight"),
                    )
                    .group_by(col)
                    .order_by(col.asc())
                )
                weight_dict[lbl] = [float(r.weight) for r in rows]

        self.train_dataloader = class_dataloader(
            split="train", batch=self.batch_size, tuning=tuning
        )
        self.val_dataloader = class_dataloader(
            split="val", batch=self.batch_size, tuning=tuning
        )
        _runs_root = Path(__file__).parents[1] / "runs" / "classification"
        self.output_dir = _runs_root / f"{self.model_version}_classification"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_csv = _runs_root / "experiments.csv"
        self.metrics = ClassificationMetrics(
            output_dir=self.output_dir,
            session_factory=session_maker,
        )
        self._init_loss_csv()
        self.model = CustomResnet(
            block=BasicBlock,
            layers=self.layers,
            num_class=self.num_labels,
            in_dim=self.in_dim,
            pretrained=self.pretrained,
            head_mode=self.head_mode,
        ).to(device=self.device)
        if self.pretrained:
            self.model.load_pretrained()

        self.loss_phase = 1
        self.consecutive_epochs = 0
        # Actual epoch a phase began — anchors the curriculum's time ramp so an early
        # gate-driven transition shortens total training instead of stalling the next
        # phase on the nominal fixed schedule.
        self.phase_start_epoch = 0
        self.criterion_species = nn.CrossEntropyLoss(
            weight=torch.tensor(weight_dict["species"]).float().to(self.device),
            label_smoothing=0.1,
        )
        self.criterion_genus = nn.CrossEntropyLoss(
            weight=torch.tensor(weight_dict["genus"]).float().to(self.device),
            label_smoothing=0.1,
        )
        self.criterion_family = nn.CrossEntropyLoss(
            weight=torch.tensor(weight_dict["family"]).float().to(self.device),
            label_smoothing=0.1,
        )
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            # Per-group max_lr when pretrained (discriminative), scalar otherwise.
            max_lr=(
                [self.head_lr, self.backbone_lr] if self.pretrained else self.max_lr
            ),
            steps_per_epoch=len(self.train_dataloader),
            epochs=self.epochs,
            pct_start=self.warmup_pct,  # was silently the 0.3 default
        )
        # Phase A of progressive unfreezing: start with the backbone frozen so the
        # randomly-init heads adapt to pretrained features before the body moves.
        if self.pretrained and self.freeze_epochs > 0:
            self._set_backbone_requires_grad(False)

    def _head_stem_params(self):
        """Params that should move fast: the 3 classifier heads + the two
        progressive-head projections (all randomly-init) + (variant A) the inflated
        5ch stem, which has new channels the pretrained body never saw.

        Anything omitted here falls into _backbone_params() by exclusion, so it would
        train at the ~10x slower backbone_lr and be frozen during Phase A warmup —
        wrong for freshly-init heads/projections."""

        modules = [
            self.model.fc_species,
            self.model.fc_genus,
            self.model.fc_family,
        ]
        # Progressive mode adds the two parent->child projection layers (also
        # randomly-init, so they belong with the fast head group); parallel mode
        # has none.
        if self.model.head_mode == "progressive":
            modules.extend([self.model.proj_family, self.model.proj_genus])
        if self.model.in_dim == 5:
            modules.extend([self.model.conv1, self.model.bn1])

        return [p for m in modules for p in m.parameters()]

    def _backbone_params(self):
        """The pretrained body — everything that is NOT head/stem."""

        head_stem_ids = {id(p) for p in self._head_stem_params()}
        return [p for p in self.model.parameters() if id(p) not in head_stem_ids]

    def _build_optimizer(self):
        """Single group for from-scratch (variant C); two discriminative groups
        (head/stem fast, backbone ~10x slower) for the pretrained variants.

        Group ORDER must match the OneCycleLR max_lr list: [head_lr, backbone_lr].
        """
        if not self.pretrained:
            return optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        return optim.AdamW(
            [
                {"params": self._head_stem_params(), "lr": self.head_lr},
                {"params": self._backbone_params(), "lr": self.backbone_lr},
            ],
            weight_decay=self.weight_decay,
        )

    def _set_backbone_requires_grad(self, flag: bool):
        """Progressive unfreeze: Phase A (epochs < freeze_epochs) trains heads/stem
        only; then the backbone is unfrozen to fine-tune at backbone_lr."""
        for p in self._backbone_params():
            p.requires_grad = flag

    def _init_loss_csv(self):
        """Create this run's per-epoch losses.csv with its header if absent."""
        loss_csv = self.output_dir / "losses.csv"
        if not loss_csv.exists():
            with open(loss_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "epoch",
                        "lr",
                        "train_species",
                        "train_genus",
                        "train_family",
                        "train_total",
                        "val_loss",
                    ]
                )

    def _log_losses(self, epoch: int, train_losses: tuple, val_loss: float):
        """Append one epoch's LR, per-head/total train losses and val loss to
        losses.csv for later plotting."""
        lr = self.lr_scheduler.get_last_lr()[0]
        with open(self.output_dir / "losses.csv", "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{lr:.6f}",
                    *[f"{v:.6f}" for v in train_losses],
                    f"{val_loss:.6f}",
                ]
            )

    def _register_experiment(self, final_metrics: dict, best_val_loss: float):
        """Append this run's config and headline metrics to the global
        experiments.csv registry so runs are comparable at a glance. If the
        header schema has changed, the old file is archived first.
        """
        # `pretrained` disambiguates which LR knobs actually drove the run:
        #   pretrained=True  → optimizer uses head_lr/backbone_lr; OneCycle max_lr is
        #                      [head_lr, backbone_lr]. lr/max_lr are NO-OPS here.
        #   pretrained=False → optimizer uses lr; OneCycle max_lr is max_lr.
        #                      head_lr/backbone_lr are unused.
        # All are logged regardless so the registry is self-describing for both arms.
        header = [
            "model_version",
            "timestamp",
            "epochs",
            "pretrained",
            "head_mode",
            "lr",
            "max_lr",
            "head_lr",
            "backbone_lr",
            "weight_decay",
            "freeze_epochs",
            "grad_clip",
            "batch_size",
            "loss_weights",
            "top1_species",
            "top3_species",
            "top5_species",
            "top1_genus",
            "top3_genus",
            "top1_family",
            "species_wrong_genus_right",
            "best_val_loss",
        ]
        row = [
            self.model_version,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.epochs,
            self.pretrained,
            self.head_mode,
            self.lr,
            self.max_lr,
            self.head_lr,
            self.backbone_lr,
            self.weight_decay,
            self.freeze_epochs,
            self.grad_clip,
            self.batch_size,
            self.loss_weights,
            final_metrics.get("top1_species"),
            final_metrics.get("top3_species"),
            final_metrics.get("top5_species"),
            final_metrics.get("top1_genus"),
            final_metrics.get("top3_genus"),
            final_metrics.get("top1_family"),
            final_metrics.get("species_wrong_genus_right"),
            f"{best_val_loss:.6f}",
        ]

        # Schema migration: if an existing registry has a different header, archive
        # it so we never append mismatched columns onto the old (no-op-LR) schema.
        expected = ",".join(header)
        write_header = True
        if self.experiments_csv.exists():
            lines = self.experiments_csv.read_text().splitlines()
            first_line = lines[0] if lines else ""
            if first_line == expected:
                write_header = False
            elif first_line:
                archive = self.experiments_csv.with_name(
                    f"experiments_legacy_{datetime.now():%Y%m%d_%H%M%S}.csv"
                )
                self.experiments_csv.rename(archive)
                logger.info(f"Registry schema changed; archived old CSV → {archive}")

        with open(self.experiments_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
        logger.info(f"Experiment registered → {self.experiments_csv}")

    def _save_checkpoint(self, epoch: int, val_loss: float, filename: str):
        # Full training state so a preempted run can resume; for serving/ONNX
        # export load only checkpoint["model"].
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "num_labels": self.num_labels,
                "model_version": self.model_version,
            },
            self.output_dir / filename,
        )
        logger.info(f"Checkpoint saved → {self.output_dir / filename}")

    def _freeze_backbone_bn(self):
        """Set BACKBONE BatchNorm layers to eval so their frozen ImageNet running
        stats aren't overwritten by the new input distribution during Phase A.
        """
        head_stem_ids = {id(p) for p in self._head_stem_params()}
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) and id(m.weight) not in head_stem_ids:
                m.eval()

    def _curriculum_loss(self, epoch: int, time_constraint: int = None) -> list[float]:
        """Return this epoch's [family, genus, species] loss weights for the
        taxonomic curriculum.

        The schedule walks family-only → +genus → all three, ramping the weights
        between phase endpoints. A phase advances once it has run a minimum time
        AND either the time ramp completes or the performance gate (family/genus
        recall+F1) has held for several epochs — so a model that learns the level
        quickly moves on early rather than waiting out the fixed schedule.
        """
        if self.loss_phase == 3:
            return self.loss_weights_3

        time_constraint = time_constraint or self.epochs // 4
        elapsed = epoch - self.phase_start_epoch

        START = self.loss_weights if self.loss_phase == 1 else self.loss_weights_2
        END = self.loss_weights_2 if self.loss_phase == 1 else self.loss_weights_3

        f_gate, g_gate = self.metrics.fam_gate, self.metrics.genus_gate
        gate = f_gate if self.loss_phase == 1 else g_gate
        self.consecutive_epochs = self.consecutive_epochs + 1 if gate >= 0.9 else 0

        phase_local = elapsed / time_constraint
        B_time = min(max(phase_local, 0.0), 1.0)
        B = min(max(B_time, gate / 0.9), 1.0)
        logger.info(
            f"Current gate values: \nTime gate: {B_time:.4f} | Performance gate: {gate:.4f}"
        )

        if elapsed > self.min_time_per_phase and (
            B_time >= 1 or self.consecutive_epochs >= 3
        ):
            self.loss_phase += 1
            self.consecutive_epochs = 0
            self.phase_start_epoch = epoch
            return END

        return (np.array(START) * (1 - B) + np.array(END) * B).tolist()

    def train_one_epoch(self, epoch: int):
        """Run one training epoch under the current curriculum loss weights.

        Combines the three class-weighted head losses by this epoch's curriculum
        weights, clips gradients before stepping (guarding the backbone-unfreeze
        blow-up), and steps OneCycleLR per batch. During the frozen warmup it
        re-pins backbone BatchNorm to eval. Returns the mean per-head and total
        training losses.
        """
        self.model.train()
        # Phase A: while the backbone is frozen (epoch < freeze_epochs), re-apply the
        # BN eval after model.train() reset it. Once epoch >= freeze_epochs this is
        # skipped, so model.train() leaves the backbone BN updating again — no restore
        # needed; it pairs with the requires_grad unfreeze in train().
        if self.pretrained and self.freeze_epochs > 0 and epoch < self.freeze_epochs:
            self._freeze_backbone_bn()

        running_species = 0.0
        running_genus = 0.0
        running_family = 0.0
        running_total = 0.0
        num_batches = len(self.train_dataloader)
        loss_weights = self._curriculum_loss(epoch=epoch)

        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data = data.to(self.device, non_blocking=True)
            target = {
                k: v.to(self.device, non_blocking=True) for k, v in target.items()
            }

            self.optimizer.zero_grad()

            out_species, out_genus, out_family = self.model(data)

            loss_species = self.criterion_species(out_species, target["species"])
            loss_genus = self.criterion_genus(out_genus, target["genus"])
            loss_family = self.criterion_family(out_family, target["family"])

            loss = (
                loss_species * loss_weights[0]
                + loss_genus * loss_weights[1]
                + loss_family * loss_weights[2]
            )
            loss.backward()
            # Clip BEFORE stepping so the optimizer never sees an exploding grad.
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )
            self.optimizer.step()
            self.lr_scheduler.step()

            running_species += loss_species.item()
            running_genus += loss_genus.item()
            running_family += loss_family.item()
            running_total += loss.item()

            if batch_idx % 100 == 0:
                logger.info(
                    f"{batch_idx}/{num_batches}, {(batch_idx / num_batches) * 100:.2f}% completed. Batch loss: "
                    f"species={loss_species:.4f}, genus={loss_genus:.4f}, family={loss_family:.4f}, total={loss:.4f}."
                )

        return (
            running_species / num_batches,
            running_genus / num_batches,
            running_family / num_batches,
            running_total / num_batches,
        )

    def eval_one_epoch(self, epoch: int) -> dict:
        """Evaluate on the val split and return val loss plus the full metric set.

        Val loss is always scored with the fixed phase-3 weights (not the drifting
        curriculum weights) so best.pt selection compares the same final objective
        every epoch. Feeds each batch into ClassificationMetrics for the macro/
        hierarchical reports.
        """
        self.model.eval()
        test_loss = 0.0
        num_batches = len(self.val_dataloader)

        with torch.no_grad():
            for data, target in self.val_dataloader:
                data = data.to(self.device, non_blocking=True)
                target = {
                    k: v.to(self.device, non_blocking=True) for k, v in target.items()
                }

                out_species, out_genus, out_family = self.model(data)

                loss_species = self.criterion_species(
                    out_species, target["species"]
                ).item()
                loss_genus = self.criterion_genus(out_genus, target["genus"]).item()
                loss_family = self.criterion_family(out_family, target["family"]).item()
                # Val loss is weighted by the FIXED end-state (phase-3) weights, not
                # the drifting curriculum weights, so val_loss measures the same final
                # objective every epoch and best.pt selection stays comparable across
                # phases. (self.loss_weights is the phase-1 START = [0,0,1].)
                test_loss += (
                    loss_species * self.loss_weights_3[0]
                    + loss_genus * self.loss_weights_3[1]
                    + loss_family * self.loss_weights_3[2]
                )

                self.metrics.update(out_species, out_genus, out_family, target)

        test_loss /= num_batches
        logger.info(f"Epoch {epoch} val loss: {test_loss:.4f}")

        return {"val_loss": test_loss, **self.metrics.compute(epoch)}

    def train(self):
        """Run the full training loop over all epochs.

        Each epoch trains, evaluates, checkpoints last.pt and (on val improvement)
        best.pt, and unfreezes the backbone once the warmup freeze_epochs elapses.
        Registers the run in experiments.csv at the end and returns the best val
        loss + its metrics so the in-process tuner can rank trials.
        """
        start_time = time.perf_counter()
        best_val_loss = float("inf")
        final_metrics = {}

        for epoch in range(self.epochs):
            epoch_start_time = time.perf_counter()
            logger.info(
                f"Beginning epoch {epoch}. Total elapsed: {epoch_start_time - start_time:.1f}s"
            )

            # Phase B: unfreeze the backbone once the heads have warmed up.
            if (
                self.pretrained
                and self.freeze_epochs > 0
                and epoch == self.freeze_epochs
            ):
                self._set_backbone_requires_grad(True)
                logger.info(f"Unfroze backbone at epoch {epoch} (fine-tune phase).")

            train_losses = self.train_one_epoch(epoch)
            logger.info(
                f"Epoch {epoch} train — "
                f"species={train_losses[0]:.4f}, genus={train_losses[1]:.4f}, "
                f"family={train_losses[2]:.4f}, total={train_losses[3]:.4f}"
            )

            val_metrics = self.eval_one_epoch(epoch)
            val_loss = val_metrics["val_loss"]

            self._save_checkpoint(epoch, val_loss, "last.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                final_metrics = val_metrics
                self._save_checkpoint(epoch, val_loss, "best.pt")

            self._log_losses(epoch, train_losses, val_loss)

            epoch_elapsed = time.perf_counter() - epoch_start_time
            logger.info(f"Epoch {epoch} finished in {epoch_elapsed:.1f}s")

        self._register_experiment(final_metrics, best_val_loss)
        total_elapsed = time.perf_counter() - start_time
        logger.info(f"Training complete — {self.epochs} epochs in {total_elapsed:.1f}s")
        # Returned so the in-process search can rank trials (lowest val loss wins,
        # mirroring best.pt selection).
        return best_val_loss, final_metrics


_TUNE_FNS = {
    "uniform": tune.uniform,
    "loguniform": tune.loguniform,
    "choice": tune.choice,
    "randint": tune.randint,
}


def load_param_space(path: Path, dataset: str) -> dict:
    """Load a ray.tune param space from the structured YAML config.

    Supported entry types:
      uniform / loguniform / randint  — require min + max
      choice                          — requires a list under 'values'
      fixed                           — requires a scalar under 'value';
                                        passed through as a constant (not sampled)
    Raises ValueError if the dataset key is not found in the config.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    space = raw.get(dataset)
    if space is None:
        raise ValueError(f"No param space defined for dataset '{dataset}' in {path}")

    result = {}
    for k, v in space.items():
        if v["type"] == "fixed":
            # PyYAML parses unpointed sci-notation (`1e-3`) as a STRING, not a float
            # (its float regex needs a dot: `1.0e-3`). Coerce numeric-looking fixed
            # values so a config typo can't feed a string into AdamW/OneCycleLR;
            # genuine strings (e.g. head_mode="progressive") fall through unchanged.
            val = v["value"]
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    pass
            result[k] = val
        elif v["type"] == "choice":
            result[k] = tune.choice(v["values"])
        else:
            result[k] = _TUNE_FNS[v["type"]](float(v["min"]), float(v["max"]))
    return result


class CustomResnetTuner:
    """In-process random search over the classifier's hyperparameter space.

    Samples configs from the param-space YAML, trains each serially in this
    process (deliberately avoiding Ray's per-trial workers, which OOM'd the single
    GPU), tracks the lowest-val-loss trial, and writes the winner overlaid onto
    the base config as a runnable tuned YAML.
    """

    def __init__(
        self,
        restore_path: Path = None,
        param_space: Path = Path(__file__).parents[1]
        / "config/cls_model_param_space_config.yaml",
        experiments: int = 7,
        base_config: Path = Path(__file__).parents[1]
        / "config/cls_model_config.yaml",
        out_config: Path = Path(__file__).parents[1]
        / "config/tuned_cls_model_config.yaml",
    ):
        self.restore_path = restore_path
        self.param_space = load_param_space(param_space, "classification")
        self.experiments = experiments
        # Base config supplies the non-tuned fields (model variant, loss weights,
        # batch size...); the winning trial's sampled values overlay it. Kept
        # separate from cls_model_config.yaml until the pattern's proven, then
        # promoted by hand.
        self.base_config = base_config
        self.out_config = out_config

    def _sample_config(self) -> dict:
        """Draw one config from the param space. ray.tune search-space objects
        (tune.uniform/loguniform/choice/...) expose .sample(); 'fixed' entries are
        plain scalars and pass through unchanged."""
        return {
            k: (v.sample() if hasattr(v, "sample") else v)
            for k, v in self.param_space.items()
        }

    def _write_tuned_config(self, best: tuple):
        """Overlay the winning trial's sampled hyperparameters onto the base config
        and write a complete, runnable YAML to out_config. The result is drop-in:
        point CustomResnetTrainer at it, or promote it by copying into
        cls_model_config.yaml once the search pattern is trusted."""
        val_loss, config, metrics = best
        with open(self.base_config) as f:
            merged = yaml.safe_load(f) or {}
        # dict.update keeps base ordering for existing keys and appends new ones
        # (e.g. head_mode if it isn't already in the base config).
        merged.update(config)

        header = (
            f"# ── Tuned classification config (auto-generated) ───────────────────\n"
            f"# Best of {self.experiments} trials — val_loss={val_loss:.6f}, "
            f"top1_species={metrics.get('top1_species')}, "
            f"top3_species={metrics.get('top3_species')}\n"
            f"# Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"# Tuned keys overlaid onto {self.base_config.name}; review, then copy "
            f"into cls_model_config.yaml to promote.\n\n"
        )
        with open(self.out_config, "w") as f:
            f.write(header)
            yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)
        logger.info(f"Tuned config written → {self.out_config}")

    def train_fn(self, config: dict):
        """Train one trial: build a trainer from the sampled config and run it.

        Forwards the sampled hyperparameters as kwargs (so they override the YAML),
        returning (best_val_loss, metrics) for the search to rank by.
        """
        # Only forward head_mode when the param space actually samples it (option B,
        # mixed search). If it's absent (option A, fixed sweep), DON'T pass it — a
        # kwarg would override the value set in cls_model_config.yaml, defeating the
        # whole point of fixing it there.
        extra = {}
        if "head_mode" in config:
            extra["head_mode"] = config["head_mode"]
        model = CustomResnetTrainer(
            tuning=True,
            backbone_lr=config["backbone_lr"],
            head_lr=config["head_lr"],
            weight_decay=config["weight_decay"],
            lr=config["lr"],
            warmup_pct=config["warmup_pct"],
            max_lr=config["max_lr"],
            min_time_per_phase=config["min_time_per_phase"],
            **extra,
        )
        result = model.train()
        return result

    def tune(self):
        """In-process random search. Keeps ray.tune's search-space samplers (the
        param space is still built from tune.uniform/loguniform/... in
        load_param_space) but drops tune.Tuner and its per-trial worker processes —
        those spawned a second torch/CUDA process that OOM'd this single-GPU box.

        Trials run serially in THIS process; selection mirrors the trainer's own
        checkpointing (lowest val loss wins). Lost vs. Tuner: schedulers (ASHA),
        parallel trials, and resumable search state — acceptable on one GPU.
        """
        if self.restore_path:
            logger.warning(
                "restore_path is ignored: in-process search keeps no Ray Tuner "
                "state to restore from."
            )

        best = None  # (val_loss, config, metrics)
        for i in range(self.experiments):
            config = self._sample_config()
            logger.info(f"Trial {i + 1}/{self.experiments} — config: {config}")

            val_loss, metrics = self.train_fn(config)

            if best is None or val_loss < best[0]:
                best = (val_loss, config, metrics)
                logger.info(
                    f"New best (trial {i + 1}): val_loss={val_loss:.4f}, "
                    f"top1_species={metrics.get('top1_species')}"
                )

        logger.info(f"Search complete — best val_loss={best[0]:.4f}, config={best[1]}")
        self._write_tuned_config(best)
        return best


def main(**kwargs):
    """CLI entry point: run a tuning search or a single full training run."""
    load_dotenv()
    if kwargs.get("type", None) == "tune":
        logger.info("Starting tuning of classification model.")
        tuner = CustomResnetTuner()
        tuner.tune()
    else:
        model = CustomResnetTrainer()
        model.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trains an image classifier to predict family, genus, species.",
        epilog=(
            "Examples of the use of this module:\n"
            "Hyperparameter tuning: --type tune\n"
            "Full retraining/training: --type full"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--type", choices=["tune", "full"], default="full")
    args = parser.parse_args()
    main(type=args.type)
