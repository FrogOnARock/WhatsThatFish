from pathlib import Path

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

from ..database.models import InatClassificationDataset
from ..models.loaders.c_dataloader import class_dataloader
from ..models.c_custom_resnet import CustomResnet, BasicBlock
from ..database.config import get_session_factory
from ..evaluation.cls_metrics import ClassificationMetrics
from ..config import _get_logger

logger = _get_logger(__name__)


class CustomResnetTrainer:
    def __init__(
        self,
        config: Path = Path(__file__).parents[1] / "config/cls_model_config.yaml",
        session_maker: sessionmaker = None,
    ):
        # Lazy default — get_session_factory() at definition time would make the
        # module unimportable without DATABASE_URL set.
        session_maker = session_maker or get_session_factory()

        if config:
            if config.suffix not in (".yaml", ".yml"):
                raise ValueError("Config must be a yaml file.")
            with open(config, "r") as f:
                data = yaml.safe_load(f)

            self.lr = data.get("lr", 0.001)
            self.weight_decay = data.get("weight_decay", 0.01)
            self.epochs = data.get("epochs", 50)
            self.warmup_pct = data.get("warmup_pct", 0.05)
            self.batch_size = data.get("batch_size", 16)
            self.loss_weights = data.get("loss_weights", [0.6, 0.3, 0.1])
            self.max_lr = data.get("max_lr", 0.01)
            # Model variant (A/B/C harness) + fine-tune schedule
            self.pretrained = data.get("pretrained", False)
            self.in_dim = data.get("in_dim", 5)
            self.layers = data.get("layers", [8, 8, 12, 6])
            self.backbone_lr = data.get("backbone_lr", 1e-4)
            self.head_lr = data.get("head_lr", 1e-3)
            self.freeze_epochs = data.get("freeze_epochs", 0)
            self.model_version = data.get(
                "model_version", datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        col_set = {
            "species": InatClassificationDataset.zero_indexed_species,
            "genus": InatClassificationDataset.zero_indexed_genus,
            "subfamily": InatClassificationDataset.zero_indexed_subfamily,
        }

        with session_maker() as session:
            rows = session.execute(
                select(
                    func.max(InatClassificationDataset.zero_indexed_species),
                    func.max(InatClassificationDataset.zero_indexed_genus),
                    func.max(InatClassificationDataset.zero_indexed_subfamily),
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

        # TODO when tuning we'll need to add in max-samples
        self.train_dataloader = class_dataloader(split="train", batch=self.batch_size)
        self.val_dataloader = class_dataloader(split="val", batch=self.batch_size)
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
        ).to(device=self.device)
        if self.pretrained:
            self.model.load_pretrained()

        # W3.2 — de-double-correction (you implement the chosen axis):
        # Today imbalance is corrected TWICE — these inverse-frequency `weight=`
        # tensors AND the UIQM sampler (which actually weights by quality, not
        # class freq, so it distorts the distribution these weights assume).
        # Pick ONE: (a) swap weight_dict[...] for class-balanced effective-number
        # weights (1-β)/(1-β^n), or (b) drop weight= here and make the sampler
        # class-balanced. Do not run both at full strength.
        # label_smoothing=0.1 stays regardless (consistent Top-1/Top-3 gain).
        self.criterion_species = nn.CrossEntropyLoss(
            weight=torch.tensor(weight_dict["species"]).float().to(self.device),
            label_smoothing=0.1,
        )
        self.criterion_genus = nn.CrossEntropyLoss(
            weight=torch.tensor(weight_dict["genus"]).float().to(self.device),
            label_smoothing=0.1,
        )
        self.criterion_subfamily = nn.CrossEntropyLoss(
            weight=torch.tensor(weight_dict["subfamily"]).float().to(self.device),
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
        """Params that should move fast: the 3 classifier heads + (variant A) the
        inflated 5ch stem, which has new channels the pretrained body never saw."""

        modules = [self.model.fc_species, self.model.fc_genus, self.model.fc_subfamily]
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
        loss_csv = self.output_dir / "losses.csv"
        if not loss_csv.exists():
            with open(loss_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "epoch",
                        "lr",
                        "train_species",
                        "train_genus",
                        "train_subfamily",
                        "train_total",
                        "val_loss",
                    ]
                )

    def _log_losses(self, epoch: int, train_losses: tuple, val_loss: float):
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
        header = [
            "model_version",
            "timestamp",
            "epochs",
            "lr",
            "max_lr",
            "weight_decay",
            "batch_size",
            "loss_weights",
            "top3_species",
            "top5_species",
            "top3_genus",
            "species_wrong_genus_right",
            "best_val_loss",
        ]
        write_header = not self.experiments_csv.exists()
        with open(self.experiments_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(
                [
                    self.model_version,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epochs,
                    self.lr,
                    self.max_lr,
                    self.weight_decay,
                    self.batch_size,
                    self.loss_weights,
                    final_metrics.get("top3_species"),
                    final_metrics.get("top5_species"),
                    final_metrics.get("top3_genus"),
                    final_metrics.get("species_wrong_genus_right"),
                    f"{best_val_loss:.6f}",
                ]
            )
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

    def train_one_epoch(self, epoch: int):

        self.model.train()
        # Phase A: while the backbone is frozen (epoch < freeze_epochs), re-apply the
        # BN eval after model.train() reset it. Once epoch >= freeze_epochs this is
        # skipped, so model.train() leaves the backbone BN updating again — no restore
        # needed; it pairs with the requires_grad unfreeze in train().
        if self.pretrained and self.freeze_epochs > 0 and epoch < self.freeze_epochs:
            self._freeze_backbone_bn()

        running_species = 0.0
        running_genus = 0.0
        running_subfamily = 0.0
        running_total = 0.0
        num_batches = len(self.train_dataloader)

        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data = data.to(self.device, non_blocking=True)
            target = {
                k: v.to(self.device, non_blocking=True) for k, v in target.items()
            }

            self.optimizer.zero_grad()

            out_species, out_genus, out_subfamily = self.model(data)

            loss_species = self.criterion_species(out_species, target["species"])
            loss_genus = self.criterion_genus(out_genus, target["genus"])
            loss_subfamily = self.criterion_subfamily(
                out_subfamily, target["subfamily"]
            )

            loss = (
                loss_species * self.loss_weights[0]
                + loss_genus * self.loss_weights[1]
                + loss_subfamily * self.loss_weights[2]
            )
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            running_species += loss_species.item()
            running_genus += loss_genus.item()
            running_subfamily += loss_subfamily.item()
            running_total += loss.item()

            if batch_idx % 100 == 0:
                logger.info(
                    f"{batch_idx}/{num_batches}, {(batch_idx / num_batches) * 100:.2f}% completed. Batch loss: "
                    f"species={loss_species:.4f}, genus={loss_genus:.4f}, subfamily={loss_subfamily:.4f}, total={loss:.4f}."
                )

        return (
            running_species / num_batches,
            running_genus / num_batches,
            running_subfamily / num_batches,
            running_total / num_batches,
        )

    def eval_one_epoch(self, epoch: int) -> dict:

        self.model.eval()
        test_loss = 0.0
        num_batches = len(self.val_dataloader)

        with torch.no_grad():
            for data, target in self.val_dataloader:
                data = data.to(self.device, non_blocking=True)
                target = {
                    k: v.to(self.device, non_blocking=True) for k, v in target.items()
                }

                out_species, out_genus, out_subfamily = self.model(data)

                loss_species = self.criterion_species(
                    out_species, target["species"]
                ).item()
                loss_genus = self.criterion_genus(out_genus, target["genus"]).item()
                loss_subfamily = self.criterion_subfamily(
                    out_subfamily, target["subfamily"]
                ).item()
                test_loss += (
                    loss_species * self.loss_weights[0]
                    + loss_genus * self.loss_weights[1]
                    + loss_subfamily * self.loss_weights[2]
                )

                self.metrics.update(out_species, out_genus, out_subfamily, target)

        test_loss /= num_batches
        logger.info(f"Epoch {epoch} val loss: {test_loss:.4f}")

        return {"val_loss": test_loss, **self.metrics.compute(epoch)}

    def train(self):

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
                f"subfamily={train_losses[2]:.4f}, total={train_losses[3]:.4f}"
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


if __name__ == "__main__":
    load_dotenv()
    model = CustomResnetTrainer()
    model.train()
