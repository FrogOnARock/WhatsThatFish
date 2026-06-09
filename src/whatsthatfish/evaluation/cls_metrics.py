import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
from sqlalchemy import select
from sqlalchemy.orm import aliased, sessionmaker

from ..database.models import InatClassificationDataset, InatTaxa
from ..database.config import get_session_factory
from ..config import _get_logger

logger = _get_logger(__name__)


class ClassificationMetrics:
    """
    Accumulates per-batch outputs during eval, then computes and saves all metrics at epoch end.

    Outputs per epoch (written to output_dir):
      - metrics_epoch_{n}.html  — sortable per-class table (accuracy/precision/recall/F1/support)
                                   with top-3 and top-5 summary at the top
      - sunburst_epoch_{n}.html — Plotly hierarchy: subfamily → genus → species,
                                   sector size = val sample count, color = F1
      - pr_curves_epoch_{n}.png — macro-averaged PR curves for each head
    """

    def __init__(self, output_dir: Path, session_factory: sessionmaker = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sf = session_factory or get_session_factory()
        self._taxonomy = self._load_taxonomy(sf)
        self.reset()

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def reset(self):
        self._logits: dict[str, list] = {"species": [], "genus": [], "subfamily": []}
        self._targets: dict[str, list] = {"species": [], "genus": [], "subfamily": []}

    def update(
        self,
        out_species: torch.Tensor,
        out_genus: torch.Tensor,
        out_subfamily: torch.Tensor,
        target: dict,
    ):
        self._logits["species"].append(out_species.cpu())
        self._logits["genus"].append(out_genus.cpu())
        self._logits["subfamily"].append(out_subfamily.cpu())
        self._targets["species"].append(target["species"].cpu())
        self._targets["genus"].append(target["genus"].cpu())
        self._targets["subfamily"].append(target["subfamily"].cpu())

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def compute(self, epoch: int) -> dict:
        logits = {k: torch.cat(v) for k, v in self._logits.items()}
        targets = {k: torch.cat(v) for k, v in self._targets.items()}

        metrics_dfs = {
            head: self._per_class_metrics(logits[head], targets[head])
            for head in ("species", "genus", "subfamily")
        }

        topk = {
            "top3_species": self._top_k_accuracy(
                logits["species"], targets["species"], 3
            ),
            "top5_species": self._top_k_accuracy(
                logits["species"], targets["species"], 5
            ),
            "top3_genus": self._top_k_accuracy(logits["genus"], targets["genus"], 3),
        }

        consistency = self._hierarchical_consistency(logits, targets)

        self._html_table(metrics_dfs, topk, consistency, epoch)
        self._sunburst(metrics_dfs, epoch)
        self._pr_curves(logits, targets, epoch)

        self.reset()
        return {**topk, **consistency}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _per_class_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> pd.DataFrame:
        preds = logits.argmax(dim=1).numpy()
        y = targets.numpy()
        classes = np.unique(y)

        precision, recall, f1, support = precision_recall_fscore_support(
            y, preds, labels=classes, average=None, zero_division=0
        )
        accuracy = np.array(
            [(preds[y == c] == c).mean() if (y == c).any() else 0.0 for c in classes]
        )

        return pd.DataFrame(
            {
                "class_idx": classes,
                "accuracy": accuracy.round(4),
                "precision": precision.round(4),
                "recall": recall.round(4),
                "f1": f1.round(4),
                "support": support,
            }
        )

    def _top_k_accuracy(
        self, logits: torch.Tensor, targets: torch.Tensor, k: int
    ) -> float:
        top_k = torch.topk(logits, k, dim=1).indices
        correct = top_k.eq(targets.unsqueeze(1).expand_as(top_k))
        return round(correct.any(dim=1).float().mean().item(), 4)

    def _hierarchical_consistency(self, logits: dict, targets: dict) -> dict:
        """
        Measures whether errors stay within the correct taxonomic neighborhood.

        Denominator is always the error set, not total samples — so 1.0 means
        every species mistake still landed in the right genus, not that all
        species were correct.

        Three rates:
          species_wrong_genus_right     — missed species but predicted correct genus
          species_wrong_subfamily_right — missed species but predicted correct subfamily
          genus_wrong_subfamily_right   — missed genus but predicted correct subfamily
        """
        pred_species = logits["species"].argmax(dim=1)
        pred_genus = logits["genus"].argmax(dim=1)
        pred_subfamily = logits["subfamily"].argmax(dim=1)

        species_wrong = pred_species != targets["species"]
        genus_wrong = pred_genus != targets["genus"]

        def _rate(
            mask: torch.Tensor, pred: torch.Tensor, target: torch.Tensor
        ) -> float:
            if not mask.any():
                return 1.0
            return (pred[mask] == target[mask]).float().mean().item()

        return {
            "species_wrong_genus_right": round(
                _rate(species_wrong, pred_genus, targets["genus"]), 4
            ),
            "species_wrong_subfamily_right": round(
                _rate(species_wrong, pred_subfamily, targets["subfamily"]), 4
            ),
            "genus_wrong_subfamily_right": round(
                _rate(genus_wrong, pred_subfamily, targets["subfamily"]), 4
            ),
        }

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def _sunburst(self, metrics_dfs: dict, epoch: int):
        species_f1 = metrics_dfs["species"].set_index("class_idx")[["f1", "support"]]
        genus_f1 = metrics_dfs["genus"].set_index("class_idx")["f1"].to_dict()
        subfamily_f1 = metrics_dfs["subfamily"].set_index("class_idx")["f1"].to_dict()

        merged = self._taxonomy.merge(
            species_f1, left_on="species_idx", right_index=True, how="left"
        )
        merged["f1"] = merged["f1"].fillna(0.0)
        merged["support"] = merged["support"].fillna(0).astype(int)

        genus_support = merged.groupby("genus_idx")["support"].sum().to_dict()
        subfamily_support = merged.groupby("subfamily_idx")["support"].sum().to_dict()

        ids, labels, parents, values, colors = [], [], [], [], []

        for _, row in (
            self._taxonomy[["subfamily_idx", "subfamily_name"]]
            .drop_duplicates()
            .iterrows()
        ):
            ids.append(f"sf_{row.subfamily_idx}")
            labels.append(row.subfamily_name or f"Subfamily {int(row.subfamily_idx)}")
            parents.append("")
            values.append(subfamily_support.get(row.subfamily_idx, 0))
            colors.append(subfamily_f1.get(row.subfamily_idx, 0.0))

        for _, row in (
            self._taxonomy[["genus_idx", "genus_name", "subfamily_idx"]]
            .drop_duplicates()
            .iterrows()
        ):
            ids.append(f"ge_{row.genus_idx}")
            labels.append(row.genus_name or f"Genus {int(row.genus_idx)}")
            parents.append(f"sf_{row.subfamily_idx}")
            values.append(genus_support.get(row.genus_idx, 0))
            colors.append(genus_f1.get(row.genus_idx, 0.0))

        for _, row in merged.iterrows():
            ids.append(f"sp_{row.species_idx}")
            labels.append(row.species_name or f"Species {int(row.species_idx)}")
            parents.append(f"ge_{row.genus_idx}")
            values.append(int(row.support))
            colors.append(row.f1)

        fig = go.Figure(
            go.Sunburst(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                marker=dict(
                    colors=colors,
                    colorscale=[[0.0, "#d62728"], [0.65, "#ff7f0e"], [1.0, "#2ca02c"]],
                    cmin=0.0,
                    cmax=1.0,
                    showscale=True,
                    colorbar=dict(title="F1"),
                ),
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>F1: %{color:.3f}<br>Val samples: %{value}<extra></extra>",
                maxdepth=3,
            )
        )
        fig.update_layout(title=f"Taxonomy F1 — Epoch {epoch}", width=1000, height=1000)
        out = self.output_dir / f"sunburst_epoch_{epoch}.html"
        fig.write_html(str(out))
        logger.info(f"Sunburst → {out}")

    def _pr_curves(self, logits: dict, targets: dict, epoch: int):
        # Note: species head iterates 1500 classes — expect ~30-60s on CPU.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        recall_grid = np.linspace(0, 1, 200)

        for ax, head in zip(axes, ("species", "genus", "subfamily")):
            y = targets[head].numpy()
            probs = torch.softmax(logits[head], dim=1).numpy()
            classes = np.unique(y)
            y_bin = label_binarize(y, classes=np.arange(probs.shape[1]))

            ap = average_precision_score(
                y_bin[:, classes], probs[:, classes], average="macro"
            )

            all_precision = []
            for c in classes:
                p, r, _ = precision_recall_curve(y_bin[:, c], probs[:, c])
                all_precision.append(np.interp(recall_grid, r[::-1], p[::-1]))

            mean_p = np.mean(all_precision, axis=0)
            ax.plot(
                recall_grid, mean_p, lw=2, color="#1f77b4", label=f"macro AP = {ap:.3f}"
            )
            ax.fill_between(recall_grid, mean_p, alpha=0.15, color="#1f77b4")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{head.capitalize()}")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        fig.suptitle(f"Precision-Recall Curves — Epoch {epoch}", fontsize=14)
        fig.tight_layout()
        out = self.output_dir / f"pr_curves_epoch_{epoch}.png"
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        logger.info(f"PR curves → {out}")

    def _html_table(self, metrics_dfs: dict, topk: dict, consistency: dict, epoch: int):
        species_df = (
            metrics_dfs["species"]
            .merge(
                self._taxonomy[
                    ["species_idx", "species_name", "genus_name", "subfamily_name"]
                ].drop_duplicates("species_idx"),
                left_on="class_idx",
                right_on="species_idx",
                how="left",
            )
            .drop(columns=["species_idx"])
            .sort_values("f1")
        )

        genus_df = (
            metrics_dfs["genus"]
            .merge(
                self._taxonomy[
                    ["genus_idx", "genus_name", "subfamily_name"]
                ].drop_duplicates("genus_idx"),
                left_on="class_idx",
                right_on="genus_idx",
                how="left",
            )
            .drop(columns=["genus_idx"])
            .sort_values("f1")
        )

        subfamily_df = (
            metrics_dfs["subfamily"]
            .merge(
                self._taxonomy[["subfamily_idx", "subfamily_name"]].drop_duplicates(
                    "subfamily_idx"
                ),
                left_on="class_idx",
                right_on="subfamily_idx",
                how="left",
            )
            .drop(columns=["subfamily_idx"])
            .sort_values("f1")
        )

        consistency_str = ""
        if consistency:
            sgr = consistency.get("species_wrong_genus_right", "—")
            ssfr = consistency.get("species_wrong_subfamily_right", "—")
            gsfr = consistency.get("genus_wrong_subfamily_right", "—")
            consistency_str = (
                f"<p><b>Hierarchical consistency — </b>"
                f"Sp wrong → genus right: <b>{sgr}</b> &nbsp;|&nbsp; "
                f"Sp wrong → subfamily right: <b>{ssfr}</b> &nbsp;|&nbsp; "
                f"Genus wrong → subfamily right: <b>{gsfr}</b></p>"
            )

        header = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script>$(document).ready(function(){{$('table.dt').DataTable({{pageLength:50}});}});</script>
<style>body{{font-family:sans-serif;padding:1rem}}h2{{margin-top:2rem}}</style>
</head><body>
<h1>Classification Metrics — Epoch {epoch}</h1>
<p>
  <b>Top-3 Species:</b> {topk["top3_species"]:.4f} &nbsp;
  <b>Top-5 Species:</b> {topk["top5_species"]:.4f} &nbsp;
  <b>Top-3 Genus:</b>   {topk["top3_genus"]:.4f}
</p>
{consistency_str}"""

        def section(title, df, tid):
            return f"<h2>{title}</h2>" + df.to_html(
                index=False, table_id=tid, classes="dt display", border=0
            )

        html = (
            header
            + section("Species", species_df, "t_species")
            + section("Genus", genus_df, "t_genus")
            + section("Subfamily", subfamily_df, "t_subfamily")
            + "</body></html>"
        )

        out = self.output_dir / f"metrics_epoch_{epoch}.html"
        out.write_text(html, encoding="utf-8")
        logger.info(f"Metrics table → {out}")

    # ------------------------------------------------------------------
    # Taxonomy helper
    # ------------------------------------------------------------------

    def _load_taxonomy(self, session_factory: sessionmaker) -> pd.DataFrame:
        """One-time query: zero-indexed labels → taxon names for all 3 levels."""
        sp = aliased(InatTaxa)
        ge = aliased(InatTaxa)
        sf = aliased(InatTaxa)

        with session_factory() as session:
            rows = session.execute(
                select(
                    InatClassificationDataset.zero_indexed_species,
                    InatClassificationDataset.zero_indexed_genus,
                    InatClassificationDataset.zero_indexed_subfamily,
                    sp.name.label("species_name"),
                    ge.name.label("genus_name"),
                    sf.name.label("subfamily_name"),
                )
                .outerjoin(sp, InatClassificationDataset.species == sp.taxon_id)
                .outerjoin(ge, InatClassificationDataset.genus == ge.taxon_id)
                .outerjoin(sf, InatClassificationDataset.subfamily == sf.taxon_id)
                .where(InatClassificationDataset.zero_indexed_species.isnot(None))
                .distinct()
            ).all()

        return pd.DataFrame(
            rows,
            columns=[
                "species_idx",
                "genus_idx",
                "subfamily_idx",
                "species_name",
                "genus_name",
                "subfamily_name",
            ],
        )
