"""Hierarchical evaluation metrics for the family/genus/species classifier.

Accumulates each eval batch's logits and targets, then at epoch end computes the
headline macro Top-1/3/5 numbers (split out by geographic vs IID-top-up val),
per-class precision/recall/F1, hierarchical-consistency rates and the curriculum
gates the trainer reads — and writes an HTML table, a taxonomy sunburst and
per-head PR curves for inspection.
"""

import gc

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
      - sunburst_epoch_{n}.html — Plotly hierarchy: family → genus → species,
                                   sector size = val sample count, color = F1
      - pr_curves_epoch_{n}.png — macro-averaged PR curves for each head
    """

    def __init__(
        self,
        output_dir: Path,
        session_factory: sessionmaker = None,
        plot_every: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sf = session_factory or get_session_factory()
        self._taxonomy = self._load_taxonomy(sf)
        self.fam_gate = 0.0
        self.genus_gate = 0.0
        # The sunburst + PR-curve writers are O(N_val × n_classes) in host RAM;
        # run them every `plot_every` epochs (and on the forced final epoch) so a
        # large class set doesn't OOM the box every epoch. The lightweight
        # per-class table + curriculum gates still run every epoch. plot_every=1
        # restores the old every-epoch behaviour.
        self.plot_every = max(1, int(plot_every))
        self.reset()

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def reset(self):
        """Clear the accumulated logits/targets/top-up buffers for a new epoch."""
        self._logits: dict[str, list] = {"species": [], "genus": [], "family": []}
        self._targets: dict[str, list] = {"species": [], "genus": [], "family": []}
        # Per-sample topped-up flag (1 = IID top-up, 0 = geographic val).
        self._topup: list = []

    def update(
        self,
        out_species: torch.Tensor,
        out_genus: torch.Tensor,
        out_family: torch.Tensor,
        target: dict,
    ):
        """Stash one batch's head logits and targets (on CPU) for epoch-end compute.

        Also records each sample's top-up flag (IID rare-class coverage vs true
        geographic val); callers that omit it have the whole batch treated as
        geographic.
        """
        self._logits["species"].append(out_species.cpu())
        self._logits["genus"].append(out_genus.cpu())
        self._logits["family"].append(out_family.cpu())
        self._targets["species"].append(target["species"].cpu())
        self._targets["genus"].append(target["genus"].cpu())
        self._targets["family"].append(target["family"].cpu())
        # Absent (e.g. legacy callers) → treat the whole batch as geographic val.
        topup = target.get("topup")
        self._topup.append(
            topup.cpu()
            if topup is not None
            else torch.zeros_like(target["species"].cpu())
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def compute(self, epoch: int, force_plots: bool = False) -> dict:
        """Compute every metric for the accumulated epoch and emit the reports.

        Builds the micro top-k numbers, the macro blocks (all / geographic /
        topped-up), per-class tables, hierarchical consistency and the curriculum
        gates; writes the HTML table every epoch, and the heavy sunburst + PR
        curves every `plot_every` epochs (or when `force_plots`, e.g. the final
        epoch); then resets the buffers and returns the flat metric dict.
        """
        logits = {k: torch.cat(v) for k, v in self._logits.items()}
        targets = {k: torch.cat(v) for k, v in self._targets.items()}

        metrics_dfs = {
            head: self._per_class_metrics(logits[head], targets[head])
            for head in ("species", "genus", "family")
        }

        topk = {
            "top1_species": self._top_k_accuracy(
                logits["species"], targets["species"], 1
            ),
            "top3_species": self._top_k_accuracy(
                logits["species"], targets["species"], 3
            ),
            "top5_species": self._top_k_accuracy(
                logits["species"], targets["species"], 5
            ),
            "top1_genus": self._top_k_accuracy(logits["genus"], targets["genus"], 1),
            "top3_genus": self._top_k_accuracy(logits["genus"], targets["genus"], 3),
            "top1_family": self._top_k_accuracy(logits["family"], targets["family"], 1),
        }

        # Macro (per-class average) of the stated targets, split three ways by the
        # topped-up flag so the headline isn't contaminated by IID top-up rows:
        #   macro_*  — all val (kept for backward-compat / experiments.csv)
        #   geo_*    — pure held-out geographic val  ← the honest generalization number
        #   topup_*  — IID top-up rows (rare-class coverage; NOT generalization)
        topup = (
            torch.cat(self._topup)
            if self._topup
            else torch.zeros(targets["species"].shape[0], dtype=torch.long)
        )
        geo_mask = topup == 0
        topup_mask = topup == 1

        macro = self._macro_block(logits, targets, None, "macro")
        geo = self._macro_block(logits, targets, geo_mask, "geo")
        tu = self._macro_block(logits, targets, topup_mask, "topup")
        counts = {
            "geo_n": int(geo_mask.sum()),
            "topup_n": int(topup_mask.sum()),
            "geo_classes": int(targets["species"][geo_mask].unique().numel())
            if bool(geo_mask.any())
            else 0,
            "topup_classes": int(targets["species"][topup_mask].unique().numel())
            if bool(topup_mask.any())
            else 0,
        }

        consistency = self._hierarchical_consistency(logits, targets)

        self.fam_gate, self.genus_gate = self._lc_gate(metrics_dfs)
        self._html_table(metrics_dfs, topk, macro, geo, tu, counts, consistency, epoch)
        # Heavy O(N_val × n_classes) writers — throttled to avoid OOM on large
        # class sets. The table + gates above already ran (cheap, every epoch).
        if force_plots or epoch % self.plot_every == 0:
            self._sunburst(metrics_dfs, epoch)
            self._pr_curves(logits, targets, epoch)

        self.reset()
        return {**topk, **macro, **geo, **tu, **consistency}

    # ------------------------------------------------------------------
    # Learning Curriculum Gate
    # ------------------------------------------------------------------

    def _lc_gate(self, metrics_df: dict) -> tuple[float, float]:
        """Curriculum performance gates: fraction of family (and genus) classes
        that have cleared recall>0.6 and F1>0.6.

        The trainer reads these to decide a level is 'learned' and advance the
        loss curriculum. Returned as (family_gate, genus_gate); the trainer stores
        the family gate in its `fam_gate` attribute.
        """
        fam_df = metrics_df["family"]
        genus_df = metrics_df["genus"]

        family_gate = len(
            fam_df[(fam_df["recall"] > 0.6) & (fam_df["f1"] > 0.6)]
        ) / len(fam_df)

        genus_gate = len(
            genus_df[(genus_df["recall"] > 0.6) & (genus_df["f1"] > 0.6)]
        ) / len(genus_df)

        return float(family_gate), float(genus_gate)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _per_class_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> pd.DataFrame:
        """Per-class accuracy/precision/recall/F1/support for one head, over only
        the classes that actually appear in this split's targets."""
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
        """Micro (sample-weighted) top-k accuracy: fraction of samples whose true
        class is among the head's top-k predictions."""
        top_k = torch.topk(logits, k, dim=1).indices
        # Check if any of the tensors are equal to each other: [0, 0, 1, 0, 0] & [0, 0, 1, 0, 0] for example.
        correct = top_k.eq(targets.unsqueeze(1).expand_as(top_k))
        return round(correct.any(dim=1).float().mean().item(), 4)

    def _macro_top_k_accuracy(
        self, logits: torch.Tensor, targets: torch.Tensor, k: int
    ) -> float:
        """Per-class top-k accuracy averaged over classes (macro). Unlike
        _top_k_accuracy (micro, sample-weighted), every class counts equally so the
        rare-species tail isn't masked by abundant ones — this matches how the
        evaluation targets are defined. Classes absent from this split contribute
        nothing (torch.unique only iterates classes actually present in targets)."""
        top_k = torch.topk(logits, k, dim=1).indices
        hit = top_k.eq(targets.unsqueeze(1).expand_as(top_k)).any(dim=1)
        per_class = [hit[targets == c].float().mean() for c in torch.unique(targets)]
        return round(torch.stack(per_class).mean().item(), 4)

    def _macro_block(self, logits: dict, targets: dict, mask, prefix: str) -> dict:
        """The six gated macro metrics (Top-1 + Top-3 for species/genus/family),
        optionally restricted to a per-sample boolean `mask`. Returns None for a
        metric when the masked subset is empty (e.g. no topped-up rows).

        Top-3 genus/family were added alongside species so the promotion gate can
        cover all six (`promotion.CLASSIFIER_KEYS`); the four defined eval targets
        remain species Top-1/Top-3, genus Top-1, family Top-1."""

        def m(head: str, k: int):
            lg, tg = logits[head], targets[head]
            if mask is not None:
                if not bool(mask.any()):
                    return None
                lg, tg = lg[mask], tg[mask]
            return self._macro_top_k_accuracy(lg, tg, k)

        return {
            f"{prefix}_top1_species": m("species", 1),
            f"{prefix}_top3_species": m("species", 3),
            f"{prefix}_top1_genus": m("genus", 1),
            f"{prefix}_top3_genus": m("genus", 3),
            f"{prefix}_top1_family": m("family", 1),
            f"{prefix}_top3_family": m("family", 3),
        }

    def _hierarchical_consistency(self, logits: dict, targets: dict) -> dict:
        """
        Measures whether errors stay within the correct taxonomic neighborhood.

        Denominator is always the error set, not total samples — so 1.0 means
        every species mistake still landed in the right genus, not that all
        species were correct.

        Three rates:
          species_wrong_genus_right     — missed species but predicted correct genus
          species_wrong_family_right — missed species but predicted correct family
          genus_wrong_family_right   — missed genus but predicted correct family
        """
        pred_species = logits["species"].argmax(dim=1)
        pred_genus = logits["genus"].argmax(dim=1)
        pred_family = logits["family"].argmax(dim=1)

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
            "species_wrong_family_right": round(
                _rate(species_wrong, pred_family, targets["family"]), 4
            ),
            "genus_wrong_family_right": round(
                _rate(genus_wrong, pred_family, targets["family"]), 4
            ),
        }

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def _sunburst(self, metrics_dfs: dict, epoch: int):
        """Write an interactive family→genus→species sunburst HTML where sector
        size is the val sample count and color is F1, so weak taxonomic branches
        stand out at a glance."""
        species_f1 = metrics_dfs["species"].set_index("class_idx")[["f1", "support"]]
        genus_f1 = metrics_dfs["genus"].set_index("class_idx")["f1"].to_dict()
        family_f1 = metrics_dfs["family"].set_index("class_idx")["f1"].to_dict()

        merged = self._taxonomy.merge(
            species_f1, left_on="species_idx", right_index=True, how="left"
        )
        merged["f1"] = merged["f1"].fillna(0.0)
        merged["support"] = merged["support"].fillna(0).astype(int)

        genus_support = merged.groupby("genus_idx")["support"].sum().to_dict()
        family_support = merged.groupby("family_idx")["support"].sum().to_dict()

        ids, labels, parents, values, colors = [], [], [], [], []

        for _, row in (
            self._taxonomy[["family_idx", "family_name"]].drop_duplicates().iterrows()
        ):
            ids.append(f"sf_{row.family_idx}")
            labels.append(row.family_name or f"Family {int(row.family_idx)}")
            parents.append("")
            values.append(family_support.get(row.family_idx, 0))
            colors.append(family_f1.get(row.family_idx, 0.0))

        for _, row in (
            self._taxonomy[["genus_idx", "genus_name", "family_idx"]]
            .drop_duplicates()
            .iterrows()
        ):
            ids.append(f"ge_{row.genus_idx}")
            labels.append(row.genus_name or f"Genus {int(row.genus_idx)}")
            parents.append(f"sf_{row.family_idx}")
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
        """Save a 3-panel PNG of macro-averaged precision-recall curves (one per
        head), each interpolated onto a common recall grid and labelled with its
        macro average precision."""
        # Note: species head iterates 1500 classes — expect ~30-60s on CPU.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        recall_grid = np.linspace(0, 1, 200)

        for ax, head in zip(axes, ("species", "genus", "family")):
            y = targets[head].numpy()
            probs = torch.softmax(logits[head], dim=1).numpy()  # (N, C)
            classes = np.unique(y)

            # Per-class binary mask (N,) instead of a dense (N, C) one-hot — the
            # label_binarize + matrix average_precision_score were the host-RAM
            # blow-up on large class sets. Macro AP = mean of per-class APs.
            all_precision, ap_per_class = [], []
            for c in classes:
                y_c = (y == c).astype(np.int8)
                prob_c = probs[:, c]
                p, r, _ = precision_recall_curve(y_c, prob_c)
                all_precision.append(np.interp(recall_grid, r[::-1], p[::-1]))
                ap_per_class.append(average_precision_score(y_c, prob_c))

            ap = float(np.mean(ap_per_class))
            mean_p = np.mean(all_precision, axis=0)
            # Free the big (N, C) prob array before the next head.
            del probs
            gc.collect()
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

    def _html_table(
        self,
        metrics_dfs: dict,
        topk: dict,
        macro: dict,
        geo: dict,
        tu: dict,
        counts: dict,
        consistency: dict,
        epoch: int,
    ):
        """Write the per-epoch metrics HTML report.

        Joins taxon names onto each head's per-class table and renders sortable
        DataTables, headed by the micro top-k summary and the three macro blocks
        (geographic headline with targets, topped-up, all-val) plus the
        hierarchical-consistency line.
        """
        species_df = (
            metrics_dfs["species"]
            .merge(
                self._taxonomy[
                    ["species_idx", "species_name", "genus_name", "family_name"]
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
                    ["genus_idx", "genus_name", "family_name"]
                ].drop_duplicates("genus_idx"),
                left_on="class_idx",
                right_on="genus_idx",
                how="left",
            )
            .drop(columns=["genus_idx"])
            .sort_values("f1")
        )

        family_df = (
            metrics_dfs["family"]
            .merge(
                self._taxonomy[["family_idx", "family_name"]].drop_duplicates(
                    "family_idx"
                ),
                left_on="class_idx",
                right_on="family_idx",
                how="left",
            )
            .drop(columns=["family_idx"])
            .sort_values("f1")
        )

        consistency_str = ""
        if consistency:
            sgr = consistency.get("species_wrong_genus_right", "—")
            ssfr = consistency.get("species_wrong_family_right", "—")
            gsfr = consistency.get("genus_wrong_family_right", "—")
            consistency_str = (
                f"<p><b>Hierarchical consistency — </b>"
                f"Sp wrong → genus right: <b>{sgr}</b> &nbsp;|&nbsp; "
                f"Sp wrong → family right: <b>{ssfr}</b> &nbsp;|&nbsp; "
                f"Genus wrong → family right: <b>{gsfr}</b></p>"
            )

        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

        def _line(d, p, targets=True):
            tg = {
                "top1_species": " (≥0.65)",
                "top3_species": " (≥0.80)",
                "top1_genus": " (≥0.78)",
                "top1_family": " (≥0.88)",
            }
            order = ["top1_species", "top3_species", "top1_genus", "top1_family"]
            labels = {
                "top1_species": "Top-1 Species",
                "top3_species": "Top-3 Species",
                "top1_genus": "Top-1 Genus",
                "top1_family": "Top-1 Family",
            }
            return " &nbsp; ".join(
                f"<b>{labels[k]}:</b> {_fmt(d[f'{p}_{k}'])}{tg[k] if targets else ''}"
                for k in order
            )

        geo_line = _line(geo, "geo", targets=True)
        topup_line = _line(tu, "topup", targets=False)
        all_line = _line(macro, "macro", targets=False)
        counts_line = (
            f"geographic val: {counts['geo_n']} imgs / {counts['geo_classes']} classes"
            f" &nbsp;·&nbsp; topped-up val: {counts['topup_n']} imgs /"
            f" {counts['topup_classes']} classes"
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
<p><b>Micro (sample-weighted) — </b>
  <b>Top-1 Species:</b> {topk["top1_species"]:.4f} &nbsp;
  <b>Top-3 Species:</b> {topk["top3_species"]:.4f} &nbsp;
  <b>Top-5 Species:</b> {topk["top5_species"]:.4f} &nbsp;
  <b>Top-1 Genus:</b> {topk["top1_genus"]:.4f} &nbsp;
  <b>Top-3 Genus:</b>   {topk["top3_genus"]:.4f} &nbsp;
  <b>Top-1 Family:</b> {topk["top1_family"]:.4f} &nbsp;
</p>
<p><b>Macro — geographic held-out (HEADLINE, target metric) — </b>{geo_line}</p>
<p><b>Macro — topped-up IID (rare-class coverage, NOT generalization) — </b>{topup_line}</p>
<p><b>Macro — all val (reference) — </b>{all_line}</p>
<p style="color:#666">{counts_line}</p>
{consistency_str}"""

        def section(title, df, tid):
            return f"<h2>{title}</h2>" + df.to_html(
                index=False, table_id=tid, classes="dt display", border=0
            )

        html = (
            header
            + section("Species", species_df, "t_species")
            + section("Genus", genus_df, "t_genus")
            + section("Family", family_df, "t_family")
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
                    InatClassificationDataset.zero_indexed_family,
                    sp.name.label("species_name"),
                    ge.name.label("genus_name"),
                    sf.name.label("family_name"),
                )
                .outerjoin(sp, InatClassificationDataset.species == sp.taxon_id)
                .outerjoin(ge, InatClassificationDataset.genus == ge.taxon_id)
                .outerjoin(sf, InatClassificationDataset.family == sf.taxon_id)
                .where(InatClassificationDataset.zero_indexed_species.isnot(None))
                .distinct()
            ).all()

        return pd.DataFrame(
            rows,
            columns=[
                "species_idx",
                "genus_idx",
                "family_idx",
                "species_name",
                "genus_name",
                "family_name",
            ],
        )
