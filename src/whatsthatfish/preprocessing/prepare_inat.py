"""Builds the iNaturalist classification and object-detection training sets.

The headline job is an honest, generalization-testing train/val split: cluster
observations geographically by lat/lon (KMeans, GPU via cuml for the full fit),
hold out whole clusters for validation so the model is tested on regions it never
trained on, then top up sparse taxa with whole observations to keep duplicate
photos off the split boundary. Taxa are tagged with family/genus/species from the
iNat ancestry, and the result is written to inat_classification_dataset (coral
excluded) and inat_obj_detection_dataset (coral included as negatives).
"""

from typing import Any

import numpy as np
from sqlalchemy import select, func, desc, insert, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from cuml.cluster import KMeans as cuKMeans
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib
from collections import defaultdict

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ..database.config import get_session_factory
from ..database.models import (
    InatClipContext,
    InatFilteredObservations,
    InatImageQuality,
    InatTaxa,
    InatClassificationDataset,
    InatObjDetectionDataset,
)
from ..config import _get_logger
from ..retry import db_retry


logger = _get_logger(__name__)


class InatPreparation:
    """Orchestrates the geographic split and dataset population for iNat.

    End to end: pull sampled observations, tag family/genus from ancestry,
    KMeans-cluster by location, assign a generalization-honest train/val split,
    and write the classification + OD tables. The `run()` method drives the
    whole pipeline; `kmeans_search` toggles re-sweeping k vs. reusing a cached k.
    """

    def __init__(self, session_factory: sessionmaker, kmeans_search: bool = True):
        self.session = session_factory
        self.kmeans_search = kmeans_search
        self.kmeans_path = Path(__file__).parents[2] / "kmeans_search"

    def _plot_kmeans_search(self, results: dict, best_k: int):
        """Save side-by-side silhouette-vs-k and elbow (inertia)-vs-k plots.

        Visual confirmation of the chosen k for the geographic clustering sweep.
        """
        ks = sorted(results.keys())
        silhouettes = [results[k]["silhouette"] for k in ks]
        inertias = [results[k]["inertia"] for k in ks]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(ks, silhouettes, marker="o")
        ax1.axvline(x=best_k, color="r", linestyle="--", label=f"best k={best_k}")
        ax1.set_xlabel("k")
        ax1.set_ylabel("silhouette score")
        ax1.set_title("Silhouette vs k")
        ax1.legend()

        ax2.plot(ks, inertias, marker="o")
        ax2.axvline(x=best_k, color="r", linestyle="--", label=f"best k={best_k}")
        ax2.set_xlabel("k")
        ax2.set_ylabel("inertia")
        ax2.set_title("Elbow vs k")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.kmeans_path / "kmeans_search.png")
        plt.close()

    def _visualize_clusters(
        self, coords_radians: np.ndarray, labels: np.ndarray, k: int, filename: str
    ):
        """Plot the clusters on a lat/lon canvas as colour-coded convex hulls.

        A quick sanity check that geographic clusters look spatially coherent
        (each cluster's points drawn as a hull, labelled at its centroid).
        """
        cmap = plt.get_cmap("tab20" if k <= 20 else "hsv")
        fig, ax = plt.subplots(figsize=(16, 8))

        patches = []
        patch_colors = []

        for cluster_id in range(k):
            pts = coords_radians[labels == cluster_id]
            if len(pts) < 3:
                continue
            color = cmap(cluster_id / k)
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                polygon_xy = np.column_stack(
                    [hull_pts[:, 1], hull_pts[:, 0]]
                )  # lon, lat
                patches.append(Polygon(polygon_xy, closed=True))
                patch_colors.append(color)
            except Exception:
                pass

            centroid_lon = np.mean(pts[:, 1])
            centroid_lat = np.mean(pts[:, 0])
            ax.text(
                centroid_lon,
                centroid_lat,
                str(cluster_id),
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="black",
            )

        collection = PatchCollection(
            patches,
            facecolor=[(*c[:3], 0.3) for c in patch_colors],
            edgecolor=[(*c[:3], 0.9) for c in patch_colors],
            linewidth=1,
        )
        ax.add_collection(collection)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi / 2, np.pi / 2)
        ax.set_xlabel("longitude (radians)")
        ax.set_ylabel("latitude (radians)")
        ax.set_title(f"KMeans clusters (k={k})")
        plt.tight_layout()
        plt.savefig(f"{self.kmeans_path}/{filename}")
        plt.close()

    def _write_kmeans_log(self, results: dict):
        with open(self.kmeans_path / "kmeans_log.txt", "w") as f:
            for k in sorted(results.keys()):
                f.write(
                    f"k={k}, silhouette={results[k]['silhouette']:.6f}, inertia={results[k]['inertia']:.2f}\n"
                )

    def run_kmeans_search(
        self, coords_radians, k_range: range = range(5, 100, 5)
    ) -> int:
        """
        Sweep k, scoring each with silhouette (higher = better separation) and
        inertia (elbow method — look for the kink where adding k stops helping).
        Returns best_k by silhouette; elbow plot gives visual confirmation.
        """
        results: dict[int, dict] = {}

        logger.info(f"Sweeping k in {list(k_range)}.")
        for k in k_range:
            km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
            labels = km.fit_predict(coords_radians)
            sil = silhouette_score(
                coords_radians, labels, sample_size=10_000, random_state=42
            )
            results[k] = {"silhouette": sil, "inertia": km.inertia_}
            logger.info(f"k={k} silhouette={sil:.4f} inertia={km.inertia_:.2f}")

        best_k = max(results, key=lambda k: results[k]["silhouette"])
        self._write_kmeans_log(results)
        self._plot_kmeans_search(results, best_k)

        best_labels = MiniBatchKMeans(
            n_clusters=best_k, n_init=3, random_state=42
        ).fit_predict(coords_radians)
        self._visualize_clusters(
            coords_radians, best_labels, best_k, f"clusters_subsample_k{best_k}.png"
        )
        logger.info(
            f"Search complete. best_k={best_k} silhouette={results[best_k]['silhouette']:.4f}"
        )

        return best_k

    def cu_fit(self, data, k: int) -> Any:
        """GPU KMeans for the final full-dataset fit."""
        km = cuKMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit_predict(data)
        return km.labels_

    def _subsample(self, coords_radians: np.ndarray, n: int = 10000) -> np.ndarray:
        """
        Return a representative subsample of coords_radians for hyperparameter search.
        """

        logger.info(f"Creating a subsample of {n} data points.")
        min_lat, max_lat, dev_lat = (
            np.min(coords_radians[:, 0]),
            np.max(coords_radians[:, 0]),
            np.std(coords_radians[:, 0]),
        )
        min_lon, max_lon, dev_lon = (
            np.min(coords_radians[:, 1]),
            np.max(coords_radians[:, 1]),
            np.std(coords_radians[:, 1]),
        )

        step_lat = 0
        lat_bins = []
        while True:
            current_lat = min_lat + step_lat
            lat_bins.append(current_lat)
            step_lat += dev_lat
            if current_lat + step_lat > max_lat:
                lat_bins.append(max_lat)
                break

        step_lon = 0
        lon_bins = []
        while True:
            current_lon = min_lon + step_lon
            lon_bins.append(current_lon)
            step_lon += dev_lon
            if current_lon + step_lon > max_lon:
                lon_bins.append(max_lon)
                break

        lat_bins_arr = np.array(lat_bins)
        lon_bins_arr = np.array(lon_bins)

        logger.info("Binning coordinate radians.")
        coords_radians_bin_lat = np.digitize(coords_radians[:, 0], lat_bins_arr)
        coords_radians_bin_lon = np.digitize(coords_radians[:, 1], lon_bins_arr)

        cell_ids = coords_radians_bin_lat * len(lon_bins_arr) + coords_radians_bin_lon

        unique_cells = np.unique(cell_ids)
        samples_per_cell = max(1, n // len(unique_cells))

        logger.info("Selecting subsample.")
        selected = []
        for cell in unique_cells:
            idx = np.where(cell_ids == cell)[0]
            k = min(len(idx), samples_per_cell)
            selected.append(np.random.choice(idx, size=k, replace=False))

        selected_idx = np.concatenate(selected)
        if len(selected_idx) > n:
            selected_idx = np.random.choice(selected_idx, size=n, replace=False)

        logger.info(f"Subsample selected of size: {len(selected_idx)}")
        return coords_radians[selected_idx]

    def kmeans_clustering(self, rows, search: bool):
        """Cluster rows geographically and tag each with its cluster id.

        Coordinates are converted to radians, k is either re-swept on a spatial
        subsample (when `search`) or read from the cached best_k.txt, then the
        full dataset is fit on GPU (cuKMeans). Each row gets a "cluster" key.
        """
        lat_lon_pairs = [[row["latitude"], row["longitude"]] for row in rows]
        lat_lon = np.array(lat_lon_pairs, dtype=np.float32)
        coords_radians = np.radians(lat_lon)

        if search:
            logger.info(
                f"Running k search on subsample of {len(coords_radians)} points."
            )
            search_coords = self._subsample(coords_radians)
            best_k = self.run_kmeans_search(search_coords)
            with open(self.kmeans_path / "best_k.txt", "w") as f:
                f.write(f"{best_k}")
        else:
            with open(self.kmeans_path / "best_k.txt", "r") as f:
                best_k = int(f.read())

        logger.info(f"Fitting cuKMeans k={best_k} on full dataset.")
        labels = self.cu_fit(coords_radians, best_k)
        self._visualize_clusters(
            coords_radians, np.array(labels), best_k, f"clusters_full_k{best_k}.png"
        )

        for i, row in enumerate(rows):
            row["cluster"] = int(labels[i])
        return rows

    def assign_split(
        self, rows: list[dict], val_floor: int = 20, min_train: int = 50
    ) -> list[dict]:
        """
        Geographic 80/20 cluster split, then an OBSERVATION-DISJOINT top-up that gives
        sparsely-covered taxa some validation support without leaking near-duplicates.

        Two-tier semantics written onto every row:
          train=True                         → training
          train=False, val_topup=False       → pure held-out geographic val (honest)
          train=False, val_topup=True        → moved from train to meet val_floor (IID,
                                               same regions as train — NOT generalization)

        The top-up moves WHOLE observations (all photos of an observation_uuid) so no
        observation ever straddles train/val — this is what kills the duplicate-photo
        leakage the per-photo top-up was introducing. Geographic assignment already
        keeps an observation intact (one obs = one lat/lon = one cluster), so after
        this every observation is wholly on one side.
        """
        from collections import defaultdict

        unique_clusters = list({row["cluster"] for row in rows})
        rng = np.random.default_rng(42)
        val_clusters = set(
            rng.choice(
                unique_clusters, size=round(len(unique_clusters) * 0.2), replace=False
            ).tolist()
        )
        logger.info(f"Val clusters ({len(val_clusters)}): {sorted(val_clusters)}")

        for row in rows:
            row["train"] = row["cluster"] not in val_clusters
            row["val_topup"] = False  # geographic val (and train) until topped up

        def _train_counts(rows):
            train_ct, val_ct = defaultdict(int), defaultdict(int)
            for row in rows:
                (train_ct if row["train"] else val_ct)[row["taxon_id"]] += 1
            return train_ct, val_ct

        train_ct, val_ct = _train_counts(rows)

        # Drop taxa that can't field a real training set (also drops val-only taxa,
        # whose train_ct is 0).
        drop_taxon_ids = {
            taxon_id
            for taxon_id in set(train_ct) | set(val_ct)
            if train_ct.get(taxon_id, 0) < min_train
        }
        rows = [row for row in rows if row["taxon_id"] not in drop_taxon_ids]
        logger.info(f"Dropped {len(drop_taxon_ids)} taxa with <{min_train} train rows.")

        # Group remaining TRAIN rows by taxon → observation, so we can move whole obs.
        train_obs: dict[int, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        val_ct = defaultdict(int)
        for row in rows:
            if row["train"]:
                train_obs[row["taxon_id"]][row["observation_uuid"]].append(row)
            else:
                val_ct[row["taxon_id"]] += 1

        moved_obs = moved_photos = topped_taxa = 0
        for taxon_id, obs_map in train_obs.items():
            need = val_floor - val_ct.get(taxon_id, 0)
            if need <= 0:
                continue
            train_remaining = sum(len(g) for g in obs_map.values())
            obs_uuids = list(obs_map.keys())
            # Deterministic shuffle of whole observations for this taxon.
            order = rng.permutation(len(obs_uuids))
            moved_here = False
            for j in order:
                if need <= 0:
                    break
                group = obs_map[obs_uuids[j]]
                # Never let the top-up starve training below the floor.
                if train_remaining - len(group) < min_train:
                    continue
                for row in group:
                    row["train"] = False
                    row["val_topup"] = True
                train_remaining -= len(group)
                need -= len(group)
                moved_obs += 1
                moved_photos += len(group)
                moved_here = True
            topped_taxa += int(moved_here)

        val_count = sum(1 for r in rows if not r["train"])
        topup_count = sum(1 for r in rows if r["val_topup"])
        logger.info(
            f"Top-up moved {moved_photos} photos / {moved_obs} whole observations "
            f"across {topped_taxa} taxa (no observation straddles the split)."
        )
        logger.info(
            f"Split complete — train: {len(rows) - val_count}, val: {val_count} "
            f"({val_count - topup_count} geographic, {topup_count} topped-up)."
        )
        return rows

    def split_taxa(
        self, row: dict[str, str], family_set: set[int], genus_set: set[int]
    ) -> dict[str, str]:
        """Tag a row with its family/genus/species taxon ids from ancestry.

        Walks the slash-delimited ancestry, matching each ancestor against the
        family/genus membership sets; species is just the row's own taxon_id.
        Missing ranks stay None so every row keeps a uniform set of keys.
        """
        # Default to None so every row has uniform keys (the bulk insert requires
        # it) and a species lacking a family/genus rank in inat_taxa doesn't KeyError.
        row["family"] = None
        row["genus"] = None

        for anc in row["ancestry"].split("/"):
            if not anc:
                continue
            anc = int(anc)
            if anc in family_set:
                row["family"] = anc
            if anc in genus_set:
                row["genus"] = anc

        row["species"] = row["taxon_id"]
        return row

    @db_retry
    def retrieve_sampled(self):
        """
        Retrieve the relevant iNaturalist observations with:
        longitude and latitude for geographic clustering -> required for the train test split
        ancestry + taxon_id -> required to split into Family, Genus, Species
        photo_id + extension -> together form the URL for GCP GET Request
        uiqm -> required for weighted sampling based on image quality
        is_underwater -> filter to ensure we're leveraging the underwater images as defined by CLIP

        results: rows dict containing data
        """

        logger.info("Retrieving lat, lon pairs for iNaturalist Observations.")

        stmt = (
            select(
                InatTaxa.ancestry,
                InatClipContext.photo_uuid,
                InatFilteredObservations.observation_uuid,
                InatFilteredObservations.photo_id,
                InatFilteredObservations.extension,
                InatFilteredObservations.taxon_id,
                InatFilteredObservations.latitude,
                InatFilteredObservations.longitude,
                InatImageQuality.uiqm,
                InatClipContext.is_underwater,
                func.dense_rank()
                .over(
                    partition_by=InatFilteredObservations.taxon_id,
                    order_by=desc(InatImageQuality.uiqm),
                )
                .label("uiqm_rank"),
                func.count()
                .over(partition_by=InatFilteredObservations.taxon_id)
                .label("taxon_count"),
            )
            .join(
                InatFilteredObservations,
                InatClipContext.photo_uuid == InatFilteredObservations.photo_uuid,
            )
            .join(
                InatImageQuality,
                InatClipContext.photo_uuid == InatImageQuality.photo_uuid,
            )
            .join(InatTaxa, InatFilteredObservations.taxon_id == InatTaxa.taxon_id)
            .where(InatClipContext.is_underwater == 1)
            .where(InatImageQuality.uiqm.isnot(None))
            .where(InatTaxa.rank == "species")
            .where(
                InatFilteredObservations.latitude.isnot(None)
                & InatFilteredObservations.longitude.isnot(None)
            )
        )
        cte = stmt.cte("ranked")
        outer = select(
            cte.c.photo_uuid,
            cte.c.observation_uuid,
            func.concat(cte.c.photo_id, ".", cte.c.extension).label("filename"),
            cte.c.longitude,
            cte.c.latitude,
            cte.c.taxon_id,
            cte.c.uiqm_rank,
            cte.c.uiqm,
            cte.c.is_underwater,
            cte.c.ancestry,
        ).where(cte.c.taxon_count > 300, cte.c.uiqm_rank <= 300)

        session_factory = get_session_factory()
        with session_factory() as session:
            rows = session.execute(outer).all()

        logger.info("Pairs retrieved.")

        return_list = [
            {
                "photo_uuid": row.photo_uuid,
                "observation_uuid": row.observation_uuid,
                "filename": row.filename,
                "longitude": row.longitude,
                "latitude": row.latitude,
                "taxon_id": row.taxon_id,
                "uiqm_rank": row.uiqm_rank,
                "uiqm": row.uiqm,
                "is_underwater": row.is_underwater,
                "ancestry": row.ancestry,
            }
            for row in rows
        ]

        return return_list

    @db_retry
    def retrieve_family_genus(self):
        """Return {'family': [...], 'genus': [...]} of all taxon ids at each rank.

        Used to build the membership sets that split_taxa matches ancestry against.
        """

        def _build_stmt(label: str):
            return select(InatTaxa.taxon_id.label(label)).where(InatTaxa.rank == label)

        ancestry_dict = {}
        session_factory = get_session_factory()
        with session_factory() as session:
            for label in ["family", "genus"]:
                rows = session.execute(_build_stmt(label)).all()
                ancestry_dict[label] = [row[0] for row in rows]

        return ancestry_dict

    @db_retry
    def load(self, rows) -> bool:
        """Truncate and repopulate the classification + OD dataset tables.

        Coral (taxon 47533 in ancestry) is excluded from classification but kept
        in the OD set as a negative; classification additionally requires a family
        tag. Both tables are TRUNCATEd then bulk-inserted in one transaction.
        """
        # 47533 in ancestry → coral; excluded from classification, included in OD
        classification_rows = [
            r for r in rows if "47533" not in r["ancestry"] and r["family"]
        ]
        # observation_uuid is used only for the observation-disjoint top-up grouping;
        # it has no column on inat_classification_dataset, so drop it before insert.
        for r in classification_rows:
            r.pop("observation_uuid", None)
        od_rows = [
            {
                "photo_uuid": r["photo_uuid"],
                "filename": r["filename"],
                "uiqm": r["uiqm"],
                "train": r["train"],
            }
            for r in rows
        ]

        logger.info(
            "load: %d total rows → %d classification, %d OD (includes coral)",
            len(rows),
            len(classification_rows),
            len(od_rows),
        )

        with self.session() as session:
            session.execute(text("TRUNCATE TABLE inat_classification_dataset"))
            session.execute(text("TRUNCATE TABLE inat_obj_detection_dataset"))
            session.commit()

            session.execute(
                insert(InatClassificationDataset).values(classification_rows)
            )
            session.execute(insert(InatObjDetectionDataset).values(od_rows))
            session.commit()
        return True

    def count_genus_family(self, rows):
        """Log how many rows carry a genus / family tag — a coverage sanity check."""
        total = len(rows)
        genus = 0
        family = 0
        for row in rows:
            if row["genus"]:
                genus += 1
            if row["family"]:
                family += 1

        logger.info(
            f"""
            Total rows: {total}\n
            Total Genus: {genus}, pct: {genus / total:.4f}\n
            Total Family: {family}, pct: {family / total:.4f}"""
        )

    def run(self):
        """Drive the full pipeline: retrieve → tag taxa → cluster → split → load."""
        load_dotenv()
        Path.mkdir(self.kmeans_path, exist_ok=True)
        rows = self.retrieve_sampled()
        ancestry_dict = self.retrieve_family_genus()
        # Build membership sets ONCE — not per row — for O(1) ancestry matching.
        family_set = set(ancestry_dict["family"])
        genus_set = set(ancestry_dict["genus"])
        ancestry_incl_rows = [
            self.split_taxa(row, family_set, genus_set) for row in rows
        ]
        self.count_genus_family(ancestry_incl_rows)
        clustered_rows = self.kmeans_clustering(ancestry_incl_rows, search=False)
        split_rows = self.assign_split(clustered_rows)
        self.load(split_rows)
