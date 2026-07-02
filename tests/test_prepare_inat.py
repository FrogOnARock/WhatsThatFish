"""
Tests for the train/val split logic in `prepare_inat.InatPreparation`.

`assign_split` and `split_taxa` are the pure-Python core of the geographic
split — no DB, no GPU, fully deterministic (seeded rng=42). They decide which
photos become validation, which underpins EVERY headline macro metric. A
regression here silently corrupts the eval split, so it's worth pinning down.

The current design writes two flags per row:
    train=True                    → training
    train=False, val_topup=False  → pure held-out geographic val (honest metric)
    train=False, val_topup=True   → moved train→val to meet val_floor (IID)
and moves WHOLE observations so no observation straddles the split.

We construct `InatPreparation` with `session_factory=None` — these methods
never touch `self.session`.
"""

import pytest

from whatsthatfish.preprocessing.prepare_inat import InatPreparation


@pytest.fixture
def prep():
    return InatPreparation(session_factory=None, kmeans_search=False)


def _make(
    n_clusters: int = 10, taxa: dict[int, int] | None = None, photos_per_obs: int = 2
):
    """Synthetic clustered rows with observation grouping.

    taxa: {taxon_id: n_observations}. Each observation lives in exactly ONE
    cluster (geographic reality — one obs = one lat/lon), and carries
    `photos_per_obs` photos. Observations are spread round-robin across clusters.
    """
    taxa = taxa or {1: 40, 2: 40}
    rows = []
    oid = 0
    for taxon_id, n_obs in taxa.items():
        for k in range(n_obs):
            cluster = oid % n_clusters
            obs = f"obs-{taxon_id}-{k}"
            for _ in range(photos_per_obs):
                rows.append(
                    {
                        "taxon_id": taxon_id,
                        "observation_uuid": obs,
                        "cluster": cluster,
                    }
                )
            oid += 1
    return rows


# ─── split_taxa: ancestry → (species, genus, family) via rank sets ──────


class TestSplitTaxa:
    def test_resolves_family_and_genus_from_sets(self, prep):
        row = {"taxon_id": 4001, "ancestry": "48460/1/2/355675/47178/85497/8492"}
        out = prep.split_taxa(row, family_set={85497}, genus_set={8492})
        assert out["species"] == 4001  # species = the row's own taxon_id
        assert out["family"] == 85497
        assert out["genus"] == 8492

    def test_missing_ranks_default_to_none(self, prep):
        """A species whose ancestry has no family/genus in the sets → None, not KeyError."""
        out = prep.split_taxa(
            {"taxon_id": 5, "ancestry": "1/2/3"}, family_set=set(), genus_set=set()
        )
        assert out["family"] is None
        assert out["genus"] is None
        assert out["species"] == 5


# ─── assign_split: labels + flags ───────────────────────────────────────


class TestAssignSplitBasics:
    def test_every_row_gets_both_flags(self, prep):
        out = prep.assign_split(_make(taxa={1: 60, 2: 60}))
        assert all("train" in r and "val_topup" in r for r in out)
        assert all(isinstance(r["train"], bool) for r in out)
        assert all(isinstance(r["val_topup"], bool) for r in out)

    def test_deterministic_under_fixed_seed(self, prep):
        out_a = prep.assign_split(_make(taxa={1: 60, 2: 60}))
        out_b = prep.assign_split(_make(taxa={1: 60, 2: 60}))
        key = lambda rs: [
            (r["taxon_id"], r["observation_uuid"], r["train"], r["val_topup"])
            for r in rs
        ]
        assert key(out_a) == key(out_b)

    def test_drops_taxa_below_min_train(self, prep):
        # taxon 99 has 5 observations (10 photos) — far below min_train=50.
        out = prep.assign_split(_make(taxa={1: 60, 2: 60, 99: 5}))
        surviving = {r["taxon_id"] for r in out}
        assert 99 not in surviving
        assert {1, 2} <= surviving

    def test_both_splits_populated(self, prep):
        out = prep.assign_split(_make(taxa={1: 80, 2: 80, 3: 80}))
        assert any(r["train"] for r in out)
        assert any(not r["train"] for r in out)


# ─── Anti-leakage invariants (the reason this split exists) ─────────────


class TestSplitInvariants:
    def test_no_observation_straddles_the_split(self, prep):
        """Every observation_uuid is wholly train or wholly val — never both.

        This is the core duplicate-leakage guard: photos of one observation are
        near-duplicates, so splitting them across train/val would inflate val.
        """
        out = prep.assign_split(_make(taxa={1: 80, 2: 80}))
        by_obs: dict[str, set[bool]] = {}
        for r in out:
            by_obs.setdefault(r["observation_uuid"], set()).add(r["train"])
        straddlers = {o for o, sides in by_obs.items() if len(sides) > 1}
        assert straddlers == set(), f"observations on both sides: {straddlers}"

    def test_no_geographic_cluster_mixes_geo_val_and_train(self, prep):
        """A geographic val cluster is PURE: it never also contains train rows.

        Top-up only moves train→val (flagged val_topup=True), so a cluster that
        contributed an honest geographic-val row (train=False, val_topup=False)
        must have ALL its rows on the val side.
        """
        out = prep.assign_split(_make(taxa={1: 80, 2: 80, 3: 80}))
        geo_val_clusters = {
            r["cluster"] for r in out if not r["train"] and not r["val_topup"]
        }
        train_clusters = {r["cluster"] for r in out if r["train"]}
        leaked = geo_val_clusters & train_clusters
        assert leaked == set(), f"clusters mixing geo-val and train: {leaked}"

    def test_topup_rows_are_flagged_and_are_val(self, prep):
        """Any topped-up row is on the val side (train=False)."""
        out = prep.assign_split(_make(taxa={1: 80, 2: 80}))
        assert all(r["train"] is False for r in out if r["val_topup"])

    # Val-coverage guarantee — a taxon below val_floor geographically, but with a
    # deep train pool, is topped up to val_floor (without starving train below
    # min_train). 20 observations × 2 photos over 10 clusters → 4 photos in each
    # cluster; with 2 val clusters the geographic split yields only 8 val photos
    # (< val_floor=10), and ~32 train photos (≫ min_train=10) of headroom for the
    # top-up to move whole observations across.
    def test_val_floor_coverage(self, prep):
        out = prep.assign_split(
            _make(taxa={1: 20}, n_clusters=10, photos_per_obs=2),
            val_floor=10,
            min_train=10,
        )
        geo_val = sum(1 for r in out if not r["train"] and not r["val_topup"])
        total_val = sum(1 for r in out if not r["train"])
        train = sum(1 for r in out if r["train"])

        assert geo_val < 10, "geographic split alone should fall short of val_floor"
        assert total_val >= 10, "top-up should lift coverage to val_floor"
        assert train >= 10, "top-up must not starve training below min_train"
