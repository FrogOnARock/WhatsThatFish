"""Repository-layer tests (P0) — the data-integrity + ownership-scoping core.

These hit the real test Postgres directly through the repositories (no HTTP, no
auth), so they're fast and exercise the SQL that actually guards user data:
ownership filters, site dedup, taxon translation, and the summary counts.
"""

import pytest

from whatsthatfish.serving.db.repository import (
    TaxaRepository,
    UserRepository,
    ObservationRepository,
)

USER_A = {"sub": "user-a", "email": "a@test.dev", "name": "Diver A", "picture": None}
USER_B = {"sub": "user-b", "email": "b@test.dev", "name": "Diver B", "picture": None}

TAXA = [
    {"taxon_id": 1001, "zero_index": 0, "species": "Amphiprion ocellaris",
     "genus": "Amphiprion", "family": "Pomacentridae", "common_name": "Clown anemonefish"},
    {"taxon_id": 1002, "zero_index": 1, "zero_genus": 0, "zero_family": 0,
     "species": "Amphiprion clarkii", "genus": "Amphiprion", "family": "Pomacentridae",
     "common_name": "Clark's anemonefish"},
    {"taxon_id": 2001, "zero_index": 2, "zero_genus": 1, "zero_family": 1,
     "species": "Thalassoma lunare", "genus": "Thalassoma", "family": "Labridae",
     "common_name": "Moon wrasse"},
]


def _user(session_factory, claims):
    with session_factory() as s:
        u = UserRepository(s).upsert_from_claims(claims)
        s.refresh(u)
        s.expunge(u)
        return u


# ─── UserRepository ───────────────────────────────────────────────────────────


class TestUserRepository:
    def test_first_upsert_creates_user_with_defaults(self, session_factory):
        with session_factory() as s:
            u = UserRepository(s).upsert_from_claims(USER_A)
            assert u.id is not None
            assert u.email == "a@test.dev"
            assert u.display_name == "Diver A"
            assert u.unit_system == "metric"  # server default
            assert u.preferred_name is None

    def test_second_upsert_same_subject_updates_not_duplicates(self, session_factory):
        with session_factory() as s:
            repo = UserRepository(s)
            first = repo.upsert_from_claims(USER_A)
            first_id = first.id
            changed = {**USER_A, "name": "Renamed", "email": "new@test.dev"}
            second = repo.upsert_from_claims(changed)
            assert second.id == first_id  # same row, keyed on sub
            assert second.display_name == "Renamed"
            assert second.email == "new@test.dev"

    def test_claims_sync_does_not_clobber_preferred_name(self, session_factory):
        """preferred_name is app-owned — a re-login (upsert) must not reset it."""
        with session_factory() as s:
            repo = UserRepository(s)
            u = repo.upsert_from_claims(USER_A)
            u.preferred_name = "Reef Diver"
            s.commit()
            repo.upsert_from_claims({**USER_A, "name": "Google Name"})
            s.refresh(u)
            assert u.preferred_name == "Reef Diver"
            assert u.display_name == "Google Name"

    def test_get_by_subject(self, session_factory):
        with session_factory() as s:
            repo = UserRepository(s)
            repo.upsert_from_claims(USER_A)
            assert repo.get_by_subject("user-a").email == "a@test.dev"
            assert repo.get_by_subject("nobody") is None


# ─── TaxaRepository ───────────────────────────────────────────────────────────


class TestTaxaRepository:
    def test_get_counts_distinct(self, session_factory, seed_taxa):
        seed_taxa(TAXA)
        with session_factory() as s:
            counts = TaxaRepository(s).get_counts()
        # 3 species, 2 distinct genera (Amphiprion, Thalassoma), 2 families.
        assert counts == {"species": 3, "genera": 2, "families": 2}

    def test_species_index_to_taxon(self, session_factory, seed_taxa):
        seed_taxa(TAXA)
        with session_factory() as s:
            mapping = TaxaRepository(s).species_index_to_taxon([0, 2])
        assert mapping == {0: 1001, 2: 2001}

    def test_taxa_display(self, session_factory, seed_taxa):
        seed_taxa(TAXA)
        with session_factory() as s:
            display = TaxaRepository(s).taxa_display([1001, 2001])
        assert display[1001].species == "Amphiprion ocellaris"
        assert display[1001].common_name == "Clown anemonefish"
        assert display[2001].genus == "Thalassoma"

    def test_taxa_display_empty_is_empty(self, session_factory):
        with session_factory() as s:
            assert TaxaRepository(s).taxa_display([]) == {}

    def test_search_species_matches_name_and_joins_common(self, session_factory, seed_taxa):
        seed_taxa(TAXA)
        with session_factory() as s:
            rows = TaxaRepository(s).search_species("amphiprion")
        names = {r.name for r in rows}
        assert names == {"Amphiprion ocellaris", "Amphiprion clarkii"}
        assert any(r.common_name == "Clown anemonefish" for r in rows)

    def test_search_species_excludes_non_fish_ancestry(self, session_factory, seed_taxa):
        # A coral taxon (Anthozoa ancestry) must NOT surface in the fish picker.
        seed_taxa([{"taxon_id": 9001, "zero_index": 9, "species": "Acropora coral",
                    "genus": "Acropora", "family": "Acroporidae",
                    "ancestry": "48460/1/2/47534/47533"}])
        with session_factory() as s:
            rows = TaxaRepository(s).search_species("Acropora")
        assert rows == []


# ─── ObservationRepository: site dedup ────────────────────────────────────────


class TestSiteDedup:
    def test_resolve_creates_proper_cased_site(self, session_factory):
        user = _user(session_factory, USER_A)
        with session_factory() as s:
            site = ObservationRepository(s).resolve_or_create_site("tulamben bay", user.id)
            assert site.name == "Tulamben Bay"  # proper-cased
            assert site.name_key == "tulamben bay"  # normalized key

    def test_resolve_dedupes_case_and_whitespace(self, session_factory):
        user = _user(session_factory, USER_A)
        with session_factory() as s:
            repo = ObservationRepository(s)
            a = repo.resolve_or_create_site("Blue Hole", user.id)
            b = repo.resolve_or_create_site("  blue   hole ", user.id)
            assert a.id == b.id  # same row despite case/spacing

    def test_search_sites_substring(self, session_factory):
        user = _user(session_factory, USER_A)
        with session_factory() as s:
            repo = ObservationRepository(s)
            repo.resolve_or_create_site("Blue Hole", user.id)
            repo.resolve_or_create_site("Manta Point", user.id)
            s.commit()
            hits = {site.name for site in repo.search_sites("blue")}
            assert hits == {"Blue Hole"}
            assert repo.search_sites("zzzz") == []


# ─── ObservationRepository: ownership scoping (security-critical) ──────────────


class TestOwnershipScoping:
    def _dive_with_obs(self, session_factory, seed_taxa, claims):
        """Create a user + one dive + one observation; return (user, dive_id, obs_id)."""
        seed_taxa(TAXA)
        user = _user(session_factory, claims)
        with session_factory() as s:
            repo = ObservationRepository(s)
            dive = repo.create_dive(user.id, dived_at=None)
            obs = repo.create_observation(
                user.id, dive_id=dive.id, predicted_taxon_id=1001,
                corrected_taxon_id=1001, label_status="predicted",
            )
            return user, dive.id, obs.id

    def test_get_dive_scoped_to_owner(self, session_factory, seed_taxa):
        user_a, dive_id, _ = self._dive_with_obs(session_factory, seed_taxa, USER_A)
        user_b = _user(session_factory, USER_B)
        with session_factory() as s:
            repo = ObservationRepository(s)
            assert repo.get_dive(user_a.id, dive_id) is not None
            assert repo.get_dive(user_b.id, dive_id) is None  # B can't see A's dive

    def test_get_observation_scoped_to_owner(self, session_factory, seed_taxa):
        user_a, _, obs_id = self._dive_with_obs(session_factory, seed_taxa, USER_A)
        user_b = _user(session_factory, USER_B)
        with session_factory() as s:
            repo = ObservationRepository(s)
            assert repo.get_observation(user_a.id, obs_id) is not None
            assert repo.get_observation(user_b.id, obs_id) is None

    def test_get_photo_scoped_via_observation_join(self, session_factory, seed_taxa):
        user_a, _, obs_id = self._dive_with_obs(session_factory, seed_taxa, USER_A)
        user_b = _user(session_factory, USER_B)
        with session_factory() as s:
            repo = ObservationRepository(s)
            photo = repo.create_photo(observation_id=obs_id, image_path="a/x.jpg")
            assert repo.get_photo(user_a.id, photo.id) is not None
            assert repo.get_photo(user_b.id, photo.id) is None

    def test_list_dives_only_returns_own(self, session_factory, seed_taxa):
        user_a, _, _ = self._dive_with_obs(session_factory, seed_taxa, USER_A)
        user_b = _user(session_factory, USER_B)
        with session_factory() as s:
            repo = ObservationRepository(s)
            assert len(repo.list_dives(user_a.id)) == 1
            assert repo.list_dives(user_b.id) == []


# ─── ObservationRepository: list ordering + user_stats ────────────────────────


class TestStatsAndOrdering:
    def test_list_dives_newest_first(self, session_factory):
        from datetime import datetime

        user = _user(session_factory, USER_A)
        with session_factory() as s:
            repo = ObservationRepository(s)
            repo.create_dive(user.id, dived_at=datetime(2025, 1, 1))
            repo.create_dive(user.id, dived_at=datetime(2026, 1, 1))
            dives = repo.list_dives(user.id)
            assert dives[0].dived_at == datetime(2026, 1, 1)  # desc

    def test_user_stats_counts(self, session_factory, seed_taxa):
        seed_taxa(TAXA)
        user = _user(session_factory, USER_A)
        with session_factory() as s:
            repo = ObservationRepository(s)
            dive = repo.create_dive(user.id, dived_at=None)
            # 3 observations across 2 distinct effective species (1001 twice, 2001).
            for taxon in (1001, 1001, 2001):
                repo.create_observation(
                    user.id, dive_id=dive.id, predicted_taxon_id=taxon,
                    corrected_taxon_id=taxon, label_status="predicted",
                )
            assert repo.user_stats(user.id) == {
                "dives": 1, "observations": 3, "unique_species": 2,
            }

    def test_user_stats_zero_for_new_user(self, session_factory):
        user = _user(session_factory, USER_A)
        with session_factory() as s:
            assert ObservationRepository(s).user_stats(user.id) == {
                "dives": 0, "observations": 0, "unique_species": 0,
            }
