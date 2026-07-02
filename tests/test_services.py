"""Service-layer tests (P0) — the business logic on top of the repositories:
taxon-precedence, effective-label rules, dive enrichment, field-log grouping,
settings, and the pure PredictionService helpers (no model inference).
"""

import pytest

from whatsthatfish.serving.services.service import (
    ObservationService,
    UserService,
    TaxaService,
    PredictionService,
)
from whatsthatfish.serving.schemas import (
    DiveCreate,
    DiveUpdate,
    ObservationCreate,
    ObservationUpdate,
    UserSettingsUpdate,
)
from whatsthatfish.serving.error import (
    ResourceNotFoundException,
    ValidationException,
)

USER_A = {"sub": "svc-a", "email": "a@test.dev", "name": "Diver A", "picture": None}

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


class FakeStorage:
    """Captures uploads instead of touching disk/GCS."""

    def __init__(self):
        self.uploaded = {}

    def upload(self, key, data):
        self.uploaded[key] = data
        return key

    def retrieve_image(self, key):
        return ("FILE", key)


@pytest.fixture
def obs_env(session_factory, seed_taxa):
    """A session, a persisted user (in that session), and an ObservationService
    with a fake storage — the common setup for observation-service tests."""
    seed_taxa(TAXA)
    storage = FakeStorage()
    with session_factory() as s:
        user = UserService(s).get_or_create(USER_A)
        svc = ObservationService(session=s, storage=storage)
        yield svc, user, storage


# ─── ObservationService: dives ────────────────────────────────────────────────


class TestDives:
    def test_create_dive_with_site_resolves_site(self, obs_env):
        svc, user, _ = obs_env
        out = svc.create_dive(user, DiveCreate(site_name="tulamben"))
        assert out.site_name == "Tulamben"
        assert out.site_id is not None
        assert out.observation_count == 0
        assert out.species == []

    def test_create_dive_without_site_is_null(self, obs_env):
        svc, user, _ = obs_env
        out = svc.create_dive(user, DiveCreate())
        assert out.site_name is None and out.site_id is None

    def test_update_dive_changes_site_and_notes(self, obs_env):
        svc, user, _ = obs_env
        dive = svc.create_dive(user, DiveCreate(site_name="Old Site"))
        out = svc.update_dive(user, dive.id, DiveUpdate(site_name="New Site", notes="viz 30m"))
        assert out.site_name == "New Site"
        assert out.notes == "viz 30m"

    def test_update_dive_not_found_raises(self, obs_env):
        import uuid
        svc, user, _ = obs_env
        with pytest.raises(ResourceNotFoundException):
            svc.update_dive(user, uuid.uuid4(), DiveUpdate(notes="x"))

    def test_list_dives_enrichment_counts_and_dedupes_species(self, obs_env):
        svc, user, _ = obs_env
        dive = svc.create_dive(user, DiveCreate(site_name="Reef"))
        for taxon in (1001, 1001, 2001):  # 3 obs, 2 distinct species
            svc.create_observation(user, ObservationCreate(
                dive_id=dive.id, predicted_species_index=_index_of(taxon),
            ))
        out = svc.list_dives(user)[0]
        assert out.observation_count == 3
        assert {sp.taxon_id for sp in out.species} == {1001, 2001}


# ─── ObservationService: observation taxon precedence + label ─────────────────


class TestObservationCreate:
    def test_predicted_only_sets_effective_to_predicted(self, obs_env):
        svc, user, _ = obs_env
        dive = svc.create_dive(user, DiveCreate())
        out = svc.create_observation(user, ObservationCreate(
            dive_id=dive.id, predicted_species_index=0))
        assert out.predicted_taxon_id == 1001
        assert out.corrected_taxon_id == 1001  # defaults to predicted
        assert out.label_status == "predicted"

    def test_corrected_index_overrides_effective(self, obs_env):
        svc, user, _ = obs_env
        dive = svc.create_dive(user, DiveCreate())
        out = svc.create_observation(user, ObservationCreate(
            dive_id=dive.id, predicted_species_index=0,
            corrected_species_index=2, label_status="corrected"))
        assert out.predicted_taxon_id == 1001
        assert out.corrected_taxon_id == 2001  # the corrected index wins

    def test_explicit_corrected_taxon_id_wins_over_index(self, obs_env):
        svc, user, _ = obs_env
        dive = svc.create_dive(user, DiveCreate())
        out = svc.create_observation(user, ObservationCreate(
            dive_id=dive.id, predicted_species_index=0,
            corrected_species_index=2, corrected_taxon_id=1002,
            label_status="corrected"))
        # taxon_id (report flow) beats candidate index.
        assert out.corrected_taxon_id == 1002

    def test_unknown_predicted_index_raises_validation(self, obs_env):
        svc, user, _ = obs_env
        dive = svc.create_dive(user, DiveCreate())
        with pytest.raises(ValidationException):
            svc.create_observation(user, ObservationCreate(
                dive_id=dive.id, predicted_species_index=999))

    def test_dive_not_found_raises(self, obs_env):
        import uuid
        svc, user, _ = obs_env
        with pytest.raises(ResourceNotFoundException):
            svc.create_observation(user, ObservationCreate(
                dive_id=uuid.uuid4(), predicted_species_index=0))


# ─── ObservationService: update_observation (PATCH semantics) ─────────────────


class TestObservationUpdate:
    def _obs(self, svc, user):
        dive = svc.create_dive(user, DiveCreate())
        return svc.create_observation(user, ObservationCreate(
            dive_id=dive.id, predicted_species_index=0, depth_m=5.0))

    def test_relabel_and_depth(self, obs_env):
        svc, user, _ = obs_env
        obs = self._obs(svc, user)
        out = svc.update_observation(user, obs.id, ObservationUpdate(
            corrected_taxon_id=2001, label_status="corrected", depth_m=18.0))
        assert out.corrected_taxon_id == 2001
        assert out.label_status == "corrected"
        assert out.depth_m == 18.0

    def test_exclude_unset_leaves_omitted_fields(self, obs_env):
        svc, user, _ = obs_env
        obs = self._obs(svc, user)
        # Only depth in the payload — label must be untouched.
        out = svc.update_observation(user, obs.id, ObservationUpdate(depth_m=12.0))
        assert out.depth_m == 12.0
        assert out.corrected_taxon_id == 1001  # unchanged

    def test_depth_can_be_cleared_to_null(self, obs_env):
        svc, user, _ = obs_env
        obs = self._obs(svc, user)
        out = svc.update_observation(user, obs.id, ObservationUpdate(depth_m=None))
        assert out.depth_m is None

    def test_update_not_found_raises(self, obs_env):
        import uuid
        svc, user, _ = obs_env
        with pytest.raises(ResourceNotFoundException):
            svc.update_observation(user, uuid.uuid4(), ObservationUpdate(depth_m=1.0))


# ─── ObservationService: photos + field log ───────────────────────────────────


class TestPhotosAndFieldLog:
    def test_add_photo_uploads_and_translates_index(self, obs_env):
        svc, user, storage = obs_env
        dive = svc.create_dive(user, DiveCreate())
        obs = svc.create_observation(user, ObservationCreate(
            dive_id=dive.id, predicted_species_index=0))
        photo = svc.add_photo(user, obs.id, b"jpegbytes",
                              predicted_species_index=2, confidence=0.8)
        assert len(storage.uploaded) == 1  # uploaded through storage
        assert photo.observation_id == obs.id

    def test_add_photo_obs_not_found_raises(self, obs_env):
        import uuid
        svc, user, _ = obs_env
        with pytest.raises(ResourceNotFoundException):
            svc.add_photo(user, uuid.uuid4(), b"x")

    def test_get_photo_image_not_found_raises(self, obs_env):
        import uuid
        svc, user, _ = obs_env
        with pytest.raises(ResourceNotFoundException):
            svc.get_photo_image(user, uuid.uuid4())

    def test_field_log_groups_by_effective_taxon(self, obs_env):
        svc, user, _ = obs_env
        dive = svc.create_dive(user, DiveCreate(site_name="Reef"))
        # 1001 twice + 2001 once → 2 species cards, counts 2 and 1.
        for idx in (0, 0, 2):
            svc.create_observation(user, ObservationCreate(
                dive_id=dive.id, predicted_species_index=idx))
        log = svc.get_field_log(user)
        assert log.total_species == 2
        by_taxon = {sp.taxon_id: sp.sighting_count for sp in log.species}
        assert by_taxon == {1001: 2, 2001: 1}

    def test_user_stats_via_service(self, obs_env):
        svc, user, _ = obs_env
        dive = svc.create_dive(user, DiveCreate())
        svc.create_observation(user, ObservationCreate(
            dive_id=dive.id, predicted_species_index=0))
        stats = svc.user_stats(user)
        assert (stats.dives, stats.observations, stats.unique_species) == (1, 1, 1)

    def test_search_sites_service(self, obs_env):
        svc, user, _ = obs_env
        svc.create_dive(user, DiveCreate(site_name="Coral Garden"))
        hits = svc.search_sites("coral")
        assert [h.name for h in hits] == ["Coral Garden"]


# ─── UserService: settings ────────────────────────────────────────────────────


class TestUserSettings:
    def test_profile_maps_app_owned_fields(self, session_factory):
        with session_factory() as s:
            svc = UserService(s)
            user = svc.get_or_create(USER_A)
            prof = svc.profile(user)
            assert prof.email == "a@test.dev"
            assert prof.unit_system == "metric"
            assert prof.preferred_name is None

    def test_update_sets_name_and_units(self, session_factory):
        with session_factory() as s:
            svc = UserService(s)
            user = svc.get_or_create(USER_A)
            prof = svc.update_settings(user, UserSettingsUpdate(
                preferred_name="Reef Diver", unit_system="imperial"))
            assert prof.preferred_name == "Reef Diver"
            assert prof.unit_system == "imperial"

    def test_empty_preferred_name_clears_override(self, session_factory):
        with session_factory() as s:
            svc = UserService(s)
            user = svc.get_or_create(USER_A)
            svc.update_settings(user, UserSettingsUpdate(preferred_name="X"))
            prof = svc.update_settings(user, UserSettingsUpdate(preferred_name=""))
            assert prof.preferred_name is None  # '' → cleared

    def test_partial_update_preserves_units(self, session_factory):
        with session_factory() as s:
            svc = UserService(s)
            user = svc.get_or_create(USER_A)
            svc.update_settings(user, UserSettingsUpdate(unit_system="imperial"))
            prof = svc.update_settings(user, UserSettingsUpdate(preferred_name="Y"))
            assert prof.unit_system == "imperial"  # exclude_unset left it alone


# ─── TaxaService ──────────────────────────────────────────────────────────────


class TestTaxaService:
    def test_search_species(self, session_factory, seed_taxa):
        seed_taxa(TAXA)
        with session_factory() as s:
            results = TaxaService(s).search_species("thalassoma")
        assert [r.name for r in results] == ["Thalassoma lunare"]

    def test_get_stats(self, session_factory, seed_taxa):
        seed_taxa(TAXA)
        with session_factory() as s:
            stats = TaxaService(s).get_stats()
        assert (stats.species, stats.genera, stats.families) == (3, 2, 2)


# ─── PredictionService: pure helpers (no model inference) ─────────────────────


class TestPredictionHelpers:
    def test_create_bbox_pixels_to_percent_xywh(self, session_factory):
        with session_factory() as s:
            svc = PredictionService(s, None, None)
            box = {"x1": 10, "y1": 20, "x2": 60, "y2": 120, "w": 200, "h": 400}
            bbox = svc._create_bbox(box)
            assert (bbox.x, bbox.y, bbox.w, bbox.h) == (5.0, 5.0, 25.0, 25.0)

    def test_sort_aligns_by_index_ascending(self, session_factory):
        with session_factory() as s:
            svc = PredictionService(s, None, None)
            idx, conf = svc.sort([3, 1, 2], ["a", "b", "c"])
            assert idx == (1, 2, 3)
            assert conf == ("b", "c", "a")

    def test_get_candidates_sorted_by_conf_desc(self, session_factory, seed_taxa):
        seed_taxa(TAXA)
        with session_factory() as s:
            svc = PredictionService(s, None, None)
            result = {
                "species": [0, 1, 2], "species_prob": [0.1, 0.9, 0.5],
                "genus": [0, 1], "genus_prob": [0.3, 0.7],
                "family": [0, 1], "family_prob": [0.4, 0.6],
            }
            species, genus, family = svc.get_candidates(result)
            assert species[0].conf == 0.9  # top guess first
            assert species[0].conf >= species[1].conf >= species[2].conf


def _index_of(taxon_id: int) -> int:
    """Map a seeded taxon_id back to its zero_index for ObservationCreate."""
    return {t["taxon_id"]: t["zero_index"] for t in TAXA}[taxon_id]
