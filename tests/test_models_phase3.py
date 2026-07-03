"""Phase 3 ORM tests (P2) — constraints + cascade behavior that the service
layer relies on: dive-site uniqueness, the two FKs to inat_taxa, server defaults,
and user-deletion cascading through dives → observations → photos.
"""

import pytest
from sqlalchemy.exc import IntegrityError

from whatsthatfish.database.models import (
    User,
    DiveSite,
    Dive,
    Observation,
    ObservationPhoto,
)

TAXA = [
    {
        "taxon_id": 1001,
        "zero_index": 0,
        "species": "Amphiprion ocellaris",
        "genus": "Amphiprion",
        "family": "Pomacentridae",
    },
    {
        "taxon_id": 2001,
        "zero_index": 1,
        "species": "Thalassoma lunare",
        "genus": "Thalassoma",
        "family": "Labridae",
    },
]


class TestConstraints:
    def test_dive_site_name_key_unique(self, session_factory):
        with session_factory() as s:
            s.add(DiveSite(name="Site A", name_key="dup"))
            s.add(DiveSite(name="Site B", name_key="dup"))
            with pytest.raises(IntegrityError):
                s.commit()

    def test_observation_label_status_server_default(
        self, session_factory, seed_taxa, make_user
    ):
        seed_taxa(TAXA)
        user = make_user()
        with session_factory() as s:
            dive = Dive(user_id=user.id)
            s.add(dive)
            s.flush()
            obs = Observation(
                dive_id=dive.id,
                user_id=user.id,
                predicted_taxon_id=1001,
                corrected_taxon_id=1001,
            )
            s.add(obs)
            s.commit()
            s.refresh(obs)
            assert obs.label_status == "predicted"  # server_default

    def test_two_taxon_fks_resolve_independently(
        self, session_factory, seed_taxa, make_user
    ):
        seed_taxa(TAXA)
        user = make_user()
        with session_factory() as s:
            dive = Dive(user_id=user.id)
            s.add(dive)
            s.flush()
            obs = Observation(
                dive_id=dive.id,
                user_id=user.id,
                predicted_taxon_id=1001,
                corrected_taxon_id=2001,
            )
            s.add(obs)
            s.commit()
            assert obs.predicted_taxon.taxon_id == 1001
            assert obs.corrected_taxon.taxon_id == 2001

    def test_corrected_taxon_id_is_not_nullable(
        self, session_factory, seed_taxa, make_user
    ):
        seed_taxa(TAXA)
        user = make_user()
        with session_factory() as s:
            dive = Dive(user_id=user.id)
            s.add(dive)
            s.flush()
            s.add(
                Observation(dive_id=dive.id, user_id=user.id, predicted_taxon_id=1001)
            )
            with pytest.raises(IntegrityError):
                s.commit()


class TestCascade:
    def test_delete_user_cascades_to_dives_obs_photos(
        self, session_factory, seed_taxa, make_user
    ):
        seed_taxa(TAXA)
        user = make_user()
        with session_factory() as s:
            dive = Dive(user_id=user.id)
            s.add(dive)
            s.flush()
            obs = Observation(
                dive_id=dive.id,
                user_id=user.id,
                predicted_taxon_id=1001,
                corrected_taxon_id=1001,
            )
            s.add(obs)
            s.flush()
            s.add(ObservationPhoto(observation_id=obs.id, image_path="user/x.jpg"))
            s.commit()

        with session_factory() as s:
            s.delete(s.get(User, user.id))
            s.commit()

        with session_factory() as s:
            assert s.query(Dive).count() == 0
            assert s.query(Observation).count() == 0
            assert s.query(ObservationPhoto).count() == 0

    def test_delete_dive_cascades_to_observations(
        self, session_factory, seed_taxa, make_user
    ):
        seed_taxa(TAXA)
        user = make_user()
        with session_factory() as s:
            dive = Dive(user_id=user.id)
            s.add(dive)
            s.flush()
            dive_id = dive.id
            s.add(
                Observation(
                    dive_id=dive.id,
                    user_id=user.id,
                    predicted_taxon_id=1001,
                    corrected_taxon_id=1001,
                )
            )
            s.commit()

        with session_factory() as s:
            s.delete(s.get(Dive, dive_id))
            s.commit()

        with session_factory() as s:
            assert s.query(Observation).count() == 0
