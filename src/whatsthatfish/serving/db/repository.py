from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy import select, asc, desc, or_, func, update

from whatsthatfish.database.models import (
    AppTaxa,
    InatTaxa,
    User,
    Dive,
    DiveSite,
    DiveRegion,
    SpeciesRegion,
    Observation,
    ObservationPhoto,
)


def _site_key(name: str) -> str:
    """Normalized dedup key: collapse whitespace, lowercase."""
    return " ".join(name.split()).lower()


def _proper_case(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split())


class TaxaRepository:
    def __init__(self, session):
        self.session = session

    def query_species(self):
        """Return the species catalogue: one entry per distinct trained species."""

        query = select(AppTaxa)
        return self.session.execute(query).scalars().all()

    def query_species_candidate(self, candidates: list[int]):

        query = (
            select(
                AppTaxa.species,
                AppTaxa.zero_indexed_species,
                AppTaxa.description,
                AppTaxa.common_name,
                AppTaxa.location,
            )
            .where(AppTaxa.zero_indexed_species.in_(candidates))
            .order_by(asc(AppTaxa.zero_indexed_species))
        )

        return self.session.execute(query).all()

    def query_genus_candidate(self, candidates: list[int]):

        query = (
            select(AppTaxa.genus, AppTaxa.zero_indexed_genus)
            .distinct()
            .where(AppTaxa.zero_indexed_genus.in_(candidates))
            .order_by(asc(AppTaxa.zero_indexed_genus))
        )

        return self.session.execute(query).all()

    def query_family_candidate(self, candidates: list[int]):

        query = (
            select(AppTaxa.family, AppTaxa.zero_indexed_family)
            .distinct()
            .where(AppTaxa.zero_indexed_family.in_(candidates))
            .order_by(asc(AppTaxa.zero_indexed_family))
        )

        return self.session.execute(query).all()

    def get_counts(self) -> dict[str, int]:
        """Distinct trained class counts straight from app_taxa (the source of
        truth for 'how many species/genera/families the model knows')."""
        row = self.session.execute(
            select(
                func.count(func.distinct(AppTaxa.zero_indexed_species)),
                func.count(func.distinct(AppTaxa.zero_indexed_genus)),
                func.count(func.distinct(AppTaxa.zero_indexed_family)),
            )
        ).one()
        return {"species": row[0], "genera": row[1], "families": row[2]}

    def species_index_to_taxon(self, indices: list[int]) -> dict[int, int]:
        """Map the classifier's zero-index species ids → stable iNat taxon_ids.
        This is the translation done at save time so history never persists the
        drift-prone zero-index."""
        rows = self.session.execute(
            select(AppTaxa.zero_indexed_species, AppTaxa.taxon_id).where(
                AppTaxa.zero_indexed_species.in_(indices)
            )
        ).all()
        return {r.zero_indexed_species: r.taxon_id for r in rows}

    def taxa_display(self, taxon_ids: list[int]):
        """Rich display fields per taxon_id for the field-log cards. Trained taxa
        live in app_taxa; corrections to untrained taxa would need an inat_taxa
        fallback (deferred with the taxa picker)."""
        if not taxon_ids:
            return {}
        rows = self.session.execute(
            select(
                AppTaxa.taxon_id,
                AppTaxa.species,
                AppTaxa.genus,
                AppTaxa.family,
                AppTaxa.common_name,
            ).where(AppTaxa.taxon_id.in_(taxon_ids))
        ).all()
        return {r.taxon_id: r for r in rows}

    # Class/phylum ancestors that scope the correction picker to ray-finned
    # fishes + sharks/rays (excludes corals). Species sit deep in the tree, so
    # these always appear mid-path → slash-delimited LIKE is collision-safe.
    _FISH_ANCESTORS = ("%/47178/%", "%/196614/%")  # Actinopterygii, Chondrichthyes

    def search_species(self, q: str, limit: int = 25):
        """Search rank='species' taxa (fish + sharks) by scientific OR common name.

        Common name is COALESCE(app_taxa, inat_taxa): app_taxa carries the trained
        set's curated names today, inat_taxa is the seeded superset — coalescing
        both the selected value and the filter means common-name search widens to
        the full taxonomy automatically once inat_taxa.common_name is seeded."""
        common_name = func.coalesce(AppTaxa.common_name, InatTaxa.common_name).label(
            "common_name"
        )
        stmt = (
            select(InatTaxa.taxon_id, InatTaxa.name, common_name)
            .outerjoin(AppTaxa, AppTaxa.taxon_id == InatTaxa.taxon_id)
            .where(
                InatTaxa.rank == "species",
                InatTaxa.active.is_(True),
                or_(*[InatTaxa.ancestry.like(p) for p in self._FISH_ANCESTORS]),
            )
            .order_by(InatTaxa.name)
            .limit(limit)
        )
        if q.strip():
            term = f"%{q.strip()}%"
            stmt = stmt.where(
                or_(
                    InatTaxa.name.ilike(term),
                    AppTaxa.common_name.ilike(term),
                    InatTaxa.common_name.ilike(term),
                )
            )
        return self.session.execute(stmt).all()

    def regions_for_taxa(self, taxon_ids: list[int]) -> dict[int, list]:
        """Map taxon_id → its regions (species_regions ⨝ dive_regions), for the
        species library's structured 'where to find it'. Ordered continent →
        country → area (alpha within kind — 'area'<'continent'<'country' isn't
        semantic, so callers that need strict hierarchy sort by kind rank)."""
        if not taxon_ids:
            return {}
        rows = self.session.execute(
            select(
                SpeciesRegion.taxon_id,
                DiveRegion.id,
                DiveRegion.name,
                DiveRegion.kind,
                DiveRegion.parent_id,
            )
            .join(DiveRegion, DiveRegion.id == SpeciesRegion.region_id)
            .where(SpeciesRegion.taxon_id.in_(taxon_ids))
            .order_by(DiveRegion.name)
        ).all()
        out: dict[int, list] = {}
        for r in rows:
            out.setdefault(r.taxon_id, []).append(r)
        return out


class ObservationRepository:
    """Reads/writes for the dive → observation → photo tree. Every read is
    ownership-scoped by user_id; the denormalized user_id makes that a single
    filter without joining through the dive."""

    def __init__(self, session):
        self.session = session

    # ── dive sites (deduplicated named locations) ──────────────────────────
    def resolve_or_create_site(
        self,
        name: str,
        user_id: UUID,
        google_place_id: str | None = None,
        lat: float | None = None,
        lng: float | None = None,
    ) -> DiveSite:
        key = _site_key(name)
        site = self.session.execute(
            select(DiveSite).where(DiveSite.name_key == key)
        ).scalar_one_or_none()
        if site is None:
            site = DiveSite(
                name=_proper_case(name),
                name_key=key,
                created_by_user_id=user_id,
                google_place_id=google_place_id,
                lat=lat,
                lng=lng,
            )
            self.session.add(site)
            self.session.flush()
        else:
            # Backfill place id / coords when a later visit resolves the same site
            # through Places Autocomplete and we didn't have them before.
            if google_place_id and not site.google_place_id:
                site.google_place_id = google_place_id
            if lat is not None and site.lat is None:
                site.lat = lat
            if lng is not None and site.lng is None:
                site.lng = lng
        return site

    def search_sites(self, q: str, limit: int = 10):
        return (
            self.session.execute(
                select(DiveSite)
                .where(DiveSite.name_key.like(f"%{_site_key(q)}%"))
                .limit(limit)
            )
            .scalars()
            .all()
        )

    # ── dives ──────────────────────────────────────────────────────────────
    def create_dive(self, user_id: UUID, **fields) -> Dive:
        dive = Dive(user_id=user_id, **fields)
        self.session.add(dive)
        self.session.commit()
        self.session.refresh(dive)
        return dive

    def get_dive(self, user_id: UUID, dive_id: UUID) -> Dive | None:
        return self.session.execute(
            select(Dive).where(Dive.id == dive_id, Dive.user_id == user_id)
        ).scalar_one_or_none()

    def list_dives(self, user_id: UUID):
        return (
            self.session.execute(
                select(Dive)
                .where(Dive.user_id == user_id)
                .options(
                    joinedload(Dive.site),
                    selectinload(Dive.observations),
                )
                .order_by(desc(Dive.dived_at))
            )
            .unique()
            .scalars()
            .all()
        )

    def save(self):
        """Commit pending changes (e.g. after a dive PATCH)."""
        self.session.commit()

    def delete(self, obj) -> None:
        """Delete an ORM row and commit. Child rows (observations → photos) are
        removed by the ON DELETE CASCADE / delete-orphan cascade on the mapper;
        the caller is responsible for removing the corresponding storage blobs
        BEFORE calling this (the image_path is gone once the row is)."""
        self.session.delete(obj)
        self.session.commit()

    def dive_image_paths(self, user_id: UUID, dive_id: UUID) -> list[str]:
        """Every contribution-photo storage key under a dive (all its
        observations' photos), so the blobs can be removed before the cascade
        delete drops the rows. Ownership-scoped via the join to the dive."""
        rows = self.session.execute(
            select(ObservationPhoto.image_path)
            .join(Observation, ObservationPhoto.observation_id == Observation.id)
            .join(Dive, Observation.dive_id == Dive.id)
            .where(Dive.id == dive_id, Dive.user_id == user_id)
        ).all()
        return [r[0] for r in rows]

    # ── observations ─────────────────────────────────────────────────────────
    def create_observation(self, user_id: UUID, **fields) -> Observation:
        obs = Observation(user_id=user_id, **fields)
        self.session.add(obs)
        self.session.commit()
        self.session.refresh(obs)
        return obs

    def get_observation(self, user_id: UUID, obs_id: UUID) -> Observation | None:
        return (
            self.session.execute(
                select(Observation)
                .where(Observation.id == obs_id, Observation.user_id == user_id)
                .options(
                    selectinload(Observation.photos),
                    joinedload(Observation.dive).joinedload(Dive.site),
                )
            )
            .unique()
            .scalar_one_or_none()
        )

    # ── photos ───────────────────────────────────────────────────────────────
    def create_photo(self, **fields) -> ObservationPhoto:
        photo = ObservationPhoto(**fields)
        self.session.add(photo)
        self.session.commit()
        self.session.refresh(photo)
        return photo

    def get_photo(self, user_id: UUID, photo_id: UUID) -> ObservationPhoto | None:
        """Ownership-scoped via the join to the owning observation."""
        return self.session.execute(
            select(ObservationPhoto)
            .join(Observation, ObservationPhoto.observation_id == Observation.id)
            .where(ObservationPhoto.id == photo_id, Observation.user_id == user_id)
        ).scalar_one_or_none()

    def set_hero(self, user_id: UUID, photo_id: UUID) -> bool:
        """Make `photo_id` the hero (card image) for its effective species,
        clearing any prior hero among that user's photos of the same
        corrected_taxon_id. Returns False if the photo isn't found/owned."""
        photo = self.get_photo(user_id, photo_id)
        if photo is None:
            return False
        taxon_id = self.session.scalar(
            select(Observation.corrected_taxon_id).where(
                Observation.id == photo.observation_id
            )
        )
        # Ids of every photo this user has under the same effective taxon.
        sibling_ids = (
            select(ObservationPhoto.id)
            .join(Observation, ObservationPhoto.observation_id == Observation.id)
            .where(
                Observation.user_id == user_id,
                Observation.corrected_taxon_id == taxon_id,
            )
        )
        self.session.execute(
            update(ObservationPhoto)
            .where(ObservationPhoto.id.in_(sibling_ids))
            .values(is_hero=False)
            .execution_options(synchronize_session=False)
        )
        photo.is_hero = True
        self.session.commit()
        return True

    def user_stats(self, user_id: UUID) -> dict[str, int]:
        """Summary counts for the Settings page: dives, observations, and
        distinct effective species (corrected_taxon_id)."""
        dives = self.session.scalar(
            select(func.count()).select_from(Dive).where(Dive.user_id == user_id)
        )
        observations = self.session.scalar(
            select(func.count())
            .select_from(Observation)
            .where(Observation.user_id == user_id)
        )
        unique_species = self.session.scalar(
            select(func.count(func.distinct(Observation.corrected_taxon_id))).where(
                Observation.user_id == user_id
            )
        )
        return {
            "dives": dives or 0,
            "observations": observations or 0,
            "unique_species": unique_species or 0,
        }

    # ── history (all of a user's observations, eager-loaded for grouping) ────
    def list_user_observations(self, user_id: UUID):
        return (
            self.session.execute(
                select(Observation)
                .where(Observation.user_id == user_id)
                .options(
                    joinedload(Observation.dive).joinedload(Dive.site),
                    selectinload(Observation.photos),
                )
                .order_by(desc(Observation.created_at))
            )
            .unique()
            .scalars()
            .all()
        )


class UserRepository:
    def __init__(self, session):
        self.session = session

    def get_by_subject(self, subject_id: str) -> User | None:
        """Look up a user by their Google OIDC subject (the stable account id)."""
        return self.session.execute(
            select(User).where(User.google_subject_id == subject_id)
        ).scalar_one_or_none()

    def save(self):
        """Commit pending changes (e.g. after a settings PATCH)."""
        self.session.commit()

    def upsert_from_claims(self, claims: dict) -> User:
        """Create the user on first sign-in, else refresh profile + last_login.

        Keyed on the OIDC `sub` (immutable per Google account); email/name/avatar
        are mirrored each login since the user may have changed them upstream.
        """
        user = self.get_by_subject(claims["sub"])
        if user is None:
            user = User(google_subject_id=claims["sub"])
            self.session.add(user)
        user.email = claims.get("email")
        user.display_name = claims.get("name")
        user.avatar_url = claims.get("picture")
        user.last_login_at = datetime.now(timezone.utc)
        self.session.commit()
        self.session.refresh(user)
        return user
