from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import select, asc, desc, or_, func

from whatsthatfish.database.models import (
    AppTaxa,
    InatTaxa,
    User,
    Dive,
    DiveSite,
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
        """Search rank='species' taxa (fish + sharks) by scientific name, joined
        to app_taxa for a common name where the species was in the trained set."""
        stmt = (
            select(InatTaxa.taxon_id, InatTaxa.name, AppTaxa.common_name)
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
            stmt = stmt.where(InatTaxa.name.ilike(f"%{q.strip()}%"))
        return self.session.execute(stmt).all()


class ObservationRepository:
    """Reads/writes for the dive → observation → photo tree. Every read is
    ownership-scoped by user_id; the denormalized user_id makes that a single
    filter without joining through the dive."""

    def __init__(self, session):
        self.session = session

    # ── dive sites (deduplicated named locations) ──────────────────────────
    def resolve_or_create_site(self, name: str, user_id: UUID) -> DiveSite:
        key = _site_key(name)
        site = self.session.execute(
            select(DiveSite).where(DiveSite.name_key == key)
        ).scalar_one_or_none()
        if site is None:
            site = DiveSite(
                name=_proper_case(name), name_key=key, created_by_user_id=user_id
            )
            self.session.add(site)
            self.session.flush()
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
