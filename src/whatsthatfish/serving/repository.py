from sqlalchemy.orm import Session
from sqlalchemy import select, asc

from ..database.models import AppTaxa, InatTaxa


class TaxaRepository:
    def __init__(self, session):
        self.session = session

    def query_species(self):
        """Return the species catalogue: one entry per distinct trained species."""

        query = select(AppTaxa)
        return self.session.execute(query).scalars().all()

    def query_species_candidate(self, candidates: list[int]):

        query = select(
            AppTaxa.species,
            AppTaxa.zero_indexed_species,
            AppTaxa.description,
            AppTaxa.common_name,
            AppTaxa.location,
        ).where(AppTaxa.zero_indexed_species.in_(candidates)).order_by(asc(AppTaxa.zero_indexed_species))

        return self.session.execute(query).all()

    def query_genus_candidate(self, candidates: list[int]):

        query = select(AppTaxa.genus, AppTaxa.zero_indexed_genus).where(
            AppTaxa.zero_indexed_genus.in_(candidates)
        ).order_by(asc(AppTaxa.zero_indexed_genus))

        return self.session.execute(query).all()

    def query_family_candidate(self, candidates: list[int]):

        query = select(AppTaxa.family, AppTaxa.zero_indexed_family).where(
            AppTaxa.zero_indexed_family.in_(candidates)
        ).order_by(asc(AppTaxa.zero_indexed_family))

        return self.session.execute(query).all()
