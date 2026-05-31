from readline import insert_text

from tenacity import retry_base

from ..database.models import InatClassificationDataset
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from ..config import _get_logger

logger = _get_logger(__name__)
class ZeroIndexClassification:

    def __init__(self, session_factory: sessionmaker):
        self.session = session_factory

    def _select_classification_taxa(self):
        logger.info("Retrieving classification subfamily, genus, species.")
        with self.session() as session:
            rows = session.execute(select(
                InatClassificationDataset.photo_uuid, InatClassificationDataset.subfamily, InatClassificationDataset.genus, InatClassificationDataset.species)
            )

        return [{"photo_uuid": r.photo_uuid, "subfamily": r.subfamily, "genus": r.genus, "species": r.species} for r in rows]

    def _zero_index(self, rows):
        logger.info("Zero indexing subfamily, genus, species for load to inat_classification_dataset.")
        subfamily_list = list(set([r["subfamily"] for r in rows]))
        genus_list = list(set([r["genus"] for r in rows]))
        species_list = list(set([r["species"] for r in rows]))

        for r in rows:
            r["zero_index_subfamily"] = subfamily_list.index(r["subfamily"])
            r["zero_index_genus"] = genus_list.index(r["genus"])
            r["zero_index_species"] = species_list.index(r["species"])

        return rows


    def _update_classification_dataset(self, rows):
        logger.info("Loading: zero indexed classes to inat_classification_dataset.")
        rows_to_insert = [{"photo_uuid": r["photo_uuid"],
                           "zero_indexed_subfamily": r["zero_index_subfamily"],
                           "zero_indexed_genus": r["zero_index_genus"],
                           "zero_indexed_species": r["zero_index_species"]} for r in rows]

        with self.session() as session:
            stmt = insert(InatClassificationDataset).values(rows_to_insert)
            insert_stmt = stmt.on_conflict_do_update(
                index_elements=[InatClassificationDataset.photo_uuid],
                set_={
                    "zero_indexed_subfamily": stmt.excluded.zero_indexed_subfamily,
                    "zero_indexed_genus": stmt.excluded.zero_indexed_genus,
                    "zero_indexed_species": stmt.excluded.zero_indexed_species
                }
            )
            session.execute(insert_stmt)
            session.commit()


    def run(self):
        rows = self._select_classification_taxa()
        zero_indexed = self._zero_index(rows)
        self._update_classification_dataset(zero_indexed)







