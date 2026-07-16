"""Backfill species_regions from observation coordinates (point-in-polygon).

`inat_classification_dataset` carries per-photo lat/lng for the trained
catalogue. This maps each observation to the country it falls in (Natural Earth
admin_0), aggregates distinct countries per taxon, and upserts
(taxon_id, region_id, obs_count) into species_regions with source='observation'.
Free + data-driven — no LLM. Requires `seed_regions` to have run first (country
rows must exist with iso_country populated).

CLI: python -m whatsthatfish.preprocessing.derive_species_regions [--source PATH]
"""

import argparse

import geopandas as gpd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker

from ..database import get_session_factory
from ..database.models import InatClassificationDataset, DiveRegion, SpeciesRegion
from ..config import _get_logger

logger = _get_logger(__name__)


class DeriveSpeciesRegions:
    """Point-in-polygon backfill of species ranges from observation coords."""

    def __init__(self, session_factory: sessionmaker, source: str):
        self.session = session_factory
        self.source = source

    def _observations(self):
        with self.session() as s:
            return s.execute(
                select(
                    InatClassificationDataset.taxon_id,
                    InatClassificationDataset.latitude,
                    InatClassificationDataset.longitude,
                ).where(
                    InatClassificationDataset.taxon_id.isnot(None),
                    InatClassificationDataset.latitude.isnot(None),
                    InatClassificationDataset.longitude.isnot(None),
                )
            ).all()

    def run(self) -> None:
        rows = self._observations()
        if not rows:
            logger.info("No observation coordinates found; nothing to derive.")
            return

        countries = gpd.read_file(self.source)[["ISO_A2", "geometry"]].to_crs(
            "EPSG:4326"
        )
        points = gpd.GeoDataFrame(
            {"taxon_id": [r.taxon_id for r in rows]},
            geometry=gpd.points_from_xy(
                [r.longitude for r in rows], [r.latitude for r in rows]
            ),
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(points, countries, predicate="within")
        counts = (
            joined.groupby(["taxon_id", "ISO_A2"]).size().reset_index(name="obs_count")
        )

        with self.session() as session:
            iso_to_region = {
                r.iso_country: r.id
                for r in session.execute(
                    select(DiveRegion.id, DiveRegion.iso_country).where(
                        DiveRegion.kind == "country",
                        DiveRegion.iso_country.isnot(None),
                    )
                ).all()
            }
            payload = []
            for _, row in counts.iterrows():
                region_id = iso_to_region.get(row["ISO_A2"])
                if region_id is None:
                    continue  # a country with observations but no seeded region
                payload.append(
                    {
                        "taxon_id": int(row["taxon_id"]),
                        "region_id": region_id,
                        "source": "observation",
                        "obs_count": int(row["obs_count"]),
                    }
                )
            if payload:
                stmt = insert(SpeciesRegion).values(payload)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        SpeciesRegion.taxon_id,
                        SpeciesRegion.region_id,
                    ],
                    set_={
                        "obs_count": stmt.excluded.obs_count,
                        "source": stmt.excluded.source,
                    },
                )
                session.execute(stmt)
                session.commit()
            logger.info("Upserted %d species-region links.", len(payload))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Derive species_regions from observation coordinates"
    )
    p.add_argument(
        "--source",
        default="data/geo/ne_10m_admin_0_countries.shp",
        help="Natural Earth admin_0 shapefile/GeoJSON (must match seed_regions)",
    )
    args = p.parse_args()
    DeriveSpeciesRegions(get_session_factory(), args.source).run()


if __name__ == "__main__":
    main()
