"""Preseed the dive_regions geographic hierarchy (continent → country → area).

Source of truth: a Natural Earth admin_0 countries file (shapefile or GeoJSON)
with NAME / ISO_A2 / CONTINENT attributes — the SAME file
`derive_species_regions` uses for point-in-polygon, so region ids line up.
Continents and countries come from the file; a curated set of famous dive areas
are added as children of their country. Idempotent (safe to re-run).

Prereq: place the Natural Earth admin_0 file at the --source path (default
data/geo/ne_10m_admin_0_countries.shp). geopandas/shapely are train extras.

CLI: python -m whatsthatfish.etl.seed_regions [--source PATH]
"""

import argparse

import geopandas as gpd
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ..database import get_session_factory
from ..database.models import DiveRegion
from ..config import _get_logger
from ..serving.db.repository import _site_key  # reuse the dedup normalizer

logger = _get_logger(__name__)

# Curated dive areas → parent country ISO 3166-1 alpha-2. Extend freely.
_DIVE_AREAS = [
    ("Great Barrier Reef", "AU"),
    ("Red Sea (Egypt)", "EG"),
    ("Bonaire National Marine Park", "BQ"),
    ("Blue Hole (Belize)", "BZ"),
    ("Tubbataha Reefs", "PH"),
    ("Raja Ampat", "ID"),
    ("Cocos Island", "CR"),
    ("Galapagos", "EC"),
    ("Sipadan", "MY"),
    ("Palau", "PW"),
]
_SKIP_CONTINENTS = {"Antarctica", "Seven seas (open ocean)"}


class SeedRegions:
    """Idempotent preseed of the dive_regions hierarchy from Natural Earth."""

    def __init__(self, session_factory: sessionmaker, source: str):
        self.session = session_factory
        self.source = source

    def _get_or_create(
        self, session, kind, name, parent_id, iso=None, lat=None, lng=None
    ) -> DiveRegion:
        """Fetch-or-insert a region. Dedup scoped to (name_key, kind, parent).
        Continents (parent NULL) can't use the unique constraint (NULLs are
        distinct in PG), so this explicit lookup is what keeps re-runs idempotent."""
        key = _site_key(name)
        stmt = select(DiveRegion).where(
            DiveRegion.name_key == key, DiveRegion.kind == kind
        )
        stmt = (
            stmt.where(DiveRegion.parent_id.is_(None))
            if parent_id is None
            else stmt.where(DiveRegion.parent_id == parent_id)
        )
        region = session.execute(stmt).scalar_one_or_none()
        if region is None:
            region = DiveRegion(
                kind=kind,
                name=name,
                name_key=key,
                parent_id=parent_id,
                iso_country=iso,
                lat=lat,
                lng=lng,
            )
            session.add(region)
            session.flush()
        return region

    def run(self) -> None:
        gdf = gpd.read_file(self.source).reset_index(drop=True)
        # Project → centroid → back to lat/lng (avoids the geographic-CRS
        # centroid warning and gives a usable point per country).
        centroids = gdf.to_crs(3857).geometry.centroid.to_crs(4326)

        with self.session() as session:
            continents = {
                cname: self._get_or_create(session, "continent", cname, None)
                for cname in sorted(set(gdf["CONTINENT"]) - _SKIP_CONTINENTS)
            }
            for i, row in gdf.iterrows():
                cont = row["CONTINENT"]
                if cont in _SKIP_CONTINENTS:
                    continue
                iso = str(row.get("ISO_A2") or "").strip()
                iso = iso if iso and iso != "-99" else None
                self._get_or_create(
                    session,
                    "country",
                    row["NAME"],
                    continents[cont].id,
                    iso=iso,
                    lat=float(centroids.iloc[i].y),
                    lng=float(centroids.iloc[i].x),
                )

            country_by_iso = {
                r.iso_country: r
                for r in session.execute(
                    select(DiveRegion).where(DiveRegion.kind == "country")
                )
                .scalars()
                .all()
                if r.iso_country
            }
            for area_name, iso in _DIVE_AREAS:
                parent = country_by_iso.get(iso)
                if parent is None:
                    logger.warning(
                        "no country row for area %s (iso %s); skipping",
                        area_name,
                        iso,
                    )
                    continue
                self._get_or_create(
                    session, "area", area_name, parent.id, iso=iso
                )
            session.commit()
        logger.info("Region seed complete.")


def main() -> None:
    p = argparse.ArgumentParser(description="Preseed the dive_regions hierarchy")
    p.add_argument(
        "--source",
        default="data/geo/ne_10m_admin_0_countries.shp",
        help="Natural Earth admin_0 shapefile/GeoJSON (NAME/ISO_A2/CONTINENT)",
    )
    args = p.parse_args()
    SeedRegions(get_session_factory(), args.source).run()


if __name__ == "__main__":
    main()
