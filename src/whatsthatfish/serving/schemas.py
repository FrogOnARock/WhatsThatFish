"""Pydantic response models — the serving-layer contract.

These mirror the frontend's `src/api/types.ts`. Keep the two in sync: a field
renamed here must be renamed there (and vice versa), since the SPA deserialises
exactly these shapes.
"""

from pydantic import BaseModel, Field, HttpUrl


class SpeciesEntry(BaseModel):
    """One row in the Species Library catalogue.

    Names are SCIENTIFIC (iNat `inat_taxa.name`); the schema has no common
    names. `species_id` is the zero-indexed class id — the stable identifier the
    classifier's species head emits, and what the frontend keys cards on.
    """

    species_id: int
    name: str
    genus: str
    family: str
    image_count: int
    common_name: str
    description: str
    location: list[str]
    filename: str
    depth: str

class SpeciesCatalogue(BaseModel):
    """Envelope so we can add catalogue-level metadata without reshaping the list."""

    species: list[SpeciesEntry]
    total: int


class SpeciesInfo(BaseModel):
    """The LLM-enriched fields for a species — common name, blurb, range, depth.

    The field descriptions double as the prompt contract the LLM fills in.
    """

    common_name: str = Field(description="Common English name, e.g. 'Clownfish'")
    description: str = Field(description="A 100 word description of the species")
    location: list[str] = Field(
        description="The most common location(s) for that species"
    )
    depth: str = Field(
        description="The average depth that species is found at in comma separated meters and feet metrics.  e.g., '1-5 meters, 0-15 feet'"
    )

