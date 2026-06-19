from .repository import TaxaRepository
from .schemas import (
    SpeciesEntry,
    SpeciesInfo,
    SpeciesCatalogue,
    Prediction,
    Bbox,
    Candidate,
)
from ..inference.bbox_inference import BoundingBoxInference
from ..inference.class_inference import ClassInference


class TaxaService:
    def __init__(self, session):
        self.repo = TaxaRepository(session)

    def get_species(self):

        rows = self.repo.query_species()

        return [
            SpeciesEntry(
                species_id=row.zero_indexed_species,
                name=row.species,
                genus=row.genus,
                family=row.family,
                image_count=row.img_count,
                description=row.description,
                filename=row.filename,
                common_name=row.common_name,
                location=row.location,
                depth=row.depth,
            )
            for row in rows
        ]


class PredictionService:
    def __init__(self, session, bbox_inferrer, class_inferrer):
        self.repo = TaxaRepository(session)
        self.bbox_inferrer = bbox_inferrer
        self.class_inferrer = class_inferrer

    def _create_candidate(self, candidate) -> Candidate:
        return Candidate(
            name=candidate.get("name", None),
            index=candidate.get("zero_index", None),
            conf=candidate.get("conf", None),
            summary=candidate.get("summary", None),
            common=candidate.get("common", None),
            habitat=candidate.get("habitat", None),
        )

    def _create_bbox(self, box) -> Bbox:
        return Bbox(x=box["x1"], y=box["y1"], x2=box["x2"], y2=box["y2"])

    def query_species_candidate(
        self, candidates: list[int], conf: list[float]
    ) -> list[Candidate]:

        rows = self.repo.query_species_candidate(candidates)

        candidates = [
            {
                "name": row.species,
                "zero_index": row.zero_indexed_species,
                "summary": row.description,
                "common": row.common_name,
                "habitat": row.location,
            }
            for row in rows
        ]

        for i in range(len(candidates)):
            candidates[i]["conf"] = conf[i]

        return [self._create_candidate(candidate) for candidate in candidates]

    def query_genus_candidate(
        self, candidates: list[int], conf: list[float]
    ) -> list[Candidate]:

        rows = self.repo.query_genus_candidate(candidates)

        candidates = [
            {
                "name": row.genus,
                "zero_index": row.zero_indexed_genus,
            }
            for row in rows
        ]

        for i in range(len(candidates)):
            candidates[i]["conf"] = conf[i]

        return [self._create_candidate(candidate) for candidate in candidates]

    def query_family_candidate(
        self, candidates: list[int], conf: list[float]
    ) -> list[Candidate]:

        rows = self.repo.query_family_candidate(candidates)

        candidates = [
            {
                "name": row.genus,
                "zero_index": row.zero_indexed_genus,
            }
            for row in rows
        ]

        for i in range(len(candidates)):
            candidates[i]["conf"] = conf[i]

        return [self._create_candidate(candidate) for candidate in candidates]

    def get_candidates(self, result: dict):
        """Return the species catalogue: one entry per distinct trained species."""

        species_candidates = self.query_species_candidate(
            result["species"], result["species_prob"]
        )
        genus_candidates = self.query_genus_candidate(
            result["genus"], result["genus_prob"]
        )
        family_candidates = self.query_family_candidate(
            result["family"], result["family_prob"]
        )

        return species_candidates, genus_candidates, family_candidates

    def get_prediction(self, img_batch: bytes | list[bytes]):

        try:
            bbox_results = self.bbox_inferrer.infer(img_batch)
        except:
            raise Exception

        try:
            class_results = self.class_inferrer.infer(img_batch, bbox_results)
        except:
            raise Exception

        species_list = []
        genus_list = []
        family_list = []
        for result in class_results:

            species, species_prob = zip(*sorted(zip(result["species"], result["species_prob"]), key=lambda x: x[0]))
            genus, genus_prob = zip(*sorted(zip(result["genus"], result["genus_prob"]), key=lambda x: x[0]))
            family, family_prob = zip(*sorted(zip(result["family"], result["family_prob"]), key=lambda x: x[0]))

            result["species"] = species
            result["species_prob"] = species_prob

            result["genus"] = genus
            result["genus_prob"] = genus_prob

            result["family"] = family
            result["family_prob"] = family_prob

            species_can, genus_can, family_can = self.get_candidates(result=result)
            species_list.extend(species_can)
            genus_list.extend(genus_can)
            family_list.extend(family_can)

        bbox = [self._create_bbox(box) for box in bbox_results]

        return Prediction(bbox=bbox, species=species, genus=genus, family=family)
