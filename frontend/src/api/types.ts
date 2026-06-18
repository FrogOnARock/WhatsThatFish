/* Domain types — the single source of truth for the UI ↔ backend contract.
   These formalise the shapes documented in client.ts: the two-stage pipeline
   (YOLO bbox → CustomResnet's species/genus/family heads). */

/** Bounding box as PERCENT of rendered image dims (serving converts YOLO
    x1y1x2y2 pixels → %). */
export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** One ranked guess from a classifier head; `conf` is 0–100. */
export interface Candidate {
  name: string;
  conf: number;
}

/** A single identification result. `bbox` is null when the detector finds no fish. */
export interface Prediction {
  summary: string;
  common: string;
  bbox: BBox | null;
  habitat: string;
  species: Candidate[];
  genus: Candidate[];
  family: Candidate[];
}

/** Keys of a Prediction that hold a top-3 candidate list (one per head). */
export type TaxonKey = "species" | "genus" | "family";

/** A demo fish shown on the drop zone. */
export interface SampleFish {
  id: string;
  label: string;
  hue: number;
  caption: string;
}

/** A single logged encounter of a species. */
export interface Sighting {
  date: string;
  site: string;
  region: string;
  depth: number;
  tempC: number;
  conf: number;
}

/** A collected species entry in the field log. */
export interface LogSpecies {
  no: number;
  id: string;
  common: string;
  species: string;
  genus: string;
  family: string;
  hue: number;
  caption: string;
  bestConf: number;
  habitat: string;
  sightings: Sighting[];
}

/** An undiscovered-species placeholder slot in the field log. */
export interface Ghost {
  no: number;
  hint: string;
}

/** Payload returned by getFieldLog(). */
export interface FieldLog {
  species: LogSpecies[];
  ghosts: Ghost[];
  totalSpecies: number;
}

/** One catalogue entry in the Species Library — the set of classes the model
    knows. camelCase here; the backend's /species returns snake_case, mapped at
    the client seam (api/species.ts). Names are SCIENTIFIC (no common names in
    the taxa schema). */
export interface SpeciesEntry {
  speciesId: number;   // zero-indexed class id (stable)
  name: string;        // scientific species name
  genus: string;
  family: string;
  imageCount: number;
  // Not yet provided by GET /species — optional until the backend supplies them.
  // When added, populate them in the api/species.ts mapping.
  common: string;
  description: string;
  location: string[];
  filename: string;
  depth: string;
}

