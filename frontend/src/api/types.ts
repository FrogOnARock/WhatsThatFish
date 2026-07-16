/* Domain types — the single source of truth for the UI ↔ backend contract.
   These formalise the shapes documented in client.ts: the two-stage pipeline
   (YOLO bbox → CustomResnet's species/genus/family heads). */

/** The signed-in user. camelCase here; the backend's /auth/me returns snake_case
    (display_name / avatar_url), mapped at the client seam (api/auth.ts). */
export type UnitSystem = "metric" | "imperial";

export interface UserProfile {
  id: string;
  email: string | null;
  displayName: string | null;
  avatarUrl: string | null;
  /** App-owned override for the Google name; not clobbered by login sync. */
  preferredName: string | null;
  unitSystem: UnitSystem;
}

/** Bounding box as PERCENT of the original image dims — top-left (x, y) plus
    width/height (serving converts the detector's x1y1x2y2 pixels → xywh %).
    Maps 1:1 to CSS left/top/width/height. */
export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** One ranked guess from a classifier head; `conf` is 0–100. */
export interface Candidate {
  name: string;
  index: number;
  conf: number;
  summary: string;
  common: string;
  habitat: string[]
}

/** A single identification result. When `detected` is false the detector found
    no fish: `bbox` is empty, but the classifier still ran on the full frame, so
    the candidate lists are populated (low-trust, out-of-distribution). */
export interface Prediction {
  bbox: BBox[];
  species: Candidate[];
  genus: Candidate[];
  family: Candidate[];
  detected: boolean;
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
  regions: Region[];
  filename: string;
  depth: string;
}

/** A geographic region (continent / country / dive-area) — a species range or a
    dive site's location. `parentId` links up the hierarchy. */
export interface Region {
  id: string;
  name: string;
  kind: "continent" | "country" | "area";
  parentId: string | null;
}

