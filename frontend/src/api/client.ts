/* API client — the single seam between the UI and the model backend.
   The UI imports ONLY from this file; swap the re-export below from
   mock → real fetch implementation and no component changes.

   Contract (matches the two-stage pipeline: YOLO bbox → CustomResnet heads) —
   see ./types for the concrete shapes:

   Prediction = {
     bbox: { x, y, w, h } | null,   // PERCENT of image dims (serving layer
                                    // converts YOLO x1y1x2y2 pixels → %);
                                    // null when the detector finds no fish
     species:   [{ name, conf }, ...3],   // conf in 0–100
     genus:     [{ name, conf }, ...3],
     family: [{ name, conf }, ...3],   // CustomResnet's 3rd (coarsest)
                                          // head is family — the UI
                                          // labels the card accordingly
     common:  string,               // common name of top species
     summary: string,               // short display title
     habitat: string,               // habitat/range blurb
   }

   TODO(you): when the FastAPI/Cloud Run endpoint exists, implement here:
     export async function identify(file: File): Promise<Prediction> {
       const body = new FormData();
       body.append("image", file);
       const res = await fetch(`${API_BASE}/identify`, { method: "POST", body });
       ...map server response → Prediction (incl. pixel bbox → percent)...
     }
   Decide: how to surface `underwater_confidence` and the no-detection /
   coral-negative cases the prototype never designed. */

export {
  identify,
  identifySample,
  getFieldLog,
  SAMPLE_FISH,
  DEFAULT_PREDICTION_KEY,
} from "./mock";

// First REAL backend call — not from ./mock. Lives in ./species (fetches FastAPI).
export { getSpeciesLibrary } from "./species";

export type {
  BBox,
  Candidate,
  Prediction,
  TaxonKey,
  SampleFish,
  Sighting,
  LogSpecies,
  Ghost,
  FieldLog,
  SpeciesEntry,
} from "./types";
