/* Species Library data — the first REAL backend call (no mock).
   Hits FastAPI's GET /species and maps the snake_case wire shape → the
   camelCase SpeciesEntry the UI uses. This mapping IS the anti-corruption
   boundary: backend field renames are absorbed here, not in components. */
import type { SpeciesEntry } from "./types";
import { API_BASE } from "./config";
// Override at build/deploy time with VITE_API_BASE (e.g. the Cloud Run URL).
// Defaults to the local uvicorn dev server.

/** Raw wire shape from GET /species (snake_case, matches serving/schemas.py). */
interface SpeciesEntryWire {
  species_id: number;
  name: string;
  genus: string;
  family: string;
  image_count: number;
  common_name: string;
  description: string;
  location: string[];
  filename: string;
  depth: string;

}

interface SpeciesCatalogueWire {
  species: SpeciesEntryWire[];
  total: number;
}

export async function getSpeciesLibrary(): Promise<SpeciesEntry[]> {
  const res = await fetch(`${API_BASE}/species`);
  if (!res.ok) {
    throw new Error(`GET /species failed: ${res.status} ${res.statusText}`);
  }
  const data: SpeciesCatalogueWire = await res.json();
  return data.species.map((s) => ({
    speciesId: s.species_id,
    name: s.name,
    genus: s.genus,
    family: s.family,
    imageCount: s.image_count,
    common: s.common_name,
    description: s.description,
    location: s.location,
    filename: s.filename,
    depth: s.depth
  }));
}

