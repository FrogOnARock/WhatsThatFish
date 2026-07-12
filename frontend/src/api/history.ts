/* Field log client — GET /history (auth-gated), mapped to camelCase.
   Species are grouped by effective taxon (corrected_taxon_id) on the server. */
import { API_BASE } from "./config";
import { authedFetch, raiseForStatus } from "./http";

export interface LogPhoto {
  id: string;
  bbox: { x: number; y: number; w: number; h: number } | null;
  width: number | null;
  height: number | null;
  /** User-chosen card image for the species; UI falls back to the first photo. */
  isHero: boolean;
}

export interface FieldSighting {
  observationId: string;
  diveId: string;
  divedAt: string | null;
  siteName: string | null;
  depthM: number | null;
  labelStatus: "predicted" | "confirmed" | "corrected";
  photos: LogPhoto[];
}

export interface FieldSpecies {
  taxonId: number;
  species: string | null;
  genus: string | null;
  family: string | null;
  commonName: string | null;
  sightingCount: number;
  sightings: FieldSighting[];
}

export async function getFieldLog(): Promise<FieldSpecies[]> {
  const res = await authedFetch(`${API_BASE}/history`);
  await raiseForStatus(res, "Couldn't load your field log");
  const data = await res.json();
  return data.species.map((s: any) => ({
    taxonId: s.taxon_id,
    species: s.species,
    genus: s.genus,
    family: s.family,
    commonName: s.common_name,
    sightingCount: s.sighting_count,
    sightings: s.sightings.map((g: any) => ({
      observationId: g.observation_id,
      diveId: g.dive_id,
      divedAt: g.dived_at,
      siteName: g.site_name,
      depthM: g.depth_m,
      labelStatus: g.label_status,
      photos: g.photos.map((p: any) => ({
        id: p.id,
        bbox: p.bbox,
        width: p.width,
        height: p.height,
        isHero: p.is_hero ?? false,
      })),
    })),
  }));
}

/** Endpoint for a contribution photo. Needs the Bearer token, so render it via
    AuthedImage (fetch → blob), not a bare <img src>. */
export function photoImageEndpoint(photoId: string): string {
  return `${API_BASE}/observation_photos/${photoId}/image`;
}
