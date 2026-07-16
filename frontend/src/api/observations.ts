/* Observation tracking client — POST/PATCH dives, observations, photos.
   All routes are auth-gated; authedFetch attaches the token and turns a 401 into
   a clear "session expired" error. snake_case ↔ camelCase mapping lives here. */
import { API_BASE } from "./config";
import { authedFetch, raiseForStatus } from "./http";

export interface DiveSpecies {
  taxonId: number;
  name: string | null;
  commonName: string | null;
}

/** Extended dive-log fields. Stored METRIC-canonical (m, °C, kg, bar); the UI
    converts to the user's unit_system at the edge (lib/units.ts). */
export interface DiveLogFields {
  visibilityM: number | null;
  airTempC: number | null;
  waterTempC: number | null;
  weightKg: number | null;
  exposureSuit: string | null;
  depthAvgM: number | null;
  depthMaxM: number | null;
  startedAt: string | null;
  bottomTimeMin: number | null;
  totalTimeMin: number | null;
  endPressureBar: number | null;
  diveShop: string | null;
}

export interface Dive extends DiveLogFields {
  id: string;
  siteId: string | null;
  siteName: string | null;
  gpsLat: number | null;
  gpsLng: number | null;
  divedAt: string | null;
  notes: string | null;
  verified: boolean;
  verifiedSource: string | null;
  createdAt: string;
  observationCount: number;
  species: DiveSpecies[];
}

interface DiveWire {
  id: string;
  site_id: string | null;
  site_name: string | null;
  gps_lat: number | null;
  gps_lng: number | null;
  dived_at: string | null;
  notes: string | null;
  visibility_m: number | null;
  air_temp_c: number | null;
  water_temp_c: number | null;
  weight_kg: number | null;
  exposure_suit: string | null;
  depth_avg_m: number | null;
  depth_max_m: number | null;
  started_at: string | null;
  bottom_time_min: number | null;
  total_time_min: number | null;
  end_pressure_bar: number | null;
  dive_shop: string | null;
  verified: boolean;
  verified_source: string | null;
  created_at: string;
  observation_count: number;
  species: { taxon_id: number; name: string | null; common_name: string | null }[];
}

const mapDive = (d: DiveWire): Dive => ({
  id: d.id,
  siteId: d.site_id,
  siteName: d.site_name,
  gpsLat: d.gps_lat,
  gpsLng: d.gps_lng,
  divedAt: d.dived_at,
  notes: d.notes,
  visibilityM: d.visibility_m,
  airTempC: d.air_temp_c,
  waterTempC: d.water_temp_c,
  weightKg: d.weight_kg,
  exposureSuit: d.exposure_suit,
  depthAvgM: d.depth_avg_m,
  depthMaxM: d.depth_max_m,
  startedAt: d.started_at,
  bottomTimeMin: d.bottom_time_min,
  totalTimeMin: d.total_time_min,
  endPressureBar: d.end_pressure_bar,
  diveShop: d.dive_shop,
  verified: d.verified ?? false,
  verifiedSource: d.verified_source,
  createdAt: d.created_at,
  observationCount: d.observation_count ?? 0,
  species: (d.species ?? []).map((s) => ({
    taxonId: s.taxon_id,
    name: s.name,
    commonName: s.common_name,
  })),
});

/** camelCase dive-log fields → the snake_case wire keys the backend expects.
    Used by create (send all) and update (send only provided). */
const DIVE_LOG_WIRE: Record<keyof DiveLogFields, string> = {
  visibilityM: "visibility_m",
  airTempC: "air_temp_c",
  waterTempC: "water_temp_c",
  weightKg: "weight_kg",
  exposureSuit: "exposure_suit",
  depthAvgM: "depth_avg_m",
  depthMaxM: "depth_max_m",
  startedAt: "started_at",
  bottomTimeMin: "bottom_time_min",
  totalTimeMin: "total_time_min",
  endPressureBar: "end_pressure_bar",
  diveShop: "dive_shop",
};

export interface SiteOption {
  id: string;
  name: string;
}

export async function searchSites(q: string): Promise<SiteOption[]> {
  const res = await authedFetch(`${API_BASE}/dive_sites?q=${encodeURIComponent(q)}`);
  await raiseForStatus(res, "Couldn't search dive sites");
  return (await res.json()).map((s: { id: string; name: string }) => ({
    id: s.id,
    name: s.name,
  }));
}

export async function listDives(): Promise<Dive[]> {
  const res = await authedFetch(`${API_BASE}/dives`);
  await raiseForStatus(res, "Couldn't load your dives");
  return (await res.json()).map(mapDive);
}

export async function createDive(
  body: {
    siteName?: string;
    googlePlaceId?: string | null;
    divedAt?: string;
    gpsLat?: number | null;
    gpsLng?: number | null;
    notes?: string;
  } & Partial<DiveLogFields>,
): Promise<Dive> {
  const payload: Record<string, unknown> = {
    site_name: body.siteName ?? null,
    google_place_id: body.googlePlaceId ?? null,
    dived_at: body.divedAt ?? null,
    gps_lat: body.gpsLat ?? null,
    gps_lng: body.gpsLng ?? null,
    notes: body.notes ?? null,
  };
  for (const [cam, snake] of Object.entries(DIVE_LOG_WIRE)) {
    const v = (body as Record<string, unknown>)[cam];
    if (v !== undefined) payload[snake] = v ?? null;
  }
  const res = await authedFetch(`${API_BASE}/dives`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await raiseForStatus(res, "Couldn't create the dive");
  return mapDive(await res.json());
}

export async function updateDive(
  id: string,
  body: {
    siteName?: string;
    googlePlaceId?: string | null;
    gpsLat?: number | null;
    gpsLng?: number | null;
    divedAt?: string;
    notes?: string;
  } & Partial<DiveLogFields>,
): Promise<Dive> {
  const payload: Record<string, unknown> = {};
  if (body.siteName !== undefined) payload.site_name = body.siteName;
  if (body.googlePlaceId !== undefined) payload.google_place_id = body.googlePlaceId;
  if (body.gpsLat !== undefined) payload.gps_lat = body.gpsLat;
  if (body.gpsLng !== undefined) payload.gps_lng = body.gpsLng;
  if (body.divedAt !== undefined) payload.dived_at = body.divedAt || null;
  if (body.notes !== undefined) payload.notes = body.notes;
  for (const [cam, snake] of Object.entries(DIVE_LOG_WIRE)) {
    const v = (body as Record<string, unknown>)[cam];
    if (v !== undefined) payload[snake] = v;
  }
  const res = await authedFetch(`${API_BASE}/dives/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await raiseForStatus(res, "Couldn't update the dive");
  return mapDive(await res.json());
}

export type LabelStatus = "predicted" | "confirmed" | "corrected";

export async function updateObservation(
  id: string,
  body: { correctedTaxonId?: number; labelStatus?: LabelStatus; depthM?: number | null },
): Promise<void> {
  // Only the keys present are sent (backend uses exclude_unset for PATCH).
  const payload: Record<string, unknown> = {};
  if (body.correctedTaxonId !== undefined) payload.corrected_taxon_id = body.correctedTaxonId;
  if (body.labelStatus !== undefined) payload.label_status = body.labelStatus;
  if (body.depthM !== undefined) payload.depth_m = body.depthM;
  const res = await authedFetch(`${API_BASE}/observations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await raiseForStatus(res, "Couldn't update the sighting");
}

export async function createObservation(body: {
  diveId: string;
  predictedSpeciesIndex: number;
  correctedSpeciesIndex?: number | null;
  correctedTaxonId?: number | null;
  labelStatus: LabelStatus;
  confidence?: number | null;
  depthM?: number | null;
}): Promise<{ id: string }> {
  const res = await authedFetch(`${API_BASE}/observations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dive_id: body.diveId,
      predicted_species_index: body.predictedSpeciesIndex,
      corrected_species_index: body.correctedSpeciesIndex ?? null,
      corrected_taxon_id: body.correctedTaxonId ?? null,
      label_status: body.labelStatus,
      confidence: body.confidence ?? null,
      depth_m: body.depthM ?? null,
    }),
  });
  await raiseForStatus(res, "Couldn't save the observation");
  return await res.json();
}

export async function deleteObservation(id: string): Promise<void> {
  const res = await authedFetch(`${API_BASE}/observations/${id}`, { method: "DELETE" });
  await raiseForStatus(res, "Couldn't delete the sighting");
}

export async function deleteDive(id: string): Promise<void> {
  const res = await authedFetch(`${API_BASE}/dives/${id}`, { method: "DELETE" });
  await raiseForStatus(res, "Couldn't delete the dive");
}

export async function deletePhoto(id: string): Promise<void> {
  const res = await authedFetch(`${API_BASE}/observation_photos/${id}`, {
    method: "DELETE",
  });
  await raiseForStatus(res, "Couldn't delete the photo");
}

/** Mark a photo as the card/hero image for its effective species (clears any
    prior hero for that species). */
export async function setHeroPhoto(id: string): Promise<void> {
  const res = await authedFetch(`${API_BASE}/observation_photos/${id}/hero`, {
    method: "POST",
  });
  await raiseForStatus(res, "Couldn't set the card image");
}

export async function uploadPhoto(body: {
  observationId: string;
  image: Blob;
  bbox?: object | null;
  predictedSpeciesIndex?: number | null;
  confidence?: number | null;
}): Promise<{ id: string }> {
  const fd = new FormData();
  fd.append("observation_id", body.observationId);
  fd.append("img", body.image, "photo.jpg");
  if (body.bbox) fd.append("bbox", JSON.stringify(body.bbox));
  if (body.predictedSpeciesIndex != null)
    fd.append("predicted_species_index", String(body.predictedSpeciesIndex));
  if (body.confidence != null) fd.append("confidence", String(body.confidence));
  // NB: no Content-Type — the browser sets the multipart boundary.
  const res = await authedFetch(`${API_BASE}/observation_photos`, {
    method: "POST",
    body: fd,
  });
  await raiseForStatus(res, "Couldn't upload the photo");
  return await res.json();
}
