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

export interface Dive {
  id: string;
  siteId: string | null;
  siteName: string | null;
  gpsLat: number | null;
  gpsLng: number | null;
  divedAt: string | null;
  notes: string | null;
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
  createdAt: d.created_at,
  observationCount: d.observation_count ?? 0,
  species: (d.species ?? []).map((s) => ({
    taxonId: s.taxon_id,
    name: s.name,
    commonName: s.common_name,
  })),
});

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

export async function createDive(body: {
  siteName?: string;
  divedAt?: string;
  gpsLat?: number | null;
  gpsLng?: number | null;
  notes?: string;
}): Promise<Dive> {
  const res = await authedFetch(`${API_BASE}/dives`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      site_name: body.siteName ?? null,
      dived_at: body.divedAt ?? null,
      gps_lat: body.gpsLat ?? null,
      gps_lng: body.gpsLng ?? null,
      notes: body.notes ?? null,
    }),
  });
  await raiseForStatus(res, "Couldn't create the dive");
  return mapDive(await res.json());
}

export async function updateDive(
  id: string,
  body: { siteName?: string; divedAt?: string; notes?: string },
): Promise<Dive> {
  const payload: Record<string, unknown> = {};
  if (body.siteName !== undefined) payload.site_name = body.siteName;
  if (body.divedAt !== undefined) payload.dived_at = body.divedAt || null;
  if (body.notes !== undefined) payload.notes = body.notes;
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
