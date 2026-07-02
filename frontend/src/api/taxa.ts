/* Correction picker source — searches rank='species' fish/shark taxa.
   Public reference data (no auth), mapped to camelCase at the seam. */
import { API_BASE } from "./config";

export interface TaxonOption {
  taxonId: number;
  name: string;
  commonName: string | null;
}

export async function searchSpecies(q: string): Promise<TaxonOption[]> {
  const res = await fetch(`${API_BASE}/taxa/species?q=${encodeURIComponent(q)}`);
  if (!res.ok) throw new Error(`GET /taxa/species failed: ${res.status}`);
  const data = await res.json();
  return data.map((t: any) => ({
    taxonId: t.taxon_id,
    name: t.name,
    commonName: t.common_name,
  }));
}
