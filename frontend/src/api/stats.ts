/* Trained-class counts from GET /stats — replaces hardcoded UI numbers.
   Wire shape is already flat ints with matching names, so no mapping needed. */
import { API_BASE } from "./config";

export interface ModelStats {
  species: number;
  genera: number;
  families: number;
}

export async function getStats(): Promise<ModelStats> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error(`GET /stats failed: ${res.status}`);
  return res.json();
}
